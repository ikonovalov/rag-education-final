import os
from dataclasses import dataclass
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.graph import START, END, StateGraph, MessagesState
from ragas import evaluate
from ragas.metrics import LLMContextRecall, AnswerRelevancy, Faithfulness, FactualCorrectness, AnswerCorrectness, \
    ContextRelevance, ContextPrecision
from typing_extensions import List

from src.generator import LLMGenerator
from src.utils import RAGAS
from src.workflow_metrics import create_ragas_dataset, evaluator_embedding, evaluator_llm
from src.prompts import rephrase_prompt, chief_prompt
from src.rag import compression_retriever
from src.reference import reference12709

graph_callbacks = []
if os.getenv("LANGFUSE_AUTH") is not None:
    graph_callbacks.append(CallbackHandler())

generator = LLMGenerator()


class RagWorkflowState(MessagesState):
    question: str
    rephrased: str
    retrieved: List[Document]
    answer: str


def rephrase(state: RagWorkflowState):
    messages = rephrase_prompt.invoke({"question": state["question"]})
    answer = generator.invoke(messages)
    print(f"rephrase: Q={state["question"]}, RE_EN={answer.content}")
    return {"rephrased": answer.content}


def retrieve_hybrid(state: RagWorkflowState):
    retrieved_docs = compression_retriever.invoke(state["rephrased"])
    for r in retrieved_docs:
        print(
            f"retrieve_hybrid: row={r.metadata['row']} (Reciprocal Rerank Fusion (RRF) score {r.metadata['relevance_score']}) => {r.metadata['title']}")

    # ==============LLM-as-Judge============================
    prompt = PromptTemplate.from_template(
        "Оцени от 0 до 1, насколько этот контекст полезен для ответа на вопрос:\n"
        "Вопрос: {question}\n"
        "Контекст: {context}\n"
        "Оценка:"
    )

    class Eval(TypedDict):
        receipt_name: str
        props: str
        cons: str
        explanation: str
        score: float

    print("Evaluation:")
    for d in retrieved_docs:
        chain = prompt | generator.llm.with_structured_output(Eval)
        score = chain.invoke({"question": state["question"], "context": d.page_content})
        print(f"{score['receipt_name']} => {score['score']:.0%}")
        print(f"\tprops: {score['props']}")
        print(f"\tcons: {score['cons']}")
        print(f"\texplanation: {score['explanation']}")
    # ======================================================

    return {"retrieved": [document for document in retrieved_docs]}


def generate_answer(state: RagWorkflowState):
    docs_content = "\n\n".join(doc.page_content for doc in state["retrieved"])
    messages = chief_prompt.invoke({"question": state["question"], "context": docs_content})
    answer = generator.invoke(messages)
    return {"answer": answer.content}


def ragas(state: RagWorkflowState):
    if RAGAS == "OFF":
        print("\nRAGas disabled\n")
        return
    evaluation_dataset = create_ragas_dataset(state, reference12709)
    metrics = [
        LLMContextRecall(),  # Полнота retrieval
        ContextRelevance(),  # Насколько contexts полезны для ответа на query
        Faithfulness(),  # Верность контексту (нет галлюцинаций)
        ContextPrecision(),  # Доля релевантных контекстов среди возвращённых
        AnswerRelevancy(embeddings=evaluator_embedding),  # Релевантность ответа вопросу
        FactualCorrectness(),  # На точность фактов (Только факты)
        AnswerCorrectness(embeddings=evaluator_embedding),  # На точность фактов (Факты + семантика)
    ]
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        callbacks=graph_callbacks,
    )
    print(result)
    # df_scores = result.to_pandas()


graph_builder = StateGraph(RagWorkflowState)
graph_builder.add_node("rephrase", rephrase)
graph_builder.add_node("retrieve_hybrid", retrieve_hybrid)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("ragas", ragas)

graph_builder.add_edge(START, "rephrase")
graph_builder.add_edge("rephrase", "retrieve_hybrid")
graph_builder.add_edge("retrieve_hybrid", "generate_answer")
graph_builder.add_edge("generate_answer", "ragas")
graph_builder.add_edge("ragas", END)
graph = graph_builder.compile()

invoked = graph.invoke(
    input={"question": "Порекомендуй рецепт настойки на водке и ягод"},
    config=RunnableConfig(callbacks=graph_callbacks),
)

print("\n== FINAL " + "="*10)
print(invoked["answer"])
