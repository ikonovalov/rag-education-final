from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from src.bm25_store import BM25Store
from src.generator import LLMGenerator
from src.vector_store import FAISSVectorStore

faiss_vector_store = FAISSVectorStore("../data/faiss_store")
faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 10})

bm25_store = BM25Store("../data/bm25")
bm25_retriever = bm25_store.as_retriever()
bm25_retriever.k=10

generator = LLMGenerator()

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Ты повар, который разбирается в любой кухне."
        "Используй следующий контекст для ответа: {context}"
        "В ответе обязательно указывай строку (row), на которой ты нашел рецепт."
        "В ответе обязательно укажи из каких вариантов ты выбирал."
        "Отвечай только на тему приготовления еды. В остальных случаях извинись и скажи, что это не к тебе."
    ),
    HumanMessagePromptTemplate.from_template(
        "Пользовательский вопрос: {question}"
    )
])

prompt_rephrase = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Ты переводчик с русского на английский. Надо перевести запрос пользователя с русского на английский."
        "Если это вопрос, то переделай его в утвердительную форму предложения."
    ),
    HumanMessagePromptTemplate.from_template(
        "{question}"
    )
])

class State(TypedDict):
    question: str
    rephrased: str
    retrieved: List[Document]
    answer: str

def rephrase(state: State):
    messages = prompt_rephrase.invoke({"question": state["question"]})
    answer = generator.invoke(messages)
    print(f"Q: {state["question"]}, RE_EN: {answer.content}")
    return {"rephrased": answer.content}

def retrieve_hybrid(state: State):
    ensemble = EnsembleRetriever(
        retrievers = [faiss_retriever, bm25_retriever], # можно bm25 попробовать через MultiQueryRetriever погонять
        weights=[0.5, 0.5],
    )
    retrieved = ensemble.invoke(state["rephrased"])[:5]
    for r in retrieved:
        print(f"Doc id={r.id} => {r.metadata['title']}")
    return {"retrieved": [document for document in retrieved]}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["retrieved"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    answer =  generator.invoke(messages)
    return {"answer": answer.content}

graph_builder = StateGraph(State).add_sequence([rephrase, retrieve_hybrid, generate])
graph_builder.add_edge(START, "rephrase")
graph = graph_builder.compile()

response = graph.invoke({"question": "Найди самый рецепт из картошки и чего-то острого"})
print(response["answer"])

