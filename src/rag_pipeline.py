from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate

from src.generator import Generator
from src.vector_store import FAISSVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document


vector_store = FAISSVectorStore("../data/faiss_store")
generator = Generator()

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

class State(TypedDict):
    question: str
    retrieved: List[Document]
    answer: str

def retrieve_vectored(state: State):
    retrieved_doc_and_scores = vector_store.similarity_search_with_score(state["question"])
    for d,s in retrieved_doc_and_scores:
        print(f"doc {d.metadata['row']} => {s:3f}")

    return {"retrieved": [document for document, _ in retrieved_doc_and_scores]}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["retrieved"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    answer =  generator.invoke(messages)
    return {"answer": answer.content}

graph_builder = StateGraph(State).add_sequence([retrieve_vectored, generate])
graph_builder.add_edge(START, "retrieve_vectored")
graph = graph_builder.compile()

response = graph.invoke({"question": "Найди самый простой рецепт из картошки и соли"})
print(response["answer"])

