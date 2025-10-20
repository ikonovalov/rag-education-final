from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from src.bm25_store import BM25Store
from src.generator import LLMGenerator
from src.vector_store import FAISSVectorStore

faiss_vector_store = FAISSVectorStore("../data/faiss_store")
faiss_retriever = faiss_vector_store.as_retriever()

bm25_store = BM25Store("../data/bm25")
bm25_retriever = bm25_store.as_retriever()

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

class State(TypedDict):
    question: str
    retrieved: List[Document]
    answer: str

def retrieve_vectored(state: State):
    ensemble = EnsembleRetriever(
        retrievers = [faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    retrieved = ensemble.invoke(state["question"])
    return {"retrieved": [document for document in retrieved]}

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

