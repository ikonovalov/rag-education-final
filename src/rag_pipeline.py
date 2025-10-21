from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from src.bm25_store import BM25Store
from src.generator import LLMGenerator
from src.prompts import rephrase_prompt, chief_prompt
from src.vector_store import FAISSVectorStore

faiss_vector_store = FAISSVectorStore("data/faiss_store")
faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 10})

bm25_store = BM25Store("data/bm25")
bm25_retriever = bm25_store.as_retriever()
bm25_retriever.k=10

generator = LLMGenerator()

class State(TypedDict):
    question: str
    rephrased: str
    retrieved: List[Document]
    answer: str

def rephrase(state: State):
    messages = rephrase_prompt.invoke({"question": state["question"]})
    answer = generator.invoke(messages)
    print(f"rephrase: Q={state["question"]}, RE_EN={answer.content}")
    return {"rephrased": answer.content}

def retrieve_hybrid(state: State):
    ensemble = EnsembleRetriever(
        retrievers = [faiss_retriever, bm25_retriever], # можно bm25 попробовать через MultiQueryRetriever погонять
        weights=[0.5, 0.5],
        id_key = "row"
    )
    retrieved = ensemble.invoke(state["rephrased"])[:5]
    for r in retrieved:
        print(f"retrieve_hybrid: doc id={r.id} => {r.metadata['title']}")
    return {"retrieved": [document for document in retrieved]}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["retrieved"])
    messages = chief_prompt.invoke({"question": state["question"], "context": docs_content})
    answer =  generator.invoke(messages)
    return {"answer": answer.content}

graph_builder = StateGraph(State).add_sequence([rephrase, retrieve_hybrid, generate])
graph_builder.add_edge(START, "rephrase")
graph = graph_builder.compile()