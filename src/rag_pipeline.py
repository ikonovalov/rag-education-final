import os

from flashrank import Ranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langfuse.langchain import CallbackHandler
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from src.bm25_store import BM25Store
from src.generator import LLMGenerator
from src.prompts import rephrase_prompt, chief_prompt
from src.vector_store import FAISSVectorStore

ranker_model = "ms-marco-MultiBERT-L-12"

bm25_retriever_top_k = 5
faiss_retriever_top_k = 5
flash_reranker_top_n = 3
ensemble_bm25_weight = 0.5
ensemble_faiss_weight = 0.5

graph_callbacks = []

if os.getenv("LANGFUSE_AUTH") is not None:
    graph_callbacks.append(CallbackHandler())

faiss_vector_store = FAISSVectorStore("data/faiss_store")
faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k": faiss_retriever_top_k})

bm25_store = BM25Store("data/bm25")
bm25_retriever = bm25_store.as_retriever()
bm25_retriever.k= bm25_retriever_top_k

# Combine
ensemble_retriever = EnsembleRetriever(
    retrievers = [faiss_retriever, bm25_retriever],     # можно bm25 попробовать через MultiQueryRetriever погонять
    weights=[ensemble_faiss_weight, ensemble_bm25_weight],
    id_key = "row"                                      # from metadata
)

# and reranker and top_n
reranker = FlashrankRerank(client=Ranker(model_name=ranker_model))
reranker.top_n= flash_reranker_top_n
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=ensemble_retriever,
)

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
    retrieved_docs = compression_retriever.invoke(state["rephrased"])
    for r in retrieved_docs:
        print(f"retrieve_hybrid: row={r.metadata['row']} => {r.metadata['title']}")
    return {"retrieved": [document for document in retrieved_docs]}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["retrieved"])
    messages = chief_prompt.invoke({"question": state["question"], "context": docs_content})
    answer =  generator.invoke(messages)
    return {"answer": answer.content}

graph_builder = StateGraph(State).add_sequence([rephrase, retrieve_hybrid, generate])
graph_builder.add_edge(START, "rephrase")
graph = graph_builder.compile()