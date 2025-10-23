import os

from flashrank import Ranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.tools import tool
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent

from src.bm25_store import BM25Store
from src.generator import LLMGenerator
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
bm25_retriever.k = bm25_retriever_top_k

# Combine
ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],  # можно bm25 попробовать через MultiQueryRetriever погонять
    weights=[ensemble_faiss_weight, ensemble_bm25_weight],
    id_key="row"  # from metadata
)

# and reranker and top_n
reranker = FlashrankRerank(client=Ranker(model_name=ranker_model))
reranker.top_n = flash_reranker_top_n
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=ensemble_retriever, name="Reranker"
)

model = LLMGenerator().model()


@tool
def cookbook(query: str):
    """
    Поиск в кулинарной книге
    Учти, что книга на английском, поэтому запросы клиента надо переводить на английский перед запросом
    """
    print(f"RAG Q: {query}")
    retrieved_docs = compression_retriever.invoke(query)
    for r in retrieved_docs:
        print(f"retrieve_hybrid: row={r.metadata['row']} => {r.metadata['title']}")
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return docs_content


agent = create_react_agent(
    model=model,
    tools=[cookbook],
    prompt=(
        "Ты повар, который разбирается в любой кухне."
        "Используй рецепты только из кулинарной книги для ответа. Если рецепта нет, то спроси клиента не хочет ли он чего-то еще?"
        "В ответе обязательно указывай строку (row)(номер рецепта), на которой ты нашел рецепт, и оригинальное название на английском"
        "С клиентом общайся на русский язык"
        "Для форматирования ответа надо применять markdown."
    ),
)
invoked = agent.invoke(
    input={"messages": [HumanMessage("Хочется чего-то острого, горячего с чили и свининой")]},
    config=RunnableConfig(
        callbacks=graph_callbacks
    )
)
print(invoked["messages"][-1].content)
