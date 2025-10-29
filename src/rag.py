from flashrank import Ranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from src.bm25_store import BM25Store
from src.vector_store import FAISSVectorStore

ranker_model = "ms-marco-MultiBERT-L-12"

bm25_retriever_top_k = 5
faiss_retriever_top_k = 5
flash_reranker_top_n = 3
ensemble_bm25_weight = 0.5
ensemble_faiss_weight = 0.5

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