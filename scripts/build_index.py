from langchain_community.document_loaders.csv_loader import CSVLoader

from scripts.preprocess_data import cleanup, truncate_oversized, SZ_LIMIT, extract_meta_and_propagate
from src.bm25_index_retriever import BM25IndexRetriever
from src.vector_store import FAISSVectorStore


datafile = "../data/raw/pes12017000148/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

loader = CSVLoader(
    file_path=datafile,
    encoding='utf-8',
)
reports_docs = loader.load()

extract_meta_and_propagate(reports_docs)
cleanup(reports_docs)

large_reports = [d for d in reports_docs if len(d.page_content) >= SZ_LIMIT]
normal_records = [d for d in reports_docs if len(d.page_content) < SZ_LIMIT]
print(f"Total: {len(reports_docs)}. Large(>{SZ_LIMIT}): {len(large_reports)}. Normal: {len(normal_records)}")

reports_docs_short = reports_docs[100:500]

# Vectorization
# vector_store = FAISSVectorStore()
# rs = vector_store.add_documents(reports_docs_short)
# vector_store.save("../data/faiss_store")
# print(reports_docs)
# print(f"{len(rs)} vectors stored")

# BM25
bm25_retriever = BM25IndexRetriever()
bm25_retriever.add_documents(reports_docs_short)
bm25_retriever.save("../data/bm25")

print("Done")