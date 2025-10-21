from langchain_community.document_loaders.csv_loader import CSVLoader

from scripts.preprocess_data import cleanup, SZ_LIMIT, extract_meta_and_propagate
from src.bm25_store import BM25Store
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

reports_docs_short = normal_records
print(f"Documents count {len(reports_docs_short)}")

# Vectorization
vector_store = FAISSVectorStore()
vector_store.add_documents(reports_docs_short)
vector_store.save("../data/faiss_store")

# BM25
bm25_retriever = BM25Store()
bm25_retriever.add_documents(reports_docs_short)
bm25_retriever.save("../data/bm25")

print("Done")