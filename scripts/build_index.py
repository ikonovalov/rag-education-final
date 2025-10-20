import random

from langchain_community.document_loaders.csv_loader import CSVLoader

from scripts.preprocess_data import cleanup, truncate_oversized, SZ_LIMIT, extract_meta_and_propagate
from src.vector_store import FAISSVectorStore

loader = CSVLoader(
    file_path="../data/raw/pes12017000148/Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
    encoding='utf-8',
)
reports_docs = loader.load()

extract_meta_and_propagate(reports_docs)
cleanup(reports_docs)

large_reports = [d for d in reports_docs if len(d.page_content) >= SZ_LIMIT]
normal_records = [d for d in reports_docs if len(d.page_content) < SZ_LIMIT]
print(f"Total: {len(reports_docs)}. Large(>{SZ_LIMIT}): {len(large_reports)}. Normal: {len(normal_records)}")

reports_docs = random.sample(normal_records, k=200)

vector_store = FAISSVectorStore()
rs = vector_store.add_documents(reports_docs)
vector_store.save("../data/faiss_store")
print(reports_docs)
print(f"{len(rs)} vectors stored")


