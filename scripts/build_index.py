import random

from langchain_community.document_loaders.csv_loader import CSVLoader

from scripts.preprocess_data import cleanup, truncate_oversized, SZ_LIMIT
from src.vector_store import FAISSVectorStore

loader = CSVLoader(
    file_path="../data/raw/wikileaks/afg-war-diary/afg.csv",
    csv_args={
        "fieldnames": ["ID", "DATE", "TYPE", "CATEGORY", "TRACK_NUMBER", "CAPTION", "REPORT", "REGION", "ATTACK_ON",
                       "UNKNOWN_FLD1", "REPORTING_UNIT", "UNIT_NAME", "TYPE_OF_UNIT",
                       "FRIENDLY_WOUNDED", "FRIENDLY_KILLED", "HOST_NATION_WOUNDED", "HOST_NATION_KILLED",
                       "CIVILIAN WOUNDED", "CIVILIAN_KILLED", "ENEMY_WOUNDED", "ENEMY_KILLED", "ENEMY_DETAINED" ,
                       "MGRS", "LONGITUDE", "LATITUDE", "ORIGINATOR_GROUP", "UPDATED_BY_GROUP", "CCIR",
                       "SIGACT", "AFFILIATION", "DCOLOR", "CLASSIFICATION"],
    },
    source_column="ID" # используется как id в vector store
)
reports_docs = loader.load()
cleared_reports_docs = cleanup(reports_docs)


large_reports = [d for d in cleared_reports_docs if len(d.page_content) >= SZ_LIMIT]
normal_records = [d for d in cleared_reports_docs if len(d.page_content) < SZ_LIMIT]
print(f"Total: {len(reports_docs)}. Large(>{SZ_LIMIT}): {len(large_reports)}. Normal: {len(normal_records)}")


reports_docs = random.sample(large_reports, k=50)

vector_store = FAISSVectorStore()
rs = vector_store.add_documents(truncate_oversized(reports_docs))
vector_store.save("../data/faiss_store")
print(f"{len(rs)} vectors stored")


