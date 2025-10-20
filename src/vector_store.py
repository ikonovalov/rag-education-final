import os
from typing import List, Tuple

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_gigachat.embeddings import GigaChatEmbeddings

from src.utils import (giga_api_key, giga_api_scope)


class BaseVectorStore:

    def __init__(self, config=None):
        if config is None:
            config = []
        self.embedding_model = GigaChatEmbeddings(
            credentials=giga_api_key,
            scope=giga_api_scope,
            verify_ssl_certs=False,
            model="EmbeddingsGigaR" # так не влезало
        )

    def get_embedding_model(self):
        return self.embedding_model


class EphemeralVectorStore(BaseVectorStore):
    def __init__(self):
        super().__init__()
        from langchain_core.vectorstores import InMemoryVectorStore
        self.vector_store = InMemoryVectorStore(super().get_embedding_model())

    def add_documents(self, docs):
        return self.vector_store.add_documents(documents=docs)

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, folder=None, config=None):
        super().__init__(config)
        if folder is None:
            vector_size  = len(super().get_embedding_model().embed_query("SALT REPORT AS FOLLOWS"))
            #index = faiss.IndexFlatL2(vector_size)
            index = faiss.IndexFlatIP(vector_size) # Большие значения соответствуют более похожим векторам
            #index = faiss.IndexHNSWFlat(vector_size)
            self.vector_store = FAISS(
                embedding_function=super().get_embedding_model(),
                index=index,
                docstore= InMemoryDocstore(),
                index_to_docstore_id={},
            )
        else:
            if os.path.exists(folder):
                self.vector_store = FAISS.load_local(
                    folder, self.embedding_model, allow_dangerous_deserialization=True
                )
            else:
                raise Exception("Store not found")

    def add_documents(self, docs: List[Document]) -> List[str]:
        rows_as_id = [str(d.metadata["row"]) for d in docs]
        stored_ids = self.vector_store.add_documents(documents=docs, ids=rows_as_id)
        return stored_ids

    def similarity_search(self, query, k=3) -> List[Document]:
        return self.vector_store.similarity_search(query, k)

    def similarity_search_with_score(self, query, k=3) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k)

    def as_retriever(self):
        return self.vector_store.as_retriever()

    def save(self, folder="faiss_store"):
        self.vector_store.save_local(folder)







