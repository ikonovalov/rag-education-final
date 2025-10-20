import os.path
import pickle
from typing import List

import nltk
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class BM25Store:
    def __init__(self, path = None, language = 'english', tokenizer = nltk.word_tokenize):
        self.language = language
        if path is not None:
            self.load(path=path)
        else:
            self.retriever = None
        self.tokenizer = tokenizer
        self.stop_words = set(nltk.corpus.stopwords.words(language))

    def add_documents(self, docs: List[Document]):
        self.retriever = BM25Retriever.from_documents(
            documents=docs,
            preprocess_func=self.tokenizer,
        )

    def as_retriever(self):
        return self.retriever

    def query(self, query: str) -> List[Document]:
        return  self.retriever.invoke(input=query)

    def _prepared_tokenization(self, text)-> List[str]:
        lower = text.lower()
        tokens = self.tokenizer(lower, language=self.language)
        filtered = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        return filtered

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(f"{path}/bm25_retriever.pkl", "wb") as f:
            pickle.dump(self.retriever, f)

    def load(self, path):
        with open(f"{path}/bm25_retriever.pkl", "rb") as f:
            self.retriever = pickle.load(f)