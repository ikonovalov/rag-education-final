from typing import List
from typing_extensions import deprecated

import nltk
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def extract_field(tag: str, text: str) -> str:
    # Находим позицию маркера
    start_pos = text.find(tag) + len(tag)  # Смещаемся после маркера
    end_pos = text.find("\n", start_pos)  # Ищем перенос строки после маркера
    if start_pos> 0 and end_pos == -1: # скорее всего последний тэг
        end_pos = len(text)
    if start_pos != -1 and end_pos != -1:
        result = text[start_pos:end_pos]
        return result
    else:
        return ""

@deprecated
class BM25Store:
    def __init__(self, folder=None, language='english'):
        self.stop_words = set(nltk.corpus.stopwords.words(language))
        self.tokenizer = nltk.word_tokenize
        if folder is None:
            self.bm25 = None
        else:
            pass

    def preprocess(self, text):
        tokens = self.tokenizer(text.lower(), language='english')
        return [t for t in tokens if t.isalpha() and t not in self.stop_words]

    def add_documents(self, docs: List[Document]):
        tokenized_corpus = []
        for doc in docs:
            page_contend = doc.page_content
            title = extract_field("Title: ", page_contend)
            ingredients = extract_field("Cleaned_Ingredients: ", page_contend)
            instruction = "" #extract_field("Instructions", page_contend)
            text = f"{title} {ingredients} {instruction}"
            tokenized = self.preprocess(text)
            doc.metadata['tokenized'] = tokenized
            tokenized_corpus.append(tokenized)
        self.bm25 = BM25Okapi(tokenized_corpus)
        return docs

    def save(self, path: str):
        pass


    def query(self, query: str):
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        print(scores)
