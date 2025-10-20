from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from src.generator import Generator
from src.vector_store import FAISSVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document


vector_store = FAISSVectorStore("../data/faiss_store")
generator = Generator()

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Используй следующий контекст для ответа: {context}"
    ),
    HumanMessagePromptTemplate.from_template(
        "Пользовательский вопрос: {question}"
    )
])

class State(TypedDict):
    question: str
    retrieved: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"retrieved": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["retrieved"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    answer =  generator.invoke(messages)
    return {"answer": answer.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "Расскажи о случаях нападения на медицинские учереждения в Багдаде"})
print(response["answer"])