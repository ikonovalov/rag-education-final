import os
from typing import List

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from src.generator import LLMGenerator
from src.rag import compression_retriever

graph_callbacks = []
if os.getenv("LANGFUSE_AUTH") is not None:
    graph_callbacks.append(CallbackHandler())

model = LLMGenerator().model()

class ChiefAgentState(AgentState):
    retrieved: List[Document]


@tool
def cookbook(query: str):
    """
    Поиск в кулинарной книге
    Учти, что книга на английском, поэтому запросы клиента надо переводить на английский перед запросом
    """
    print(f"RAG Q: {query}")
    retrieved_docs = compression_retriever.invoke(query)
    for r in retrieved_docs:
        print(f"compression_retriever: row={r.metadata['row']} => {r.metadata['title']}")
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return docs_content


agent = create_react_agent(
    model=model,
    tools=[cookbook],
    state_schema=ChiefAgentState,
    prompt=(
        "Ты повар, который разбирается в любой кухне."
        "Используй рецепты только из кулинарной книги для ответа. Если рецепта нет, то спроси клиента не хочет ли он чего-то еще?"
        "В ответе обязательно указывай строку (row)(номер рецепта), на которой ты нашел рецепт, и оригинальное название на английском"
        "С клиентом общайся на русский язык"
        "Для форматирования ответа надо применять markdown."
    ),
)

invoked = agent.invoke(
    input = ChiefAgentState(
        messages=[HumanMessage("Хочется чего-то острого, горячего с чили и свининой")],
        retrieved = []
    ),
    config=RunnableConfig(
        callbacks=graph_callbacks
    )
)
print(invoked["messages"][-1].content)
