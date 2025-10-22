from typing import List

from langchain_core.messages import BaseMessage
from langchain_gigachat import GigaChat

from src.utils import (GIGACHAT_API_KEY, GIGACHAT_API_SCOPE)


class LLMGenerator:
    """Generator отвечает за работу с генеративной моделью"""

    def __init__(self, config=None):
        if config is None:
            config = []
        self.llm = GigaChat(
            credentials=GIGACHAT_API_KEY,
            scope=GIGACHAT_API_SCOPE,
            model='GigaChat-2-Max',
            verify_ssl_certs=False,
            temperature=0.3,
        )

    def invoke(self, messages: List[BaseMessage]):
        return self.llm.invoke(messages)