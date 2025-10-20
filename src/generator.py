from langchain_gigachat import GigaChat

from src.utils import (giga_api_key, giga_api_scope)


class LLMGenerator:
    """Generator отвечает за работу с генеративной моделью"""

    def __init__(self, config=None):
        if config is None:
            config = []
        self.llm = GigaChat(
            credentials=giga_api_key,
            scope=giga_api_scope,
            model='GigaChat-2-Max',
            verify_ssl_certs=False,
            temperature=0.3,
        )

    def invoke(self, messages):
        return self.llm.invoke(messages)