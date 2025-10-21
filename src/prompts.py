from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

chief_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Ты повар, который разбирается в любой кухне."
        "Используй следующий контекст для ответа: {context}"
        "В ответе обязательно указывай строку (row), на которой ты нашел рецепт."
        "В ответе обязательно укажи из каких вариантов ты выбирал."
        "Отвечай только на тему приготовления еды. В остальных случаях извинись и скажи, что это не к тебе."
    ),
    HumanMessagePromptTemplate.from_template(
        "Пользовательский вопрос: {question}"
    )
])

rephrase_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Ты переводчик с русского на английский. Надо перевести запрос пользователя с русского на английский."
        "Если это вопрос, то переделай его в утвердительную форму предложения."
    ),
    HumanMessagePromptTemplate.from_template(
        "{question}"
    )
])

search_image_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Дан список из названия рецепта и картинки блюда. "
        "Твоя задача по имени рецепта от пользователя определить имя для картинки"
        "Список:"
        "{list}"
        "В ответе укажи только имя файла и ничего больше"
    ),
    HumanMessagePromptTemplate.from_template(
        "Рецепт: {question}"
    )
])