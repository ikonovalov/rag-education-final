from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

chief_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Ты повар, который разбирается в любой кухне."
        "В ответе обязательно указывай строку (row)(номер рецепта), на которой ты нашел рецепт."
        #"В ответе обязательно укажи из каких вариантов ты выбирал."
        #"Все номера рецептов указывай в формате [номер рецепта]. Номер рецепта - это только числа. Пример:[123]"
        #"Отвечай только на тему приготовления еды. В остальных случаях извинись и скажи, что это не к тебе."
        "Для форматирования ответа надо применять markdown."
        "Ты можешь придумать свое блюдо, но только основываясь на информации из контекста с указанием источника."
        "Используй следующий контекст для ответа: {context}"
    ),
    HumanMessagePromptTemplate.from_template(
        "Пользовательский вопрос: {question}"
    )
])

rephrase_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        # v1
        #"Ты переводчик с русского на английский. Надо перевести запрос пользователя с русского на английский."
        #"Если это вопрос, то переделай его в утвердительную форму предложения."
        #"Кроме перевода ничего не делай."

        # v2 Q=Хочется чего-то острого, горячего с чили и свининой => RE_EN=spicy, hot, chili, pork
        "Ты анализируешь запрос клиента ресторана."
        "Постарайся выделить названия рецептов, компоненты продуктов, вкусовые качества, то есть все что связано с едой, остальные слова отбрось."
        "Результат переведи на английский."
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