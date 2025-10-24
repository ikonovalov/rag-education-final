import os.path
import re
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

if "graph" not in st.session_state:
    from src.agent import (agent, graph_callbacks)
    st.session_state.agent=agent
    st.session_state.agent_callbacks=graph_callbacks


# Инициализация истории чата в session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Заголовок приложения
st.title("Giga-шеффф +Ag")

# Отображение истории чата
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода для пользователя
if prompt := st.chat_input("Давай поговорим о еде"):
    # Добавление сообщения пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Отображение сообщения пользователя
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерация ответа (здесь простой эхо-ответ для примера)
    agent_response = st.session_state.agent.invoke(
        input={"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(
            callbacks=graph_callbacks
        ))
    ai_answer = agent_response["messages"][-1].content
    print(ai_answer)


    # Отображение ответа ассистента
    with st.chat_message("assistant"):
        st.markdown(ai_answer)

    # Добавление ответа ассистента в историю
    ai_message = {"role": "assistant", "content": ai_answer, "image": None}
    st.session_state.messages.append(ai_message)

