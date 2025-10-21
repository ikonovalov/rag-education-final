import streamlit as st

if "graph" not in st.session_state:
    from src.rag_pipeline import graph
    st.session_state.graph=graph


# Инициализация истории чата в session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Заголовок приложения
st.title("AI-Шеффф")

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
    graph_response = st.session_state.graph.invoke({"question": prompt})
    response = graph_response["answer"]

    # Добавление ответа ассистента в историю
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Отображение ответа ассистента
    with st.chat_message("assistant"):
        st.markdown(response)