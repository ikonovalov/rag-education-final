import os.path
import re
from pathlib import Path

import streamlit as st

if "graph" not in st.session_state:
    from src.rag_pipeline import graph
    st.session_state.graph=graph


# Инициализация истории чата в session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Заголовок приложения
st.title("AI-Шеффф")

#st.image("C:\\Users\\igor_\\IdeaProjects\\rag-education-final\\data\\raw\\pes12017000148\\Food Images\\Food Images\\3-ingredient-buttermilk-biscuits.jpg", caption="Ваше изображение", use_container_width =True)

# Отображение истории чата
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
       # if message["image"] is not None:
       #     st.image(message["image"], use_container_width =True)

# Поле ввода для пользователя
if prompt := st.chat_input("Давай поговорим о еде"):
    # Добавление сообщения пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Отображение сообщения пользователя
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерация ответа (здесь простой эхо-ответ для примера)
    graph_response = st.session_state.graph.invoke({"question": prompt})
    ai_answer = graph_response["answer"]
    print(ai_answer)


    # Отображение ответа ассистента
    with st.chat_message("assistant"):
        st.markdown(ai_answer)

    id_n_image = [
        (
            doc.metadata['row'],
            f"{doc.metadata['image']}.jpg",
            re.search(str(doc.metadata['row']), ai_answer)
        )
        for doc in graph_response['retrieved']
    ]
    matched = [m for m in id_n_image if m[2] is not None]
    print(f"{id_n_image}")
    print(f"{matched}")
    img_path=None
    if len(matched) == 1:
        img_path = Path(os.getcwd()) / "data" / "raw" / "pes12017000148" / "Food Images" /  "Food Images" / str(matched[0][1])
        st.image(img_path, width='stretch')

    # Добавление ответа ассистента в историю
    ai_message = {"role": "assistant", "content": ai_answer, "image": img_path}
    st.session_state.messages.append(ai_message)

