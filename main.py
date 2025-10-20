import streamlit as st

# Заголовок приложения
st.title("Система 'Запрос-Ответ'")

# Создаем текстовое поле для ввода запроса
user_input = st.text_input("Введите ваш запрос:", "")

# Создаем кнопку для отправки запроса
if st.button("Отправить"):
    if user_input:
        # Здесь можно добавить логику обработки запроса, например, вызов API
        response = f"Вы ввели: {user_input}. Это пример ответа!"
        st.write("**Ответ:**")
        st.image("C:\\Users\\igor_\\IdeaProjects\\rag-education-final\\data\\raw\\pes12017000148\\Food Images\\Food Images\\3-ingredient-buttermilk-biscuits.jpg", caption="Ваше изображение", use_container_width =True)
        st.write(response)
    else:
        st.warning("Пожалуйста, введите запрос!")

# Инструкция для пользователя
st.markdown("""
### Инструкция:
1. Введите ваш запрос в текстовое поле.
2. Нажмите кнопку "Отправить".
3. Получите ответ ниже.
""")