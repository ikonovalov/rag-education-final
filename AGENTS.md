# AGENTS.md

Инструкции для AI-агентов, работающих с этим репозиторием.

## Обзор проекта

Репозиторий реализует кулинарную RAG-систему («Giga-шефф»): чат-бот,
отвечающий на вопросы о рецептах с помощью гибридного поиска (BM25 + FAISS)
по базе рецептов и генерации ответа через GigaChat. Оркестрация обработки
запроса построена на LangGraph, трассировка/метрики — через Langfuse.
Интерфейс — Streamlit-приложения.

## Структура репозитория

Ниже — фактический состав репозитория. Файл `structure.txt` в корне
**устарел** (описывает шаблонную структуру `retriever.py`, `rag_pipeline.py`,
`tests/`, которая не соответствует реальному составу файлов) и **не должен
использоваться** как источник истины о структуре проекта.

- `main_agent.py` — точка входа Streamlit-приложения на основе `src/agent.py`.
- `main_workflow.py` — точка входа Streamlit-приложения на основе
  LangGraph-графа (см. раздел «Известные несоответствия и риски» — импорт в
  этом файле не рабочий).
- `src/agent.py` — сборка агента и его callbacks.
- `src/workflow.py` — LangGraph-граф обработки запроса.
- `src/bm25_store.py`, `src/bm25_native_store.py` — хранилища/индексы BM25.
- `src/vector_store.py` — хранилище/индекс FAISS.
- `src/generator.py` — генерация ответа через GigaChat.
- `src/prompts.py` — промпты.
- `src/rag.py` — RAG-логика (объединение retrieval + generation).
- `src/reference.py` — вспомогательные справочные данные/утилиты.
- `src/utils.py` — общие утилиты.
- `src/workflow_metrics.py` — метрики для workflow.
- `scripts/download.py` — загрузка исходных данных.
- `scripts/preprocess_data.py` — предобработка данных.
- `scripts/build_index.py` — построение индексов (BM25/FAISS).
- `config/config.yaml` — см. раздел «Известные несоответствия и риски»
  (файл пустой).
- `data/raw/` — каталог для сырых данных (пустой на момент написания).
- `requirements.txt` — зависимости Python.
- `.env-example` — шаблон переменных окружения.

## Установка и запуск

```bash
pip install -r requirements.txt

cd scripts
python download.py
python build_index.py
cd ..

streamlit run main_agent.py
# или
streamlit run main_workflow.py
```

## Переменные окружения

Переменные заданы в `.env-example` (без реальных значений секретов):

- `GIGACHAT_API_KEY` — ключ доступа к GigaChat API.
- `GIGACHAT_API_SCOPE` — scope доступа к GigaChat API.
- `LANGFUSE_PUBLIC_KEY` — публичный ключ Langfuse.
- `LANGFUSE_SECRET_KEY` — секретный ключ Langfuse.
- `USER_AGENT` — значение User-Agent для исходящих HTTP-запросов.

Реальные значения ключей/токенов не публикуются нигде в документации —
только имена переменных.

## Известные несоответствия и риски

- **Нерабочий импорт в `main_workflow.py`**: файл импортирует
  `from src.rag_pipeline import (graph, graph_callbacks)`, но модуль
  `src/rag_pipeline.py` в репозитории отсутствует. Фактический аналог —
  `src/workflow.py`. Запуск `streamlit run main_workflow.py` в текущем
  состоянии репозитория завершится ошибкой импорта. Это известное
  несоответствие не исправлено в рамках документационных изменений.
- **Пустой `config/config.yaml`**: файл присутствует, но не содержит
  конфигурации (0 байт) и фактически не используется, несмотря на упоминания
  в README/`structure.txt`.
- **Устаревший `structure.txt`**: см. раздел «Структура репозитория» выше.

## Тесты, CI, линтеры

На момент написания документа в репозитории отсутствуют:

- директория `tests/` и какие-либо тесты;
- CI-конфигурация (workflows, pipelines);
- конфигурации линтеров/форматтеров (`pyproject.toml`, `setup.cfg`,
  `.flake8` и т. п.).

Не пытайтесь искать или запускать команды проверки качества кода — они не
настроены в этом репозитории.
