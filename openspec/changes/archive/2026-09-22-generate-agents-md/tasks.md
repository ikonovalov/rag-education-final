# Tasks: Создание AGENTS.md для репозитория master

## Phase 1: Подготовка содержания

- [x] T001 — Собрать актуальный список модулей `src/*.py` и `scripts/*.py` для
      раздела "Структура репозитория" (`src/`, `scripts/`)
- [x] T002 — Собрать список переменных окружения из `.env-example` без их
      значений (`.env-example`)

## Phase 2: Создание AGENTS.md

- [x] T003 — Создать файл `AGENTS.md` в корне репозитория `master` (`AGENTS.md`)
- [x] T004 — Написать раздел "Обзор проекта" (RAG для рецептов, гибридный
      поиск BM25+FAISS, LangGraph, GigaChat) (`AGENTS.md`)
- [x] T005 — Написать раздел "Структура репозитория" по фактическому составу
      файлов, с явной пометкой, что `structure.txt` устарел и не используется
      как источник истины (`AGENTS.md`)
- [x] T006 — Написать раздел "Установка и запуск": `pip install -r
      requirements.txt`, `cd scripts && python download.py && python
      build_index.py`, `streamlit run main_agent.py`, `streamlit run
      main_workflow.py` (`AGENTS.md`)
- [x] T007 — Написать раздел "Переменные окружения" с именами
      `GIGACHAT_API_KEY`, `GIGACHAT_API_SCOPE`, `LANGFUSE_PUBLIC_KEY`,
      `LANGFUSE_SECRET_KEY`, `USER_AGENT` без реальных значений (`AGENTS.md`)
- [x] T008 — Написать раздел "Известные несоответствия и риски": нерабочий
      импорт `src.rag_pipeline` в `main_workflow.py`, пустой
      `config/config.yaml` (`AGENTS.md`)
- [x] T009 — Написать раздел "Тесты, CI, линтеры" с явным указанием на их
      отсутствие на момент написания документа (`AGENTS.md`)

## Verification

- [x] T010 — Проверить, что файл создан: `test -f AGENTS.md`
- [x] T011 — Проверить, что документ упоминает все обязательные переменные
      окружения: `grep -c -E "GIGACHAT_API_KEY|GIGACHAT_API_SCOPE|LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY|USER_AGENT" AGENTS.md`
- [x] T012 — Проверить, что в документ не попали реальные значения секретов
      из `.env-example`: `! grep -q -F -f <(grep -oE '=.+' .env-example | sed "s/^=//") AGENTS.md`
- [x] T013 — Проверить, что исходный код репозитория не изменился:
      `git diff --stat -- main_agent.py main_workflow.py src scripts`
