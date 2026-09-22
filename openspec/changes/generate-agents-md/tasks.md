# Tasks: Создание AGENTS.md для репозитория master

## Phase 1: Подготовка содержания

- [ ] T001 — Собрать актуальный список модулей `src/*.py` и `scripts/*.py` для
      раздела "Структура репозитория" (`src/`, `scripts/`)
- [ ] T002 — Собрать список переменных окружения из `.env-example` без их
      значений (`.env-example`)

## Phase 2: Создание AGENTS.md

- [ ] T003 — Создать файл `AGENTS.md` в корне репозитория `master` (`AGENTS.md`)
- [ ] T004 — Написать раздел "Обзор проекта" (RAG для рецептов, гибридный
      поиск BM25+FAISS, LangGraph, GigaChat) (`AGENTS.md`)
- [ ] T005 — Написать раздел "Структура репозитория" по фактическому составу
      файлов, с явной пометкой, что `structure.txt` устарел и не используется
      как источник истины (`AGENTS.md`)
- [ ] T006 — Написать раздел "Установка и запуск": `pip install -r
      requirements.txt`, `cd scripts && python download.py && python
      build_index.py`, `streamlit run main_agent.py`, `streamlit run
      main_workflow.py` (`AGENTS.md`)
- [ ] T007 — Написать раздел "Переменные окружения" с именами
      `GIGACHAT_API_KEY`, `GIGACHAT_API_SCOPE`, `LANGFUSE_PUBLIC_KEY`,
      `LANGFUSE_SECRET_KEY`, `USER_AGENT` без реальных значений (`AGENTS.md`)
- [ ] T008 — Написать раздел "Известные несоответствия и риски": нерабочий
      импорт `src.rag_pipeline` в `main_workflow.py`, пустой
      `config/config.yaml` (`AGENTS.md`)
- [ ] T009 — Написать раздел "Тесты, CI, линтеры" с явным указанием на их
      отсутствие на момент написания документа (`AGENTS.md`)

## Verification

- [ ] T010 — Проверить, что файл создан: `test -f AGENTS.md`
- [ ] T011 — Проверить, что документ упоминает все обязательные переменные
      окружения: `grep -c -E "GIGACHAT_API_KEY|GIGACHAT_API_SCOPE|LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY|USER_AGENT" AGENTS.md`
- [ ] T012 — Проверить, что в документ не попали реальные значения секретов
      из `.env-example`: `! grep -q -F -f <(grep -oE '=.+' .env-example | sed "s/^=//") AGENTS.md`
- [ ] T013 — Проверить, что исходный код репозитория не изменился:
      `git diff --stat -- main_agent.py main_workflow.py src scripts`
