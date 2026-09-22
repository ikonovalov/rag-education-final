# Tasks: generate-agents-configuration

## Phase 1: Create AGENTS.md Structure

- [x] T001 — Создать файл `rag-education/AGENTS.md` с основной структурой и оглавлением
- [x] T002 — Добавить секцию "Обзор архитектуры" с диаграммой системы (`rag-education/AGENTS.md`)
- [x] T003 — Добавить секцию "Типология агентов" с описанием Interactive vs Programmatic (`rag-education/AGENTS.md`)

## Phase 2: Document Agent Interfaces and Types

- [x] T004 — Документировать Agent Interface: требуемые методы, типы, сигнатуры (`rag-education/AGENTS.md`)
- [x] T005 — Документировать ChiefAgent: роль, инструменты, примеры использования (`rag-education/AGENTS.md`)
- [x] T006 — Документировать RagWorkflow: шаги, состояние, примеры использования (`rag-education/AGENTS.md`)
- [x] T007 — Добавить примеры кода: как инициализировать и запустить каждый агент (`rag-education/AGENTS.md`)

## Phase 3: Document RAG Pipeline and Infrastructure

- [x] T008 — Документировать RAG Pipeline компоненты: FAISS, BM25, EnsembleRetriever (`rag-education/AGENTS.md`)
- [x] T009 — Документировать FlashrankReranker и процесс переранжирования (`rag-education/AGENTS.md`)
- [x] T010 — Документировать Generator и GigaChat LLM интеграцию (`rag-education/AGENTS.md`)
- [x] T011 — Добавить диаграмму RAG Pipeline с потоком данных (`rag-education/AGENTS.md`)

## Phase 4: Document Integration and Configuration

- [x] T012 — Документировать конфигурацию агентов (параметры, переопределения) (`rag-education/AGENTS.md`)
- [x] T013 — Документировать интеграцию с HGSDLC Runtime (узлы, input/output, контракты) (`rag-education/AGENTS.md`)
- [x] T014 — Документировать Langfuse мониторинг и настройку (`rag-education/AGENTS.md`)
- [x] T015 — Добавить инструкции по включению/отключению RAGAS metrics (`rag-education/AGENTS.md`)

## Phase 5: Document Usage Scenarios and Best Practices

- [x] T016 — Документировать сценарий: Интерактивный чат с пользователем (ChiefAgent) (`rag-education/AGENTS.md`)
- [x] T017 — Документировать сценарий: Батч-обработка запросов (RagWorkflow) (`rag-education/AGENTS.md`)
- [x] T018 — Документировать сценарий: Оценка качества с RAGAS (`rag-education/AGENTS.md`)
- [x] T019 — Документировать сценарий: Использование в HGSDLC узлах (`rag-education/AGENTS.md`)
- [x] T020 — Добавить best practices для расширения архитектуры новыми агентами (`rag-education/AGENTS.md`)

## Phase 6: Document Conventions and Guidelines

- [x] T021 — Документировать соглашения по языку (русский снаружи, английский внутри) (`rag-education/AGENTS.md`)
- [x] T022 — Документировать соглашения по логированию и трейсингу (`rag-education/AGENTS.md`)
- [x] T023 — Документировать соглашения по обработке ошибок и retry логике (`rag-education/AGENTS.md`)
- [x] T024 — Добавить troubleshooting секцию для распространённых проблем (`rag-education/AGENTS.md`)

## Phase 7: Review and Polish

- [x] T025 — Проверить консистентность документа (все примеры работают, ссылки верны)
- [x] T026 — Убедиться что AGENTS.md связывает требования из всех пяти delta-specs
- [x] T027 — Проверить что документ соответствует CLAUDE.md и HGSDLC правилам
- [x] T028 — Проверить форматирование и структуру (заголовки, ссылки, диаграммы)

## Verification

- [x] V001 — Запустить проверку markdown: `cd rag-education && mdl AGENTS.md || true` (игнорируем non-critical warnings)
- [x] V002 — Проверить что файл существует и имеет минимальный размер: `test -f rag-education/AGENTS.md && wc -l rag-education/AGENTS.md`
- [x] V003 — Проверить что all cinco delta-specs ссылаются в specs-index.md: `grep -c "spec.md" rag-education/openspec/changes/generate-agents-configuration/specs-index.md`
- [x] V004 — Проверить что design.md и proposal.md созданы и содержат ожидаемые секции: `grep -E "## (Why|What|Design|Decision)" rag-education/openspec/changes/generate-agents-configuration/design.md`
- [x] V005 — Убедиться что не осталось TODO/FIXME в документах: `grep -r "TODO\|FIXME" rag-education/openspec/changes/generate-agents-configuration/ || echo "No TODOs found"`

