# RAG Workflow Specification

## Purpose

Определяет поведение граф-ориентированного рабочего процесса (RagWorkflow) на базе LangGraph, который используется для батч-обработки запросов, оценки качества и интеграции с многошаговым HGSDLC Runtime процессом.

RagWorkflow реализует многошаговый граф с промежуточной оценкой (LLM-as-Judge) и метриками качества (RAGAS).

## Requirements

### Requirement: State Management

RagWorkflow управляет состоянием (state) на протяжении всего процесса.

#### Scenario: Initialize Workflow State
- **WHEN** RagWorkflow инициализируется
- **THEN** создаётся начальное состояние с полями: `question`, `rephrased`, `retrieved`, `answer`
- **AND** состояние типизируется как `RagWorkflowState`

#### Scenario: Update State Between Steps
- **WHEN** один из шагов графа завершён
- **THEN** состояние обновляется с результатами этого шага
- **AND** обновленное состояние передаётся следующему шагу

#### Scenario: Access Previous Step Results
- **WHEN** текущий шаг выполняется
- **THEN** может получить доступ к результатам всех предыдущих шагов через state
- **AND** данные из state используются для принятия решений

### Requirement: Multi-Step Pipeline

RagWorkflow организован как последовательность явных шагов.

#### Scenario: Rephrase and Translate Query
- **WHEN** workflow начинается с пользовательского запроса
- **THEN** первый шаг `rephrase()` перефразирует и переводит запрос на английский
- **AND** результат сохраняется в `state.rephrased`

#### Scenario: Hybrid Retrieval
- **WHEN** `retrieve_hybrid()` выполняется
- **THEN** используется compression_retriever для гибридного поиска (BM25 + FAISS + reranking)
- **AND** найденные документы сохраняются в `state.retrieved`

#### Scenario: Evaluate Context Relevance
- **WHEN** шаг `evaluate()` выполняется
- **THEN** используется LLM-as-Judge для оценки релевантности найденного контекста
- **AND** оценка влияет на качество финального ответа (может быть логирована или использована для фильтрации)

#### Scenario: Generate Answer
- **WHEN** шаг `generate()` выполняется
- **THEN** GigaChat используется для генерации ответа на основе контекста
- **AND** ответ сохраняется в `state.answer`

#### Scenario: Optional Quality Evaluation
- **WHEN** шаг `evaluate_output()` выполняется (опционально)
- **THEN** используются RAGAS metrics для оценки качества финального ответа
- **AND** метрики логируются для анализа

### Requirement: LLM-as-Judge Integration

RagWorkflow встраивает LLM-as-Judge для оценки релевантности.

#### Scenario: Judge Context Relevance
- **WHEN** контекст из retriever готов к использованию
- **THEN** LLM оценивает релевантность контекста относительно исходного вопроса
- **AND** оценка возвращает score или boolean (relevant/not relevant)

#### Scenario: Log Relevance Score
- **WHEN** оценка релевантности получена
- **THEN** score логируется для мониторинга качества
- **AND** может использоваться для филтрации низкорелевантного контекста

### Requirement: RAGAS Metrics Integration

RagWorkflow поддерживает опциональное использование RAGAS для оценки качества.

#### Scenario: Calculate Answer Relevancy
- **WHEN** ответ сгенерирован
- **THEN** опционально вычисляется метрика `AnswerRelevancy`
- **AND** результат логируется для анализа

#### Scenario: Calculate Context Recall
- **WHEN** найденный контекст оценивается
- **THEN** опционально вычисляется метрика `LLMContextRecall`
- **AND** результат логируется и может влиять на решение о качестве retrieval

### Requirement: Langfuse Monitoring

RagWorkflow интегрирован с Langfuse для трейсинга и мониторинга.

#### Scenario: Log Each Step
- **WHEN** каждый шаг графа выполняется
- **THEN** данные о выполнении логируются через Langfuse callbacks
- **AND** логирование включает input, output и метаданные

#### Scenario: Create Trace for Entire Workflow
- **WHEN** workflow завершён
- **THEN** весь граф представлен в Langfuse как одна trace для анализа
- **AND** trace содержит информацию о всех шагах и связях между ними

### Requirement: Programmatic Interface

RagWorkflow предоставляет программный интерфейс для интеграции в другие системы.

#### Scenario: Run Workflow Programmatically
- **WHEN** RagWorkflow вызван как programmatic-агент
- **THEN** имеет метод `run(state: Dict[str, Any]) -> Dict[str, Any]`
- **AND** принимает входное состояние и возвращает выходное состояние

#### Scenario: Batch Processing
- **WHEN** требуется обработать список запросов
- **THEN** RagWorkflow может быть запущен в цикле для каждого запроса
- **AND** результаты собираются для анализа и оценки

### Requirement: Error Handling and Robustness

RagWorkflow корректно обрабатывает ошибки на каждом шаге.

#### Scenario: Handle Empty Retrieval
- **WHEN** поиск не вернул релевантных документов
- **THEN** workflow продолжает работу с пустым контекстом
- **AND** LLM-as-Judge отмечает это в логах, ответ указывает на отсутствие контекста

#### Scenario: Handle LLM Timeout
- **WHEN** GigaChat недоступен на любом шаге
- **THEN** генерируется исключение с информацией о шаге, где произошла ошибка
- **AND** контекст сохраняется для retry/debugging

#### Scenario: Continue on Non-Critical Error
- **WHEN** опциональный шаг (например, RAGAS evaluation) завершился с ошибкой
- **THEN** workflow продолжает работу и не прерывает pipeline
- **AND** ошибка логируется как warning

### Requirement: Configuration and Extensibility

RagWorkflow поддерживает конфигурацию и расширяемость.

#### Scenario: Configure Pipeline Steps
- **WHEN** RagWorkflow инициализируется
- **THEN** может быть настроено какие шаги включать (например, include_ragas=True/False)
- **AND** конфигурация применяется без изменения кода

#### Scenario: Add Custom Step
- **WHEN** требуется добавить новый шаг в граф
- **THEN** новый шаг может быть добавлен как функция с подписью `step(state: RagWorkflowState) -> RagWorkflowState`
- **AND** граф автоматически перестраивается с новым шагом

### Requirement: Compliance with Agent Interface

RagWorkflow реализует требования universal Agent Interface.

#### Scenario: Implement Programmatic Agent Interface
- **WHEN** RagWorkflow используется как programmatic-агент
- **THEN** соответствует интерфейсу programmatic-агентов
- **AND** поддерживает конфигурацию через параметры инициализации
