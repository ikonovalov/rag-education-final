# Agent Interface Specification

## Purpose

Определяет универсальный интерфейс для всех типов агентов в проекте, обеспечивая согласованность в их взаимодействии, запуске и конфигурации.

Интерфейс описывает:
- Какие методы/инструменты обязаны реализовать все агенты
- Как агенты получают входные данные (конфигурация, состояние, инструменты)
- Как агенты возвращают результаты (структура output)
- Как агенты интегрируются в HGSDLC Runtime

## Requirements

### Requirement: Agent Type Classification

Все агенты в системе классифицируются по типам в зависимости от применения и архитектуры.

#### Scenario: Identify Interactive Agent Type
- **WHEN** агент предоставляет интерактивное взаимодействие с пользователем (например, через Streamlit, чат-интерфейс)
- **THEN** агент классифицируется как **Interactive Agent**
- **AND** должен иметь метод `run(user_input: str) -> AgentOutput`

#### Scenario: Identify Programmatic Agent Type
- **WHEN** агент используется для батч-обработки или встроен в граф LangGraph
- **THEN** агент классифицируется как **Programmatic Agent**
- **AND** должен иметь метод `run(state: Dict[str, Any]) -> Dict[str, Any]`

### Requirement: Agent Configuration

Каждый агент должен поддерживать конфигурируемое поведение через параметры.

#### Scenario: Create Agent with Configuration
- **WHEN** агент инициализируется с набором параметров (LLM, температура, tools)
- **THEN** агент применяет эти параметры при выполнении
- **AND** параметры должны быть документированы в виде dataclass или конфиг-файла

#### Scenario: Runtime Parameter Override
- **WHEN** параметр передан в `run()` метод
- **THEN** он переопределяет параметр из конфигурации агента
- **AND** изменение не влияет на само определение агента

### Requirement: Tool Integration

Агент может использовать инструменты (tools) для расширения своих возможностей.

#### Scenario: Agent Executes Tool
- **WHEN** агент вызывает инструмент по имени
- **THEN** инструмент выполняется и возвращает результат
- **AND** результат интегрируется в контекст агента (состояние, память)

#### Scenario: Tool Not Found
- **WHEN** агент пытается использовать инструмент, которого нет в наборе инструментов
- **THEN** генерируется исключение или error-state
- **AND** информация об ошибке включается в output агента

### Requirement: Standardized Output Format

Все агенты возвращают результаты в унифицированном формате.

#### Scenario: Return Successful Output
- **WHEN** агент завершил выполнение успешно
- **THEN** возвращается `AgentOutput` с полями: `response`, `status`, `context` (опционально)
- **AND** `status` должен быть `success`

#### Scenario: Return Error Output
- **WHEN** агент столкнулся с ошибкой во время выполнения
- **THEN** возвращается `AgentOutput` с полем `error_message` и `status = error`
- **AND** контекст ошибки должен быть сохранён для debug

### Requirement: Logging and Monitoring

Каждый агент логирует свою деятельность для мониторинга и debug.

#### Scenario: Agent Logs Execution Steps
- **WHEN** агент выполняет основные шаги (инициализация, tool-call, завершение)
- **THEN** каждый шаг логируется с уровнем INFO или выше
- **AND** логи должны быть доступны через стандартный механизм Langfuse callbacks

#### Scenario: Agent Logs Errors
- **WHEN** агент столкнулся с ошибкой
- **THEN** ошибка логируется с уровнем ERROR
- **AND** stack trace должен быть сохранён для диагностики

### Requirement: Language Support

Агенты поддерживают работу с разными языками в зависимости от типа.

#### Scenario: Interactive Agent Handles Russian Input
- **WHEN** пользователь передаёт текст на русском языке
- **THEN** агент обрабатывает текст и возвращает ответ на русском
- **AND** внутренние промежуточные шаги могут использовать английский (для retrieval, embeddings)

#### Scenario: Programmatic Agent Works Language-Agnostic
- **WHEN** Programmatic Agent получает входные данные
- **THEN** язык определяется контекстом и конфигурацией
- **AND** агент работает корректно независимо от языка входных данных

### Requirement: Extensibility and Versioning

Архитектура должна позволять добавлять новые агенты и новые версии существующих.

#### Scenario: Register New Agent Type
- **WHEN** разработчик создаёт новый класс агента (наследует от BaseAgent)
- **THEN** новый агент может быть используется в проекте без изменения интерфейса
- **AND** новый агент должен реализовать обязательные методы интерфейса

#### Scenario: Version Different Implementations
- **WHEN** существуют несколько реализаций одного типа агента (например, ChiefAgent v1 и v2)
- **THEN** версии различаются по имени или параметру конфигурации
- **AND** можно переключаться между версиями без нарушения интеграции
