# HGSDLC Integration Specification

## Purpose

Определяет требования для интеграции AI-агентов с многошаговым HGSDLC Runtime процессом разработки.

HGSDLC (Hierarchical Graph-based Software Development Lifecycle) — это система для автоматизированного управления многошаговыми процессами разработки программного обеспечения с использованием AI-компонентов.

Интеграция обеспечивает взаимодействие между агентами и узлами HGSDLC Runtime (ai-describe-change, ai-explore, ai-propose и т.д.).

## Requirements

### Requirement: Agent Participation in HGSDLC Nodes

Агенты могут участвовать в различных узлах (nodes) HGSDLC процесса.

#### Scenario: Execute Agent in HGSDLC Node
- **WHEN** узел HGSDLC Runtime требует выполнения AI-задачи
- **THEN** может быть вызван соответствующий агент (ChiefAgent, RagWorkflow, или другой)
- **AND** агент получает входные данные из узла и возвращает результаты

#### Scenario: Chain Multiple Nodes with Agents
- **WHEN** несколько узлов HGSDLC выполняются последовательно
- **THEN** агент из узла N может использовать output узла N-1
- **AND** выходные данные агента передаются в узел N+1

#### Scenario: Parallel Agent Execution
- **WHEN** несколько узлов могут выполняться параллельно
- **THEN** несколько агентов могут работать одновременно
- **AND** их результаты синхронизируются в следующем узле

### Requirement: Node Input/Output Contract

Каждый агент в HGSDLC узле соответствует контракту input/output.

#### Scenario: Receive Node Input
- **WHEN** агент запускается внутри HGSDLC узла
- **THEN** получает входные данные в формате, определённом узлом
- **AND** входные данные могут включать файлы, параметры, контекст из предыдущих узлов

#### Scenario: Produce Node Output
- **WHEN** агент завершает выполнение
- **THEN** выводит результаты в формате, ожидаемом следующим узлом
- **AND** результаты сохраняются в файлы или структурированные данные

#### Scenario: Validate Input Contract
- **WHEN** агент получает входные данные
- **THEN** проверяет что входные данные соответствуют ожиданиям
- **AND** генерирует ошибку если входные данные невалидны

### Requirement: Node-Specific Agent Roles

Разные типы агентов имеют разные роли в разных узлах.

#### Scenario: Agent in Analysis Node (ai-explore)
- **WHEN** узел ai-explore выполняется
- **THEN** может использоваться general-purpose агент для исследования кодовой базы
- **AND** агент возвращает exploration-notes.md с выводами

#### Scenario: Agent in Proposal Node (ai-propose)
- **WHEN** узел ai-propose выполняется
- **THEN** используется агент для создания архитектурных предложений
- **AND** агент возвращает proposal.md, specs и design документы

#### Scenario: Agent in Implementation Node (ai-implement)
- **WHEN** узел ai-implement выполняется
- **THEN** может использоваться специализированный агент для кодирования
- **AND** агент создаёт код, тесты и документацию

### Requirement: Configuration from HGSDLC Runtime

Агенты получают конфигурацию от HGSDLC Runtime.

#### Scenario: Receive Runtime Configuration
- **WHEN** агент запускается в узле HGSDLC
- **THEN** получает конфигурацию из node-specific файлов (например, config.json)
- **AND** конфигурация может переопределять дефолтные параметры агента

#### Scenario: Access Runtime Context
- **WHEN** агент выполняется
- **THEN** может получить доступ к контексту HGSDLC Runtime (текущий узел, attempt, статус)
- **AND** контекст используется для логирования, трейсинга и условной логики

### Requirement: Error Handling in HGSDLC Context

Агенты корректно обрабатывают ошибки в контексте HGSDLC.

#### Scenario: Agent Failure Recovery
- **WHEN** агент завершился с ошибкой
- **THEN** HGSDLC Runtime может повторить узел (retry) с теми же входными данными
- **AND** агент может вернуть информацию об ошибке для диагностики

#### Scenario: Partial Completion
- **WHEN** агент завершил часть работы перед ошибкой
- **THEN** может сохранить промежуточные результаты
- **AND** HGSDLC Runtime может использовать эти результаты при retry

#### Scenario: Timeout Handling
- **WHEN** агент превышает timeout, определённый для узла
- **THEN** HGSDLC Runtime прерывает агента
- **AND** агент может очистить ресурсы перед завершением

### Requirement: Monitoring and Tracing

Агенты, работающие в HGSDLC, должны быть монируемы и трейсируемы.

#### Scenario: Log Agent Execution in Node Context
- **WHEN** агент выполняется в узле HGSDLC
- **THEN** логирование включает информацию о узле (node_id, attempt_number)
- **AND** логи сохраняются в .hgsdlc/nodes/<node>/<attempt>/logs/

#### Scenario: Create Execution Trace
- **WHEN** агент завершает узел
- **THEN** создаётся трейс выполнения (через Langfuse или аналог)
- **AND** трейс содержит информацию о всех шагах выполнения

#### Scenario: Report Metrics to HGSDLC
- **WHEN** агент завершил работу
- **THEN** может отправить метрики (time, tokens used, quality score)
- **AND** метрики используются для оптимизации HGSDLC процесса

### Requirement: Data Persistence in HGSDLC Structure

Агенты используют структуру файлов HGSDLC для сохранения данных.

#### Scenario: Store Output in Node Directory
- **WHEN** агент завершает работу
- **THEN** сохраняет результаты в .hgsdlc/nodes/<node_id>/<attempt>/
- **AND** структура директорий следует соглашениям HGSDLC Runtime

#### Scenario: Access Input from Previous Node
- **WHEN** агент нужны результаты предыдущего узла
- **THEN** может получить их из .hgsdlc/nodes/<previous_node>/<attempt>/
- **AND** структура файлов предсказуема и документирована

#### Scenario: Save Step Summary
- **WHEN** узел завершён
- **THEN** агент создаёт step-summary.json с информацией о выполнении
- **AND** summary включает статус, действия и время выполнения

### Requirement: Versioning and Attempts

HGSDLC Runtime может повторить узел несколько раз (attempts).

#### Scenario: Handle Multiple Attempts
- **WHEN** узел повторяется (attempt > 1)
- **THEN** агент может получить доступ к результатам предыдущих attempts
- **AND** может использовать их для улучшения текущей попытки

#### Scenario: Store Attempt-Specific Data
- **WHEN** агент выполняет attempt N
- **THEN** результаты сохраняются в attempt-N директории
- **AND** предыдущие attempts остаются доступны для анализа

### Requirement: Integration with AGENTS.md Configuration

Агенты конфигурируются через AGENTS.md.

#### Scenario: Specify Agent in HGSDLC Node Configuration
- **WHEN** узел HGSDLC определяется в конфигурации
- **THEN** может быть указан какой агент (или какие агенты) используются
- **AND** конфигурация ссылается на AGENTS.md для деталей агента

#### Scenario: Override Agent Configuration per Node
- **WHEN** узел имеет специфичные требования к конфигурации агента
- **THEN** может переопределить параметры агента (температура, tools, и т.д.)
- **AND** переопределение применяется только для этого узла

### Requirement: Compatibility with Runtime Rules

Агенты соответствуют правилам HGSDLC Runtime.

#### Scenario: Follow Node Execution Rules
- **WHEN** агент выполняется в узле
- **THEN** соответствует правилам выполнения узла (языки, форматы, структура)
- **AND** не нарушает системные инварианты HGSDLC Runtime

#### Scenario: Support Serialization
- **WHEN** состояние узла нужно сохранить
- **THEN** состояние агента должно быть сериализуемо
- **AND** может быть восстановлено при retry без потери информации
