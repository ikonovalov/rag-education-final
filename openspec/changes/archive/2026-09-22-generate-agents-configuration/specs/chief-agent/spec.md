# Chief Agent Specification Delta

## Purpose

Определяет поведение интерактивного ReAct-агента (ChiefAgent), который служит основной точкой входа для пользовательского взаимодействия с RAG-системой проекта.

ChiefAgent реализует паттерн ReAct (Reasoning + Acting) через инструмент `cookbook`, который подключен к RAG pipeline для поиска рецептов и ответов на вопросы.

## Requirements

### ADDED Requirements

#### Requirement: Interactive Interface

ChiefAgent предоставляет интерактивный интерфейс для диалога с пользователем.

##### Scenario: Accept Russian Language Input
- **WHEN** пользователь отправляет запрос на русском языке
- **THEN** ChiefAgent принимает этот запрос без преобразований
- **AND** запрос используется для поиска в RAG и генерации ответа

##### Scenario: Return Russian Language Response
- **WHEN** ChiefAgent завершил обработку запроса
- **THEN** возвращается ответ на русском языке
- **AND** ответ содержит: текст ответа, номер рецепта (row) и оригинальное название рецепта (если применимо)

##### Scenario: Maintain Conversational Context
- **WHEN** пользователь отправляет последовательные запросы
- **THEN** ChiefAgent может сохранять контекст между запросами
- **AND** контекст используется для улучшения последующих ответов

#### Requirement: ReAct Pattern with Cookbook Tool

ChiefAgent реализует логику Reasoning + Acting через инструмент `cookbook`.

##### Scenario: Reason About Query
- **WHEN** ChiefAgent получает запрос от пользователя
- **THEN** агент начинает процесс рассуждения (Reasoning)
- **AND** генерирует мыслительный процесс о том, какой инструмент использовать

##### Scenario: Call Cookbook Tool
- **WHEN** агент решил что для ответа нужен поиск в рецептах
- **THEN** вызывается инструмент `cookbook(query: str)`
- **AND** query передаётся в compression_retriever RAG-pipeline

##### Scenario: Process Tool Output
- **WHEN** инструмент `cookbook` возвращает результаты
- **THEN** агент интегрирует результаты в контекст
- **AND** использует результаты для формирования финального ответа

##### Scenario: Handle Tool Not Available
- **WHEN** инструмент `cookbook` недоступен или вернул ошибку
- **THEN** ChiefAgent генерирует уместный ответ об ошибке
- **AND** предлагает пользователю переформулировать запрос или повторить попытку

#### Requirement: RAG Pipeline Integration

ChiefAgent интегрирован с RAG-системой для поиска документов.

##### Scenario: Search Through Cookbook
- **WHEN** инструмент `cookbook` вызван с поисковым запросом
- **THEN** запрос обрабатывается через compression_retriever (гибридный поиск + reranking)
- **AND** возвращаются релевантные рецепты, отранжированные по релевантности

##### Scenario: Include Retrieved Context in Response
- **WHEN** поиск вернул релевантные рецепты
- **THEN** контекст этих рецептов используется в генерации ответа
- **AND** ответ может включать ссылки на row (номер рецепта) и оригинальные названия

##### Scenario: Handle Empty Search Results
- **WHEN** поиск не вернул релевантных результатов
- **THEN** ChiefAgent генерирует ответ о том, что рецепт не найден
- **AND** предлагает альтернативные подходы (переформулировка, уточнение)

#### Requirement: LLM Integration

ChiefAgent использует LLM (GigaChat) для генерации ответов.

##### Scenario: Generate Response with GigaChat
- **WHEN** агент готов сформировать финальный ответ
- **THEN** используется LLM GigaChat для генерации текста
- **AND** температура LLM установлена на 0.3 (детерминированные ответы)

##### Scenario: Format Output in Markdown
- **WHEN** ответ сгенерирован LLM
- **THEN** ответ форматируется в Markdown для удобочитаемости
- **AND** структура включает заголовки, списки, кодовые блоки если применимо

#### Requirement: Monitoring and Logging

ChiefAgent логирует все действия для мониторинга и отладки.

##### Scenario: Log Tool Calls
- **WHEN** ChiefAgent вызывает инструмент `cookbook`
- **THEN** вызов логируется с информацией о запросе и результатах
- **AND** логирование интегрировано с Langfuse callbacks

##### Scenario: Log Generation Steps
- **WHEN** AgentChief выполняет основные шаги (reasoning, tool-call, generation)
- **THEN** каждый шаг логируется с метаданными
- **AND** метаданные включают timestamps, IDs для трейсинга

#### Requirement: Error Handling

ChiefAgent корректно обрабатывает ошибки и граничные случаи.

##### Scenario: Handle Malformed Input
- **WHEN** пользователь отправляет пустой или невалидный запрос
- **THEN** агент генерирует вежливый ответ о необходимости уточнения
- **AND** не прерывает работу сессии

##### Scenario: Handle LLM Timeout
- **WHEN** GigaChat недоступен или превышен timeout
- **THEN** ChiefAgent генерирует ошибку с указанием проблемы
- **AND** сохраняет контекст для повторной попытки

#### Requirement: Compliance with Agent Interface

ChiefAgent реализует требования universal Agent Interface.

##### Scenario: Implement run() Method for Interactive Agents
- **WHEN** ChiefAgent вызван как interactive-агент
- **THEN** имеет метод `run(user_input: str) -> AgentOutput`
- **AND** AgentOutput содержит `response`, `status`, `context`

##### Scenario: Support Configuration
- **WHEN** ChiefAgent инициализируется
- **THEN** поддерживает параметры конфигурации (LLM, temperature, tools)
- **AND** параметры могут быть переопределены в `run()` вызове

