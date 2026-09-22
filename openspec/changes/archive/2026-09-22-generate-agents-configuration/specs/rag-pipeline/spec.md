# RAG Pipeline Specification Delta

## Purpose

Определяет компоненты и поведение RAG (Retrieval-Augmented Generation) pipeline, который служит основой для обогащения ответов агентов релевантным контекстом из документов рецептов.

RAG pipeline использует гибридный подход (BM25 + FAISS), переранжирование (FlashrankReranker), и LLM (GigaChat) для генерации ответов.

## Requirements

### ADDED Requirements

#### Requirement: Vector Store (FAISS)

Pipeline использует FAISS для плотного (dense) поиска по эмбеддингам.

##### Scenario: Initialize FAISS Store
- **WHEN** RAG pipeline инициализируется
- **THEN** создаётся FAISSVectorStore с загруженными эмбеддингами рецептов
- **AND** эмбеддинги генерируются или загружаются из предварительно подготовленного индекса

##### Scenario: Search by Embedding Similarity
- **WHEN** запрос передаётся в FAISS retriever
- **THEN** запрос преобразуется в эмбеддинг
- **AND** выполняется поиск k-ближайших соседей по cosine similarity

##### Scenario: Return Top-K Results
- **WHEN** поиск в FAISS завершён
- **THEN** возвращаются top-k документов (рецепты) с наивысшей подобностью
- **AND** каждый результат содержит содержание документа и метаданные (row, название)

#### Requirement: BM25 Sparse Retriever

Pipeline использует BM25 для разреженного (sparse) полнотекстового поиска.

##### Scenario: Index Cookbook Documents
- **WHEN** BM25Store инициализируется
- **THEN** все рецепты индексируются на основе лексических токенов
- **AND** индекс создаётся с использованием стандартного BM25 алгоритма

##### Scenario: Search by Keyword Match
- **WHEN** запрос передаётся в BM25 retriever
- **THEN** выполняется полнотекстовый поиск по токенам
- **AND** результаты ранжируются по BM25 score

##### Scenario: Return Matched Documents
- **WHEN** поиск завершён
- **THEN** возвращаются документы, которые содержат релевантные ключевые слова
- **AND** каждый результат содержит текст рецепта и BM25 score

#### Requirement: Ensemble Retriever with RRF

Pipeline комбинирует результаты FAISS и BM25 через RRF (Reciprocal Rank Fusion).

##### Scenario: Combine Dense and Sparse Results
- **WHEN** оба retriever (FAISS и BM25) завершили поиск
- **THEN** результаты объединяются через EnsembleRetriever
- **AND** используется RRF с равными весами (0.5 / 0.5) для балансировки

##### Scenario: Apply RRF Formula
- **WHEN** результаты объединяются
- **THEN** применяется формула RRF: score = sum(1 / (k + rank_i))
- **AND** документы, которые ранжированы высоко в обоих retriever'ах, получают выше финальный score

##### Scenario: Return Ensemble Results
- **WHEN** RRF вычисление завершено
- **THEN** возвращаются отранжированные документы в порядке убывания финального score
- **AND** результаты содержат информацию о том, где документ был найден (FAISS/BM25)

#### Requirement: Contextual Compression and Reranking

Pipeline использует FlashrankReranker для переранжирования и сжатия контекста.

##### Scenario: Rerank with FlashrankReranker
- **WHEN** результаты ensemble retriever'а готовы
- **THEN** они передаются в ContextualCompressionRetriever с FlashrankReranker
- **AND** reranker переранжирует документы по релевантности к исходному запросу

##### Scenario: Select Top-N Results
- **WHEN** переранжирование завершено
- **THEN** выбираются top-n (обычно 3) наиболее релевантных документов
- **AND** остальные документы отбрасываются

##### Scenario: Compress Context
- **WHEN** выбранные документы компрессируются
- **THEN** может быть удалена лишняя информация (опционально)
- **AND** компрессированный контекст передаётся LLM для генерации ответа

#### Requirement: LLM Integration (GigaChat)

Pipeline использует LLM для генерации ответов на основе контекста.

##### Scenario: Initialize GigaChat LLM
- **WHEN** Generator инициализируется
- **THEN** используется GigaChat-2-Max как LLM
- **AND** температура установлена на 0.3 для детерминированных ответов

##### Scenario: Generate Answer with Context
- **WHEN** контекст из RAG pipeline готов
- **THEN** запрос и контекст передаются в GigaChat
- **AND** LLM генерирует ответ на русском языке

##### Scenario: Handle Context Window
- **WHEN** контекст слишком большой для контекстного окна LLM
- **THEN** контекст автоматически сокращается (через top-n из reranker)
- **AND** самые релевантные документы используются в первую очередь

#### Requirement: Multi-Language Support

Pipeline поддерживает работу с русским языком на входе и выходе.

##### Scenario: Accept Russian Query
- **WHEN** пользователь передаёт запрос на русском
- **THEN** запрос используется как-есть для retrieval
- **AND** внутренне может быть переведён на английский для retriever'ов (если необходимо)

##### Scenario: Return Russian Response
- **WHEN** LLM генерирует ответ
- **THEN** ответ возвращается на русском языке
- **AND** ответ содержит ссылки на найденные рецепты (row и оригинальное название)

##### Scenario: Handle Multilingual Documents
- **WHEN** рецепты содержат текст на разных языках
- **THEN** pipeline корректно индексирует и ищет во всех языках
- **AND** ответ генерируется в языке запроса

#### Requirement: Performance and Scalability

Pipeline оптимизирован для производительности.

##### Scenario: Search Large Cookbook
- **WHEN** на индекс загружено большое количество рецептов (тысячи)
- **THEN** поиск выполняется эффективно (субсекундные latency для retrieval)
- **AND** FAISS использует оптимизированные индексные структуры

##### Scenario: Cache Retrieved Results
- **WHEN** одинаковый запрос выполняется несколько раз
- **THEN** результаты могут быть закэшированы (опционально)
- **AND** кэш используется для ускорения повторяющихся запросов

#### Requirement: Monitoring and Logging

Pipeline логирует все шаги для мониторинга и отладки.

##### Scenario: Log Retrieval Scores
- **WHEN** документы найдены
- **THEN** логируются scores от каждого retriever'а (FAISS score, BM25 score, ensemble score)
- **AND** логирование интегрировано с Langfuse callbacks

##### Scenario: Log Reranker Results
- **WHEN** reranker переранжирует документы
- **THEN** логируются исходные scores и переранжированные scores
- **AND** логирование помогает отладке качества retrieval

##### Scenario: Log LLM Generation
- **WHEN** LLM генерирует ответ
- **THEN** логируется prompt, использованный контекст и сгенерированный ответ
- **AND** логирование включает latency и другие метрики

#### Requirement: Error Handling

Pipeline корректно обрабатывает ошибки.

##### Scenario: Handle Empty Index
- **WHEN** рецепты не загружены в FAISS или BM25
- **THEN** pipeline генерирует осмысленную ошибку при инициализации
- **AND** система не запускается с пустым индексом

##### Scenario: Handle Retrieval Failure
- **WHEN** retriever завершился с ошибкой
- **THEN** генерируется исключение с информацией о причине
- **AND** контекст сохраняется для retry/debugging

##### Scenario: Handle LLM Generation Failure
- **WHEN** GigaChat недоступен или превышен timeout
- **THEN** генерируется исключение с информацией об ошибке
- **AND** система может повторить попытку или вернуть fallback-ответ

#### Requirement: Configuration

Pipeline поддерживает конфигурируемое поведение.

##### Scenario: Configure Top-K Parameters
- **WHEN** pipeline инициализируется
- **THEN** можно настроить k для FAISS, top-n для reranker
- **AND** конфигурация применяется без изменения кода

##### Scenario: Configure LLM Parameters
- **WHEN** Generator инициализируется
- **THEN** можно настроить температуру, max_tokens и другие параметры GigaChat
- **AND** параметры используются при генерации ответов

