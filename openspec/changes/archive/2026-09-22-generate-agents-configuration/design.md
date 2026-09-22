# Design: generate-agents-configuration

## Overview

Создание комплексного документа AGENTS.md, который документирует многоагентную архитектуру проекта. Документ описывает две линии агентов (интерактивная ChiefAgent и программируемая RagWorkflow), их компоненты (RAG pipeline), интеграцию и сценарии использования.

Архитектурный подход:
- **Двухслойная модель агентов**: Interactive (для пользователей) и Programmatic (для батч-обработки)
- **Унифицированный интерфейс**: оба типа агентов соответствуют одному интерфейсу с разными сигнатурами методов
- **Общая инфраструктура**: оба агента используют одну и ту же RAG pipeline (compression_retriever, Generator)
- **Мониторинг**: интеграция с Langfuse для трейсинга всех операций
- **Интеграция с HGSDLC**: агенты готовы к использованию в узлах HGSDLC Runtime

## Decisions

### Decision: Двухслойная архитектура агентов

**Выбор:** Разделить агентов на Interactive (ChiefAgent) и Programmatic (RagWorkflow) линии

**Обоснование:**
- Interactive агенты лучше подходят для пользовательского диалога (принимают строку, возвращают строку)
- Programmatic агенты лучше подходят для батч-обработки и интеграции в HGSDLC (работают с состоянием)
- Разделение позволяет каждому типу оптимизировать свой интерфейс без компромиссов

**Альтернативы:**
- Один универсальный агент (менее гибкий, сложнее адаптировать к разным сценариям)
- Три и более типов (усложняет архитектуру, не даёт добавочных преимуществ)

### Decision: Унифицированный Agent Interface

**Выбор:** Определить единый BaseAgent interface с двумя вариантами `run()` метода

**Обоснование:**
- Обеспечивает согласованность в системе (все агенты следуют одним правилам)
- Позволяет легко добавлять новые типы агентов
- Упрощает интеграцию с HGSDLC Runtime (единый контракт для узлов)

**Альтернативы:**
- Полностью разные интерфейсы (нет согласованности, сложнее расширять)
- Один интерфейс для всех (снижает производительность и гибкость)

### Decision: Общая RAG Pipeline Инфраструктура

**Выбор:** ChiefAgent и RagWorkflow используют один и тот же compression_retriever и Generator

**Обоснование:**
- Избегаем дублирования кода и конфигурации
- Гарантирует согласованность результатов поиска и генерации
- Облегчает мониторинг (все операции через один компонент)
- Экономит память (один индекс FAISS вместо двух)

**Альтернативы:**
- Каждый агент имеет свой RAG pipeline (дублирование, больше памяти)
- Общий RAG но с разными конфигурациями (усложняет управление состоянием)

### Decision: LangGraph для RagWorkflow

**Выбор:** Использовать LangGraph StateGraph для реализации RagWorkflow

**Обоснование:**
- LangGraph обеспечивает явное управление состоянием и шагами
- Поддерживает асинхронное выполнение и параллелизм
- Интегрирован с LangChain (используется Generator и retrievers)
- Упрощает добавление новых шагов в граф

**Альтернативы:**
- Простой Python класс с методами (менее гибкий для батч-обработки)
- Celery/RQ для очередей (усложняет архитектуру для текущих требований)

### Decision: ReAct Pattern для ChiefAgent

**Выбор:** Использовать ReAct (Reasoning + Acting) паттерн с инструментом cookbook

**Обоснование:**
- ReAct позволяет агенту "размышлять" о том какой инструмент использовать
- Инструмент cookbook автоматически интегрирует RAG поиск
- Паттерн хорошо изучен и надёжен для диалоговых систем

**Альтернативы:**
- Прямой вызов RAG без рассуждений (менее гибко, сложнее debug)
- Chain of Thought (более тяжеловесно, не даёт добавочных преимуществ для текущих сценариев)

### Decision: Гибридный Retrieval (BM25 + FAISS)

**Выбор:** Комбинировать BM25 (разреженный поиск) и FAISS (плотный поиск) через RRF

**Обоснование:**
- BM25 хорошо работает для точного совпадения ключевых слов
- FAISS хорошо работает для семантического поиска через эмбеддинги
- RRF балансирует оба подхода ( 0.5 / 0.5)
- Комбинация часто даёт лучшие результаты чем один retriever

**Альтернативы:**
- Только BM25 (хуже для семантического поиска)
- Только FAISS (хуже для точного совпадения)
- Вес 0.7/0.3 (может быть оптимизирован позже через эксперименты)

### Decision: FlashrankReranker для Переранжирования

**Выбор:** Использовать FlashrankReranker (ms-marco-MultiBERT-L-12) в ContextualCompressionRetriever

**Обоснование:**
- FlashrankReranker высокоточен и быстр (небольшая модель L-12)
- ms-marco обучена на большом датасете (хорошо обобщается)
- Переранжирование улучшает качество top-3 результатов
- ContextualCompressionRetriever упрощает интеграцию в pipeline

**Альтернативы:**
- Без переранжирования (худше качество, но быстрее)
- Более тяжелая модель (лучшее качество, но медленнее)
- Другой reranker (мало данных для сравнения в текущем контексте)

### Decision: GigaChat-2-Max как LLM

**Выбор:** Использовать GigaChat-2-Max для генерации ответов

**Обоснование:**
- GigaChat — лучшая открытая русскоязычная LLM доступная в проекте
- GigaChat-2-Max обеспечивает приемлемое соотношение качества и скорости
- Температура 0.3 обеспечивает детерминированные ответы (важно для воспроизводимости)

**Альтернативы:**
- Другая LLM (требует интеграции, потеря русскоязычной поддержки)
- Более тяжелая версия GigaChat (медленнее, не нужно для текущих требований)

### Decision: Langfuse для Мониторинга

**Выбор:** Использовать Langfuse callbacks для трейсинга всех операций

**Обоснование:**
- Langfuse интегрирован с LangChain и LangGraph
- Позволяет трейсировать все вызовы tools, LLM, retriever'ов
- Обеспечивает observability для debug и оптимизации
- Поддерживает вложенные traces (шаги внутри шагов)

**Альтернативы:**
- Стандартное Python логирование (менее гибко для трейсинга)
- Другой инструмент (требует переинтеграции)

### Decision: RAGAS Metrics как Опциональная Оценка

**Выбор:** Включить RAGAS metrics в RagWorkflow как опциональный шаг

**Обоснование:**
- RAGAS обеспечивает несмещённую оценку качества RAG (не требует ground truth)
- Метрики LLMContextRecall, AnswerRelevancy полезны для мониторинга
- Опциональность позволяет отключить для производительности если нужно

**Альтернативы:**
- Обязательная оценка (может быть медленно для батч-обработки больших объёмов)
- Без оценки (теряется visibility в качество)

## Architecture

### High-Level System Diagram

```
┌────────────────────────────────────────────────────────────┐
│                      HGSDLC Runtime                         │
│  (Node Orchestration, Attempt Management, Logging)         │
└────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                           │
         ┌──────▼───────┐          ┌────────▼──────┐
         │ ChiefAgent   │          │ RagWorkflow   │
         │ (Interactive)│          │ (Programmatic)│
         └──────┬───────┘          └────────┬──────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Agent Interface   │
                    │ (BaseAgent)       │
                    └─────────┬─────────┘
                              │
                  ┌───────────┴───────────┐
                  │                       │
              ┌───▼──┐           ┌───────▼────┐
              │Tools │           │ Configuration
              └──────┘           └────────────┘
                  │
           ┌──────┴──────────┐
           │                 │
        ┌──▼────────┐    ┌──▼─────────────┐
        │ Generator │    │ compression_   │
        │(GigaChat) │    │ retriever      │
        └────┬──────┘    └──┬──────┬──────┘
             │              │      │
         ┌───▼──────────┐   │      │
         │ LLM Interface│   │      │
         │ (GigaChat-2) │   │      │
         └──────────────┘   │      │
                       ┌────▼──┬───▼──────┐
                       │ FAISS │ BM25     │
                       └────┬──┴────┬─────┘
                            │       │
                       ┌────▼───┬───▼────┐
                       │EnsembleRetriever│
                       │ (RRF 0.5/0.5)   │
                       └────┬────────────┘
                            │
                    ┌───────▼────────┐
                    │FlashrankReranker│
                    │ (top-3 results)│
                    └────────────────┘
                            │
                    ┌───────▼──────────┐
                    │ Recipe Documents │
                    │ (Indexed)        │
                    └──────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    Monitoring Layer                       │
│  (Langfuse callbacks, RAGAS metrics, Logging)            │
└──────────────────────────────────────────────────────────┘
```

### ChiefAgent Flow

```
User Input (Russian)
    │
    ▼
ChiefAgent (ReAct)
    │
    ├─ Reasoning: Decide to use cookbook tool
    │
    ▼
Invoke cookbook(query)
    │
    ▼
compression_retriever
    │
    ├─ BM25 Search
    ├─ FAISS Search (embeddings)
    ├─ Ensemble (RRF)
    └─ FlashrankReranker (top-3)
    │
    ▼
Retrieved Documents (recipes)
    │
    ▼
GigaChat LLM (Generate answer with context)
    │
    ▼
Return AgentOutput:
    - response: "Answer in Russian"
    - status: "success"
    - context: retrieved_docs
    │
    ▼
User Response (Russian)
```

### RagWorkflow Flow

```
Input State:
    {question: str, ...}
    │
    ▼
rephrase() step
    Rephrase + translate to English
    └─ Update state.rephrased
    │
    ▼
retrieve_hybrid() step
    BM25 + FAISS + Ensemble + Reranker
    └─ Update state.retrieved
    │
    ▼
evaluate() step [LLM-as-Judge]
    Rate relevance of context
    └─ Log relevance score
    │
    ▼
generate() step
    GigaChat generates answer
    └─ Update state.answer
    │
    ▼
[Optional] evaluate_output() step [RAGAS]
    Calculate quality metrics
    └─ Log metrics
    │
    ▼
Output State:
    {answer: str, retrieved: List[Document], ...}
```

## Data Model

### Agent Output Structure

```python
@dataclass
class AgentOutput:
    response: str                    # Main output (Russian text)
    status: str                      # "success" | "error"
    context: Optional[Dict]          # Optional context data
    error_message: Optional[str]     # Error details if status="error"
    metadata: Optional[Dict]         # Timestamps, IDs, etc.
```

### RagWorkflow State Structure

```python
@dataclass
class RagWorkflowState:
    question: str                    # Original user question
    rephrased: str                   # Rephrased + translated version
    retrieved: List[Document]        # Retrieved recipes
    answer: str                      # Generated answer
    relevance_score: Optional[float] # LLM-as-Judge score
    metrics: Optional[Dict]          # RAGAS metrics
```

### Agent Configuration Structure

```python
@dataclass
class AgentConfig:
    llm_model: str = "GigaChat-2-Max"
    llm_temperature: float = 0.3
    max_retrieved_docs: int = 3
    retriever_weights: Tuple[float, float] = (0.5, 0.5)
    enable_ragas: bool = False
    langfuse_enabled: bool = True
```

## Implementation Notes

### Key Patterns

1. **Tool Integration**: Инструменты (как `cookbook`) интегрируются через LangChain tool decorators
2. **State Management**: RagWorkflow использует LangGraph StateGraph для явного управления состоянием
3. **Error Recovery**: Агенты логируют ошибки и поддерживают retry через HGSDLC Runtime
4. **Language Handling**: Внешний интерфейс русский, внутренне может быть английский для retrieval

### Performance Considerations

- FAISS использует CPU-optimized индексные структуры (быстро для up-to millions документов)
- ReRanker (FlashrankReranker) работает только на top-3 результатов (быстро)
- Кэширование может быть добавлено для повторяющихся запросов

### Extensibility

- Новые агенты создаются наследованием от BaseAgent и реализацией `run()` метода
- Новые шаги в RagWorkflow добавляются как функции с подписью `(state) -> state`
- Новые tools интегрируются через LangChain tool decorator

## Risks

1. **LLM Dependency**: Система зависит от доступности GigaChat
   - Mitigation: Добавить fallback LLM или локальную модель

2. **Index Staleness**: FAISS индекс может устаревать если рецепты обновляются
   - Mitigation: Периодическое переиндексирование или incremental updates

3. **Embedding Quality**: Качество поиска зависит от качества embedding модели
   - Mitigation: Мониторить RAGAS metrics и A/B тестировать embedding модели

4. **Token Limits**: GigaChat имеет ограничение на контекстное окно
   - Mitigation: top-3 restriction в reranker, может быть настроено

5. **Concurrency Issues**: Несколько агентов могут обращаться к одному indey одновременно
   - Mitigation: FAISS индекс thread-safe, Generator может быть pooled

