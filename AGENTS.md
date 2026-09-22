# AGENTS.md — Архитектура многоагентной системы

Документ описывает архитектуру, компоненты и интеграцию AI-агентов в проекте rag-education.

## Оглавление

1. [Обзор архитектуры](#обзор-архитектуры)
2. [Типология агентов](#типология-агентов)
3. [Agent Interface](#agent-interface)
4. [ChiefAgent](#chiefagent-интерактивный-агент)
5. [RagWorkflow](#ragworkflow-программируемый-агент)
6. [RAG Pipeline](#rag-pipeline)
7. [Интеграция с HGSDLC Runtime](#интеграция-с-hgsdlc-runtime)
8. [Сценарии использования](#сценарии-использования)
9. [Соглашения и правила](#соглашения-и-правила)
10. [Troubleshooting](#troubleshooting)

---

## Обзор архитектуры

Многоагентная система проекта организована в две основные линии:

- **Interactive (ChiefAgent)** — для пользовательского диалога
- **Programmatic (RagWorkflow)** — для батч-обработки и HGSDLC интеграции

Обе линии используют общую инфраструктуру:
- **RAG Pipeline** — гибридный поиск (BM25 + FAISS) + переранжирование (FlashrankReranker)
- **Generator** — LLM интерфейс для GigaChat-2-Max
- **Monitoring** — Langfuse callbacks для трейсинга

### Высокоуровневая диаграмма системы

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

---

## Типология агентов

Система поддерживает два основных типа агентов:

### Interactive Agents (Интерактивные)

Предназначены для диалога с пользователем:
- **Сигнатура:** `run(user_input: str) -> AgentOutput`
- **Вход:** строка на русском языке
- **Выход:** структурированный `AgentOutput` с ответом
- **Примеры:** ChiefAgent
- **Использование:** Streamlit приложения, чат-интерфейсы

### Programmatic Agents (Программируемые)

Предназначены для автоматизации и батч-обработки:
- **Сигнатура:** `run(state: Dict[str, Any]) -> Dict[str, Any]`
- **Вход:** состояние (Dictionary с вопросами, параметрами)
- **Выход:** обновлённое состояние с результатами
- **Примеры:** RagWorkflow
- **Использование:** LangGraph графы, HGSDLC узлы, батч-обработка

---

## Agent Interface

Universal interface для всех агентов в системе.

### Структура Output

Все агенты возвращают результаты в унифицированном формате:

```python
@dataclass
class AgentOutput:
    response: str                    # Main output (Russian text)
    status: str                      # "success" | "error"
    context: Optional[Dict]          # Optional context data
    error_message: Optional[str]     # Error details if status="error"
    metadata: Optional[Dict]         # Timestamps, IDs, etc.
```

### Конфигурация

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

### Требования к реализации

Каждый агент должен:
1. **Классифицироваться** как Interactive или Programmatic
2. **Поддерживать конфигурацию** через параметры инициализации
3. **Интегрировать tools** через LangChain tool decorators
4. **Логировать действия** через Langfuse callbacks
5. **Обрабатывать ошибки** с информативными сообщениями
6. **Поддерживать язык:** русский на входе/выходе для Interactive, контекстный для Programmatic

---

## ChiefAgent (интерактивный агент)

Интерактивный ReAct-агент для диалога с RAG-системой.

### Роль

ChiefAgent — основная точка входа для пользовательского взаимодействия. Агент:
- Принимает вопросы на русском языке
- Рассуждает о том какие инструменты использовать
- Вызывает инструмент `cookbook` для поиска в рецептах
- Генерирует ответ на русском языке через GigaChat

### Интерфейс

```python
class ChiefAgent(BaseAgent):
    def run(self, user_input: str, **config_overrides) -> AgentOutput:
        """
        Выполнить диалоговый раунд.
        
        Args:
            user_input: Вопрос пользователя на русском
            **config_overrides: Переопределение параметров конфигурации
        
        Returns:
            AgentOutput с ответом и статусом
        """
```

### Шаг: Reasoning (Рассуждение)

ChiefAgent начинает с рассуждения:
1. Анализирует запрос пользователя
2. Решает какой инструмент использовать (обычно `cookbook`)
3. Генерирует мыслительный процесс (видимо в логах через Langfuse)

### Шаг: Acting (Действие)

ChiefAgent вызывает инструмент `cookbook`:

```python
def cookbook(query: str) -> Dict:
    """
    Поиск рецептов в RAG pipeline.
    
    Args:
        query: Поисковый запрос
    
    Returns:
        Dict с найденными рецептами, scores, метаданными
    """
```

**Flow инструмента:**
1. Запрос передаётся в `compression_retriever`
2. Гибридный поиск: BM25 + FAISS + Ensemble + Reranking
3. Возвращаются top-3 релевантных рецепта

### Шаг: Generation (Генерация)

ChiefAgent использует GigaChat для генерации ответа:
- Контекст: найденные рецепты из `cookbook`
- Температура: 0.3 (детерминированные ответы)
- Язык: русский

### Пример использования

```python
# Инициализация
chief = ChiefAgent(config=AgentConfig(
    llm_model="GigaChat-2-Max",
    llm_temperature=0.3,
    max_retrieved_docs=3,
    langfuse_enabled=True
))

# Выполнение
result = chief.run(user_input="Как приготовить борщ?")

# Результат
print(result.response)      # "Для приготовления борщ..."
print(result.status)        # "success"
print(result.context)       # Dict с найденными рецептами
```

### Обработка ошибок

ChiefAgent корректно обрабатывает:
- **Пустой вввод:** возвращает вежливый запрос уточнения
- **LLM timeout:** возвращает error с информацией о проблеме
- **Tool not available:** возвращает error с предложением повторить

---

## RagWorkflow (программируемый агент)

Граф-ориентированный рабочий процесс на базе LangGraph для батч-обработки.

### Роль

RagWorkflow — автоматизированная система для:
- Батч-обработки списков вопросов
- Оценки качества с LLM-as-Judge и RAGAS
- Логирования метрик для анализа
- Интеграции с HGSDLC узлами

### Интерфейс

```python
class RagWorkflow(BaseAgent):
    def run(self, state: Dict[str, Any], **config_overrides) -> Dict[str, Any]:
        """
        Выполнить граф обработки.
        
        Args:
            state: Входное состояние с вопросом и параметрами
            **config_overrides: Переопределение параметров конфигурации
        
        Returns:
            Обновлённое состояние с результатами всех шагов
        """
```

### State Structure

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

### Граф с шагами

```
Input State (question)
    │
    ▼
rephrase() — Перефразирование и перевод
    │
    ▼ state.rephrased
retrieve_hybrid() — Гибридный поиск (BM25 + FAISS + Reranking)
    │
    ▼ state.retrieved
evaluate() — LLM-as-Judge оценка релевантности
    │
    ▼ state.relevance_score
generate() — Генерация ответа через GigaChat
    │
    ▼ state.answer
[Optional] evaluate_output() — RAGAS метрики качества
    │
    ▼ state.metrics
Output State (answer, metrics)
```

### Шаг 1: rephrase()

Перефразирует вопрос и переводит на английский:

```python
def rephrase(state: RagWorkflowState) -> RagWorkflowState:
    """
    Перефразирование и перевод вопроса на английский.
    
    Используется для улучшения поиска (retriever'ы могут быть обучены на английском).
    """
```

**Примеры:**
- "Как готовить борщ?" → "How do I prepare borscht?"
- "Рецепт салата" → "Recipe for salad"

### Шаг 2: retrieve_hybrid()

Гибридный поиск через `compression_retriever`:

```python
def retrieve_hybrid(state: RagWorkflowState) -> RagWorkflowState:
    """
    Поиск рецептов через FAISS + BM25 + Ensemble + Reranking.
    """
```

**Результаты:**
- `state.retrieved`: List[Document] top-3 рецепта
- Каждый Document содержит: текст, метаданные (row, название), scores

### Шаг 3: evaluate()

LLM-as-Judge оценивает релевантность контекста:

```python
def evaluate(state: RagWorkflowState) -> RagWorkflowState:
    """
    LLM-as-Judge оценка релевантности найденного контекста.
    """
```

**Логирует:**
- `state.relevance_score`: Float (0-1)
- Может использоваться для фильтрации низкорелевантного контекста

### Шаг 4: generate()

Генерация ответа через GigaChat:

```python
def generate(state: RagWorkflowState) -> RagWorkflowState:
    """
    Генерация ответа на основе контекста и рассуждений.
    """
```

**Выход:**
- `state.answer`: Ответ на русском языке

### Шаг 5 (Optional): evaluate_output()

RAGAS метрики качества (опционально):

```python
def evaluate_output(state: RagWorkflowState) -> RagWorkflowState:
    """
    Опциональная оценка качества ответа через RAGAS.
    """
```

**Метрики:**
- `AnswerRelevancy`: Релевантность ответа к вопросу
- `LLMContextRecall`: Содержит ли контекст необходимую информацию

### Пример использования

```python
# Инициализация
workflow = RagWorkflow(config=AgentConfig(
    enable_ragas=True,
    langfuse_enabled=True
))

# Входное состояние
input_state = {
    "question": "Как готовить борщ?"
}

# Выполнение
output_state = workflow.run(input_state)

# Результаты
print(output_state["answer"])           # Ответ на русском
print(output_state["relevance_score"])  # Оценка релевантности
print(output_state["metrics"])          # RAGAS метрики
```

### Батч-обработка

Для обработки списка вопросов:

```python
questions = [
    "Как готовить борщ?",
    "Рецепт салата",
    "Что такое пельмени?"
]

results = []
for q in questions:
    result = workflow.run({"question": q})
    results.append(result)

# Анализ результатов
for r in results:
    print(f"Q: {r['question']}")
    print(f"A: {r['answer']}")
    print(f"Score: {r['relevance_score']}")
    print()
```

---

## RAG Pipeline

Компоненты и архитектура поиска и генерации.

### Компоненты

#### Vector Store (FAISS)

FAISS используется для плотного (dense) семантического поиска:
- **Модель:** sentence-transformers (встроена в LangChain)
- **Индекс:** CPU-optimized FAISS Index
- **Поиск:** k-ближайших соседей по cosine similarity
- **Top-K:** обычно k=5-10

#### BM25 Sparse Retriever

BM25 используется для лексического (sparse) поиска:
- **Алгоритм:** BM25F с поддержкой phrase queries
- **Индекс:** в памяти или на диске
- **Поиск:** полнотекстовый по токенам
- **Top-K:** обычно k=5-10

#### Ensemble Retriever (RRF)

Комбинирование FAISS и BM25 результатов через RRF (Reciprocal Rank Fusion):

```
score(d) = Σ 1 / (k + rank_i)

где:
- k = 60 (константа для избежания деления на 0)
- rank_i = позиция документа d в результатах retriever'а i
```

**Веса:** 0.5 для FAISS + 0.5 для BM25 (равный баланс)

**Пример:**
- Document A: rank 1 в FAISS (score 0.1) + rank 3 в BM25 (score 0.016) → финальный score выше
- Document B: rank 10 в FAISS (score 0.017) + rank 1 в BM25 (score 0.1) → финальный score выше
- Документы высокие в обоих получают лучший итоговый score

#### FlashrankReranker

Переранжирование top-3 результатов через cross-encoder:

- **Модель:** ms-marco-MultiBERT-L-12
- **Входные:** (query, candidate_document)
- **Выход:** relevance score
- **Top-N:** выбираются top-3 наиболее релевантных документа

**Зачем:**
- Ensemble может выбрать не совсем релевантные документы
- Reranker использует cross-encoder для более точной оценки
- Сокращает контекст (top-3 вместо top-10) для экономии токенов

#### Generator (LLM Wrapper)

Интерфейс для GigaChat LLM:

```python
class Generator:
    def __init__(self, model: str = "GigaChat-2-Max", temperature: float = 0.3):
        """Инициализация LLM interface."""
    
    def generate(self, prompt: str, context: str) -> str:
        """Генерация ответа."""
```

**Параметры:**
- **Model:** GigaChat-2-Max
- **Temperature:** 0.3 (детерминированные ответы)
- **Max tokens:** 2000 (по умолчанию, настраивается)
- **Language:** русский на выходе

### Полный flow

```
User Query (Russian)
    │
    ▼
BM25 Search → Top-10 results
    │
    ▼
FAISS Search → Top-10 results
    │
    ▼
Ensemble (RRF) → Merged & Ranked
    │
    ▼
FlashrankReranker → Top-3 results
    │
    ▼
Generator (GigaChat) → Answer in Russian
    │
    ▼
AgentOutput / RagWorkflow State
```

### Конфигурация

```python
# RAG Pipeline configuration
RAG_CONFIG = {
    "faiss": {
        "k": 10,  # Top-K для первичного поиска
        "similarity_metric": "cosine"
    },
    "bm25": {
        "k": 10,  # Top-K для первичного поиска
        "algorithm": "BM25F"
    },
    "ensemble": {
        "weights": (0.5, 0.5),  # FAISS weight, BM25 weight
        "k": 60  # RRF denominator constant
    },
    "reranker": {
        "model": "ms-marco-MultiBERT-L-12",
        "top_n": 3  # Final top-N results
    },
    "generator": {
        "model": "GigaChat-2-Max",
        "temperature": 0.3,
        "max_tokens": 2000
    }
}
```

---

## Интеграция с HGSDLC Runtime

Агенты интегрируются с многошаговым HGSDLC процессом разработки.

### Структура узлов

```
.hgsdlc/nodes/
├── ai-describe-change/       # Описание изменения
│   └── attempt-1/
│       ├── change-request.md
│       └── step-summary.json
├── ai-explore/               # Исследование кодовой базы
│   └── attempt-1/
│       ├── exploration-notes.md
│       └── step-summary.json
├── ai-propose/               # Предложение архитектуры
│   └── attempt-1/
│       ├── proposal.md
│       ├── design.md
│       ├── specs/
│       └── step-summary.json
└── ai-apply/                 # Реализация (применение)
    └── attempt-1/
        ├── step-summary.json
        └── logs/
```

### Роли агентов в узлах

#### ai-explore: General-Purpose Agent

- **Задача:** Исследование кодовой базы, анализ текущей архитектуры
- **Input:** change-request.md из ai-describe-change
- **Output:** exploration-notes.md
- **Тип:** Programmatic (запуск в узле HGSDLC)

#### ai-propose: Architect Agent

- **Задача:** Создание proposal, design документов, specs
- **Input:** exploration-notes.md из ai-explore
- **Output:** proposal.md, design.md, specs/*.md
- **Тип:** Programmatic (запуск в узле HGSDLC)

#### ai-apply: Implementation Agent (openspec-apply skill)

- **Задача:** Реализация задач из tasks.md
- **Input:** proposal.md, design.md, specs/, tasks.md
- **Output:** Изменения в коде, step-summary.json
- **Тип:** Programmatic (запуск в узле HGSDLC)

### Контракт Input/Output

Каждый агент в узле соответствует контракту:

```python
# Input: Node-specific files (читаются из .hgsdlc/nodes/<prev>/<attempt>/)
# Output: Node-specific results (пишутся в .hgsdlc/nodes/<current>/<attempt>/)

class HGSDLCAgent:
    def run_in_node(self, node_id: str, attempt: int) -> Dict:
        """
        Запуск агента в контексте HGSDLC узла.
        
        Args:
            node_id: ID узла (e.g., "ai-explore", "ai-propose")
            attempt: Номер attempt (начиная с 1)
        
        Returns:
            Dict с результатами и step-summary
        """
```

### Обработка ошибок и retry

```python
# Если агент завершился с ошибкой
if result["status"] == "error":
    # HGSDLC Runtime может повторить узел
    # attempt -> attempt + 1
    # Агент может использовать результаты предыдущих attempts
    
    previous_attempt = load_attempt(node_id, attempt - 1)
    # Улучшить логику на основе previous_attempt
```

### Конфигурация из AGENTS.md

Узлы HGSDLC могут ссылаться на AGENTS.md:

```yaml
# Example: .hgsdlc/nodes/ai-propose/config.yaml
node_id: ai-propose
agent:
  type: architect
  model: GigaChat-2-Max
  temperature: 0.3
  reference: rag-education/AGENTS.md  # Ссылка на документацию
```

---

## Сценарии использования

### Сценарий 1: Интерактивный чат с пользователем

**Компонент:** ChiefAgent

**Flow:**
1. Пользователь открывает Streamlit приложение
2. Вводит вопрос на русском: "Как приготовить борщ?"
3. ChiefAgent принимает вопрос
4. Вызывает `cookbook` tool → поиск в RAG
5. GigaChat генерирует ответ на русском
6. Ответ выводится в UI с ссылками на рецепты

**Code:**
```python
import streamlit as st
from rag_education.agents import ChiefAgent

chief = ChiefAgent()

if user_input := st.chat_input("Ваш вопрос:"):
    result = chief.run(user_input)
    st.write(result.response)
    if result.context:
        st.write("Найденные рецепты:")
        for doc in result.context.get("retrieved_docs", []):
            st.write(f"- {doc.metadata.get('title')}")
```

### Сценарий 2: Батч-обработка запросов

**Компонент:** RagWorkflow

**Flow:**
1. Загружается список вопросов (например, из файла)
2. RagWorkflow обрабатывает каждый вопрос:
   - rephrase() + retrieve_hybrid() + evaluate() + generate()
   - Опционально: evaluate_output() для RAGAS метрик
3. Результаты сохраняются в файл или базу данных
4. Анализируются метрики качества

**Code:**
```python
from rag_education.agents import RagWorkflow

workflow = RagWorkflow(config=AgentConfig(enable_ragas=True))

questions = load_questions("questions.csv")
results = []

for q in questions:
    result = workflow.run({"question": q})
    results.append(result)

# Анализ
avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
print(f"Average relevance: {avg_relevance}")
```

### Сценарий 3: Оценка качества с RAGAS

**Компонент:** RagWorkflow с enable_ragas=True

**Flow:**
1. RagWorkflow обрабатывает вопрос
2. На шаге evaluate_output() вычисляются RAGAS метрики:
   - AnswerRelevancy: Соответствует ли ответ вопросу?
   - LLMContextRecall: Содержит ли контекст нужную информацию?
3. Метрики логируются в Langfuse
4. Используются для оптимизации RAG pipeline

**Code:**
```python
workflow = RagWorkflow(config=AgentConfig(enable_ragas=True))
result = workflow.run({"question": "Как готовить борщ?"})

print(f"Answer relevancy: {result['metrics']['answer_relevancy']}")
print(f"Context recall: {result['metrics']['context_recall']}")
```

### Сценарий 4: Использование в HGSDLC узлах

**Компонент:** RagWorkflow как Programmatic Agent

**Flow:**
1. HGSDLC Runtime вызывает узел ai-apply
2. ai-apply использует специализированный агент для реализации задач
3. Агент читает proposal.md, design.md, tasks.md
4. Создаёт/модифицирует файлы в проекте
5. Логирует прогресс в step-summary.json

**Code:**
```python
# В узле ai-apply
from rag_education.agents import ImplementationAgent

agent = ImplementationAgent()
node_context = load_hgsdlc_context("ai-apply", attempt=1)

result = agent.run_in_node(
    node_id="ai-apply",
    attempt=1,
    context=node_context
)

save_step_summary(result)
```

---

## Соглашения и правила

### Язык

**Русский снаружи, английский внутри**

- **API и интерфейсы:** Русские названия методов и параметров (для локализации)
- **Вход/выход:** Русский язык (пользовательский интерфейс)
- **Коды и идентификаторы:** Английский (model names, tool names, etc.)
- **Комментарии в коде:** Русский (для команды разработки в России)

**Примеры:**
```python
# Русский интерфейс:
result = chief.run(user_input="Как готовить борщ?")

# Английский внутри:
llm_model = "GigaChat-2-Max"
retriever = "BM25"
tool_name = "cookbook"
```

### Логирование и трейсинг

**Langfuse callbacks:**

Все операции логируются через Langfuse для observability:

```python
from langfuse.callback_handler import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="<public_key>",
    secret_key="<secret_key>"
)

# ChiefAgent и RagWorkflow используют langfuse_handler автоматически
```

**Что логируется:**
- Tool calls (cookbook выполнения)
- LLM prompts и completions
- Retriever scores (FAISS, BM25, Ensemble, Reranker)
- Processing times и latencies
- Errors и stack traces

**Доступ:**
- Langfuse UI: https://cloud.langfuse.com/
- Просмотр traces, spans, metrics

### Обработка ошибок

**Retry логика:**

1. **Transient errors** (network, timeout): автоматический retry через HGSDLC Runtime
2. **Permanent errors** (validation, not found): error status с сообщением
3. **Partial failures** (optional step fails): continue с warning log

**Error messages:**

```python
# Пример обработки ошибки
if tool_error:
    return AgentOutput(
        response="",
        status="error",
        error_message="Инструмент 'cookbook' недоступен. Пожалуйста, повторите попытку.",
        metadata={"error_code": "TOOL_UNAVAILABLE", "retry": True}
    )
```

---

## Troubleshooting

### ChiefAgent возвращает пустой ответ

**Симптомы:**
- `response` пустая строка
- `status` успешный
- `context` содержит документы

**Причины и решение:**
1. GigaChat timeout → увеличить timeout в конфигурации
2. Контекст слишком большой → reranker уже ограничивает top-3, проверить логи Langfuse
3. LLM модель недоступна → проверить API ключ для GigaChat

### Низкое качество поиска (низкая релевантность)

**Симптомы:**
- `relevance_score` < 0.5
- RAGAS `context_recall` низкий

**Причины и решение:**
1. Веса ensemble неправильные → экспериментировать с (0.4, 0.6), (0.6, 0.4)
2. Top-K слишком маленький → увеличить k=10 → k=15
3. Embedding модель неоптимальна → попробовать другую sentence-transformer модель
4. Reanker weights неправильные → настроить ms-marco-MultiBERT-L-12

### RAGAS метрики недоступны

**Симптомы:**
- `metrics` None
- Ошибка "ragas not found"

**Решение:**
1. Установить ragas: `pip install ragas`
2. Включить в конфигурации: `enable_ragas=True`
3. Убедиться что GigaChat доступен (используется для оценок)

### RagWorkflow зависает на шаге generate()

**Симптомы:**
- Процесс не завершается
- Langfuse trace не обновляется

**Причины и решение:**
1. GigaChat недоступен → проверить API статус
2. Контекст слишком большой → проверить что top-3 документов от reranker'а
3. Token limit превышен → увеличить max_tokens в конфигурации

### Интеграция с HGSDLC Runtime не работает

**Симптомы:**
- Узел ai-apply не может вызвать агента
- `step-summary.json` не создаётся

**Решение:**
1. Проверить что путь к .hgsdlc/nodes/<node>/<attempt>/ существует
2. Убедиться что агент читает input файлы из правильного места
3. Проверить permissions на запись результатов
4. Посмотреть логи в .hgsdlc/nodes/<node>/<attempt>/logs/

---

## Ссылки на спецификации

Документ основан на следующих delta-specs:

1. **agent-interface** — Universal interface requirements
2. **chief-agent** — Interactive ReAct agent specification
3. **rag-workflow** — Programmatic workflow specification
4. **rag-pipeline** — RAG components and architecture
5. **hgsdlc-integration** — HGSDLC Runtime integration

Все спецификации хранятся в:
```
rag-education/openspec/changes/generate-agents-configuration/specs/
```

---

**Последнее обновление:** 2026-09-22
