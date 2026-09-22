# Specs Index: generate-agents-configuration

Change name: `generate-agents-configuration`
Change path: `rag-education/openspec/changes/generate-agents-configuration`

## Active Delta Specs

1. Capability: `agent-interface`
   Path: `openspec/changes/generate-agents-configuration/specs/agent-interface/spec.md`
   Purpose: Определение универсального интерфейса для всех типов агентов в системе, включая интерактивные и программируемые агенты

2. Capability: `chief-agent`
   Path: `openspec/changes/generate-agents-configuration/specs/chief-agent/spec.md`
   Purpose: Спецификация интерактивного ReAct-агента для работы с RAG-системой через инструмент cookbook

3. Capability: `rag-workflow`
   Path: `openspec/changes/generate-agents-configuration/specs/rag-workflow/spec.md`
   Purpose: Спецификация граф-ориентированного рабочего процесса (LangGraph) для батч-обработки с LLM-as-Judge оценкой

4. Capability: `rag-pipeline`
   Path: `openspec/changes/generate-agents-configuration/specs/rag-pipeline/spec.md`
   Purpose: Спецификация компонентов RAG (FAISS, BM25, EnsembleRetriever, FlashrankReranker, GigaChat LLM)

5. Capability: `hgsdlc-integration`
   Path: `openspec/changes/generate-agents-configuration/specs/hgsdlc-integration/spec.md`
   Purpose: Спецификация интеграции агентов с многошаговым HGSDLC Runtime процессом разработки

## Usage Contract

Для реализации, верификации и архивирования использовать только spec-файлы,
перечисленные в этом документе, как активный набор delta-specs текущего change.

Все пять спецификаций создают целостное описание архитектуры многоагентной системы проекта:
- agent-interface определяет универсальные требования
- chief-agent и rag-workflow описывают две основные линии агентов
- rag-pipeline специфицирует общую инфраструктуру (RAG компоненты)
- hgsdlc-integration связывает архитектуру с HGSDLC Runtime процессом

