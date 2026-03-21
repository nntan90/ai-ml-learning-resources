# Notebook: 13-agent-memory-cognee

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/13-agent-memory/13-agent-memory-cognee.ipynb

---

# Lesson 13 - Agent Memory with Cognee Knowledge Graphs

## Setup

This notebook demonstrates how to build an intelligent **coding assistant** with persistent memory using [**Cognee**](https://www.cognee.ai/) knowledge graphs and the **Microsoft Agent Framework** (MAF).

Cognee transforms unstructured text into a structured, queryable knowledge graph backed by vector embeddings — giving your agent a rich, relationship-aware long-term memory.

### What You'll Learn
1. **Build Knowledge Graphs**: Transform developer profiles and best practices into structured, queryable knowledge.
2. **Integrate Cognee with MAF**: Use `@tool` functions to let an MAF agent query Cognee's knowledge graph.
3. **Session-Aware Conversations**: Maintain context across multiple questions in the same session.
4. **Long-Term Memory**: Persist important knowledge across sessions and retrieve it in new conversations.

### Prerequisites
- Python 3.9+
- Redis running locally (`docker run -d -p 6379:6379 redis`) for session management
- An LLM API key (e.g. OpenAI) — set `LLM_API_KEY` in `.env`
- `CACHING=true` in `.env` (required for Cognee sessions)
- An Azure AI Foundry project with a deployed chat model
- `AZURE_AI_PROJECT_ENDPOINT` and `AZURE_AI_MODEL_DEPLOYMENT_NAME` in `.env`
- Azure CLI authenticated (`az login`)

```python
%pip install agent-framework azure-ai-projects azure-identity "cognee[redis]==0.4.0" -q
```

```python
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "")
os.environ["CACHING"] = os.getenv("CACHING", "true")

import cognee
from cognee.modules.search.types import SearchType

from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential

print(f"Cognee version: {cognee.__version__}")
print(f"CACHING: {os.environ.get('CACHING')}")
```

```python
provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())

print("✅ AzureAIProjectAgentProvider created")
```

## Types of Agent Memory

This notebook explores the same three memory types from the main Lesson 13 notebook, but uses Cognee as the long-term memory backend:

| Memory Type | Mechanism | Lifetime |
|---|---|---|
| **Working** | `agent.create_session()` (MAF) | Single conversation |
| **Short-term** | Cognee session cache (Redis) | Single session |
| **Long-term** | Cognee knowledge graph + vectors | Permanent |

### Cognee's Memory Architecture
```
┌──────────────────────────┐
│      Raw Data            │  (developer profiles, docs, conversations)
└───────────┬──────────────┘
            │  cognee.add() + cognee.cognify()
            ▼
┌──────────────────────────────────────────┐
│  Knowledge Graph + Vector Embeddings     │
└───────────┬──────────────────────────────┘
            │  cognee.search()
            ▼
┌──────────────────┐       ┌────────────────┐
│  MAF Agent       │──────▶│  @tool funcs   │
│  (AgentSession)  │       │  wrapping       │
│                  │       │  cognee.search  │
└──────────────────┘       └────────────────┘
```

## Prepare Cognee Storage

```python
DATA_ROOT = Path('.data_storage').resolve()
SYSTEM_ROOT = Path('.cognee_system').resolve()

DATA_ROOT.mkdir(parents=True, exist_ok=True)
SYSTEM_ROOT.mkdir(parents=True, exist_ok=True)

cognee.config.data_root_directory(str(DATA_ROOT))
cognee.config.system_root_directory(str(SYSTEM_ROOT))

await cognee.prune.prune_data()
await cognee.prune.prune_system(metadata=True)
print("✅ Cognee storage configured and reset")
```

## Part 1 — Building the Knowledge Base

We ingest three types of data to create a comprehensive knowledge base for our coding assistant:

1. **Developer Profile** — personal expertise and technical background
2. **Python Best Practices** — the Zen of Python with practical guidelines
3. **Historical Conversations** — past Q&A sessions between developers and AI assistants

```python
developer_intro = (
    "Hi, I'm an AI/Backend engineer. "
    "I build FastAPI services with Pydantic, heavy asyncio/aiohttp pipelines, "
    "and production testing via pytest-asyncio. "
    "I've shipped low-latency APIs on AWS, Azure, and GoogleCloud."
)

python_zen_principles = """
# The Zen of Python: Practical Guide

## Key Principles With Guidance

### 1. Beautiful is better than ugly
Prefer descriptive names, clear structure, and consistent formatting.

### 2. Explicit is better than implicit
Be clear about behavior, imports, and types.

### 3. Simple is better than complex
Choose straightforward solutions first.

### 4. Flat is better than nested
Use early returns to reduce indentation.

## Modern Python Tie-ins
- Type hints reinforce explicitness
- Context managers enforce safe resource handling
- Dataclasses improve readability for data containers
"""

human_agent_conversations = """
"conversations": [
    {
      "topic": "async/await patterns",
      "user_query": "I'm building a web scraper that needs to handle thousands of URLs concurrently. What's the best way to structure this with asyncio?",
      "assistant_response": "Use asyncio with aiohttp, a semaphore to cap concurrency, TCPConnector for connection pooling, and context managers for session lifecycle."
    },
    {
      "topic": "dataclass vs pydantic",
      "user_query": "When should I use dataclasses vs Pydantic models?",
      "assistant_response": "For API input/output, prefer Pydantic: runtime validation, type coercion, JSON serialization. Integrates cleanly with FastAPI."
    },
    {
      "topic": "testing patterns",
      "user_query": "What's the best approach for pytest with async functions?",
      "assistant_response": "Use pytest-asyncio, async fixtures, and an isolated test database or mocks to reliably test async code."
    },
    {
      "topic": "error handling and logging",
      "user_query": "What's the best approach for production-ready error management?",
      "assistant_response": "Centralized error handling with custom exceptions, structured logging, and FastAPI middleware."
    }
  ]
"""

print("✅ Data sources prepared")
```

```python
await cognee.add(developer_intro, node_set=["developer_data"])
await cognee.add(human_agent_conversations, node_set=["developer_data"])
await cognee.add(python_zen_principles, node_set=["principles_data"])

await cognee.cognify()
print("✅ Knowledge graph built")
```

## Visualize the Knowledge Graph

Cognee can render an interactive HTML visualization of the entities and relationships it extracted.

```python
from cognee import visualize_graph

await visualize_graph('./cognee_graph.html')
print("📊 Graph saved to cognee_graph.html — open it in a browser to explore.")
```

## Enrich Memory with Memify

`memify()` analyzes the knowledge graph and generates intelligent rules — identifying patterns, best practices, and relationships between concepts.

```python
await cognee.memify()
print("✅ Memory enriched with memify")
```

## Part 2 — MAF Agent with Cognee Tools

Now we create an MAF agent that can query Cognee's knowledge graph via `@tool` functions. This lets the agent leverage the full power of graph-aware semantic search while maintaining conversational context through sessions.

```python
@tool(approval_mode="never_require")
async def search_knowledge(
    query: Annotated[str, "Natural-language question to search the knowledge graph"],
) -> str:
    """Search the Cognee knowledge graph for relevant developer knowledge, best practices, and past conversations."""
    results = await cognee.search(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
    )
    if not results:
        return "No relevant knowledge found."
    return str(results)


@tool(approval_mode="never_require")
async def search_principles(
    query: Annotated[str, "Question about Python principles or best practices"],
) -> str:
    """Search only the Python principles subset of the knowledge graph."""
    from cognee.modules.engine.models.node_set import NodeSet
    results = await cognee.search(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        node_type=NodeSet,
        node_name=["principles_data"],
    )
    if not results:
        return "No relevant principles found."
    return str(results)


print("✅ Cognee tools defined: search_knowledge, search_principles")
```

```python
coding_agent = await provider.create_agent(
    name="CodingAssistant",
    instructions=(
        "You are an expert coding assistant with access to a knowledge graph "
        "containing developer profiles, Python best practices, and past conversations.\n\n"
        "WORKFLOW:\n"
        "1. Use search_knowledge() to find relevant information from the full knowledge graph.\n"
        "2. Use search_principles() when the question is specifically about Python best practices.\n"
        "3. Combine retrieved knowledge with your own expertise to give comprehensive answers.\n"
        "4. Reference the developer's known tech stack (FastAPI, asyncio, Pydantic) when relevant."
    ),
)

print("✅ CodingAssistant agent created")
```

## Working Memory with Sessions

The `AgentSession` (created via `agent.create_session()`) provides working memory within a session. The agent can refer back to earlier messages while also querying Cognee's long-term knowledge graph.

```python
session = coding_agent.create_session()

response = await coding_agent.run(
    "How does my AsyncWebScraper implementation align with Python's design principles?",
    session=session,
)
print("🤖 Agent:", response)
```

```python
response = await coding_agent.run(
    "Based on what you just said, when should I pick dataclasses versus Pydantic for this work?",
    session=session,
)
print("🤖 Agent:", response)
print("\n💡 The agent combined working memory (previous answer) with Cognee's knowledge graph.")
```

## New Session — Long-Term Memory Persists

Starting a fresh session clears working memory, but the Cognee knowledge graph is still available. The agent can retrieve the same long-term knowledge in a completely new conversation.

```python
session_2 = coding_agent.create_session()

response = await coding_agent.run(
    "What logging guidance should I follow for incident reviews?",
    session=session_2,
)
print("🤖 Agent:", response)
print("\n💡 New session, but the agent still has access to the full Cognee knowledge graph.")
```

```python
response = await coding_agent.run(
    "How should variables be named according to Python best practices?",
    session=session_2,
)
print("🤖 Agent:", response)
```

## Summary

In this notebook you built a coding assistant that combines **MAF's working memory** (`agent.create_session()`) with **Cognee's long-term knowledge graph**.

### What You've Learned
1. **Knowledge graph construction**: Cognee ingests unstructured text and builds a graph + vector memory.
2. **Graph enrichment with memify**: Derived facts and richer relationships on top of your existing graph.
3. **MAF + Cognee integration**: `@tool` functions let an MAF agent query Cognee's graph naturally.
4. **Working memory + long-term memory**: `AgentSession` (via `agent.create_session()`) provides session context while Cognee provides persistent knowledge.
5. **Filtered search with NodeSets**: Target specific subsets of the knowledge graph (e.g. only principles).

### Key Takeaways
- **Cognee** turns raw text into structured, relationship-aware memory — more powerful than a flat vector store.
- **`@tool` functions** bridge MAF agents and external knowledge systems cleanly.
- **`AgentSession`** (via `agent.create_session()`) keeps per-conversation context separate from long-lived knowledge.
- The same knowledge graph serves multiple sessions and agents.

### Real-World Applications
- **Developer copilots**: Code review, incident analysis, architecture assistants
- **Customer-facing copilots**: Support agents over product docs, FAQs, and CRM notes
- **Internal expert copilots**: Policy, legal, or security assistants reasoning over guidelines
- **Unified data layers**: Combine structured and unstructured data into one queryable graph

### Next Steps
- Experiment with temporal awareness in Cognee
- Define an OWL ontology for domain-specific graph quality
- Add user feedback loops to improve retrieval over time
- Scale to multi-agent systems sharing the same Cognee memory layer