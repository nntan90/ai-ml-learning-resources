# Notebook: 13-agent-memory

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/13-agent-memory/13-agent-memory.ipynb

---

# Lesson 13 - Agent Memory

## Setup

This notebook demonstrates how to build a travel booking agent with **persistent memory** using the **Microsoft Agent Framework** (MAF).

You will learn how different types of agent memory — working, short-term, and long-term — affect how an agent retains and uses information across conversations.

**Prerequisites:**
- An Azure AI Foundry project with a deployed chat model (e.g. `gpt-4o-mini`).
- Logged in with the Azure CLI — run `az login` in your terminal.
- `AZURE_AI_PROJECT_ENDPOINT` — your Azure AI Foundry project endpoint.
- `AZURE_AI_MODEL_DEPLOYMENT_NAME` — the name of your deployed model.

```python
%pip install agent-framework azure-ai-projects azure-identity -q
```

```python
import logging
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)

import os
import json
from typing import Annotated
from datetime import datetime

from dotenv import load_dotenv

from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential

load_dotenv()
```

```python
provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())

print("✅ AzureAIProjectAgentProvider created")
```

## Types of Agent Memory

AI agents can leverage different kinds of memory, each serving a distinct purpose:

### Working Memory
The conversation thread itself — the messages exchanged in a single session. The agent can refer back to earlier messages in the same thread to maintain coherence. In MAF this is created via **`agent.create_session()`**, which returns an `AgentSession`.

### Short-Term Memory
Information that persists for the duration of a task or session but is not stored permanently. For example, the agent may accumulate facts during a multi-turn planning conversation and use them to produce a final itinerary.

### Long-Term Memory
Preferences and facts that persist **across sessions**. A returning user should not have to repeat their dietary restrictions or travel style. Long-term memory is typically backed by an external store — a database, file, or vector index — and surfaced to the agent via tools.

## Working Memory with Sessions

The simplest form of memory is the conversation session. When you pass the same session (created via `agent.create_session()`) to successive `agent.run()` calls, the agent sees the full history of that conversation and can recall earlier details.

Let's create a travel agent and demonstrate working memory.

```python
agent = await provider.create_agent(
    tools=[save_preference, get_preferences],
    name="TravelMemoryAgent",
    instructions=(
        "You are a travel agent who remembers user preferences across conversations. "
        "Track destinations mentioned, budget constraints, and travel dates."
    ),
)

session = agent.create_session()

# First message — the user shares preferences
response = await agent.run(
    "I love beach destinations and my budget is $3000",
    session=session,
)
print("Agent:", response)

# Second message — the agent should recall the budget from the thread
response = await agent.run(
    "What did I say my budget was?",
    session=session,
)
print("Agent:", response)
```

The agent correctly recalled the budget because both messages share the same session. This is **working memory** — it exists only for the lifetime of the session.

### What happens with a new thread?

If we create a **new** session, the agent has no memory of the previous conversation:

```python
new_session = agent.create_session()

response = await agent.run(
    "What is my budget?",
    session=new_session,
)
print("Agent:", response)
print("\n💡 The agent has no memory of the previous conversation — it's a fresh session.")
```

## Long-Term Memory Pattern

To remember user preferences **across sessions**, we need a persistent store that lives outside the conversation thread. The agent accesses this store through **tools** — functions it can call to save and retrieve information.

Below we implement a simple in-memory preference store (in production you would back this with a database or vector index) and expose it as tools the agent can use.

### Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  MAF Agent      │────▶│  @tool functions  │────▶│  Preference     │
│  (LLM)          │     │  save / retrieve  │     │  Store (dict)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                                 │
    AgentSession                                   Persists across
    (working memory)                               sessions
```

```python
# --- Persistent preference store (simulated) ---
preference_store: dict[str, list[str]] = {}


@tool(approval_mode="never_require")
def save_preference(
    user_id: Annotated[str, "User identifier"],
    preference: Annotated[str, "A travel preference to remember"],
) -> str:
    """Save a user travel preference to long-term memory."""
    preference_store.setdefault(user_id, []).append(preference)
    return f"✅ Stored: {preference}"


@tool(approval_mode="never_require")
def get_preferences(
    user_id: Annotated[str, "User identifier"],
) -> str:
    """Retrieve all saved travel preferences for a user."""
    prefs = preference_store.get(user_id, [])
    if not prefs:
        return f"No saved preferences for {user_id}."
    return "Saved preferences:\n- " + "\n- ".join(prefs)


@tool(approval_mode="never_require")
def search_hotels(
    query: Annotated[str, "Search query — location, amenities, or tags"],
) -> str:
    """Search the hotel database for matching properties."""
    hotels = [
        {"name": "Le Meurice Paris", "location": "Paris, France", "price": 850, "tags": ["luxury", "romantic", "spa"]},
        {"name": "Four Seasons Maui", "location": "Maui, Hawaii", "price": 695, "tags": ["beach", "family", "resort"]},
        {"name": "Aman Tokyo", "location": "Tokyo, Japan", "price": 780, "tags": ["luxury", "city", "spa"]},
        {"name": "Hotel Sacher Vienna", "location": "Vienna, Austria", "price": 420, "tags": ["historic", "accessible", "cultural"]},
        {"name": "Fairmont Whistler", "location": "Whistler, Canada", "price": 380, "tags": ["ski", "family", "mountain"]},
    ]
    q = query.lower()
    matches = [
        h for h in hotels
        if q in h["name"].lower()
        or q in h["location"].lower()
        or any(q in t for t in h["tags"])
    ]
    if not matches:
        matches = hotels[:3]
    return json.dumps(matches, indent=2)


print("✅ Tools defined: save_preference, get_preferences, search_hotels")
```

### Scenario 1 — First-time user books an anniversary trip

Sarah visits for the first time. The agent should store her preferences via the tools and use them to recommend hotels.

```python
travel_agent = await provider.create_agent(
    tools=[save_preference, get_preferences],
    name="TravelBookingAssistant",
    instructions=(
        "You are a personalized travel booking assistant with long-term memory.\n"
        "WORKFLOW:\n"
        "1. When a user starts a conversation, call get_preferences() to check for saved information.\n"
        "2. Store any new preferences the user mentions using save_preference().\n"
        "3. Use search_hotels() to find suitable options that match their preferences and budget.\n"
        "4. Do NOT recommend hotels that exceed the user's budget.\n\n"
        "IMPORTANT: Always use user_id='sarah_johnson_123' for all memory operations."
    ),
)

session_1 = travel_agent.create_session()

response = await travel_agent.run(
    "Hi! I'm Sarah and I'm planning a trip for my 10th wedding anniversary. "
    "We love romantic destinations, fine dining, and spa experiences. "
    "My husband has mobility issues, so we need accessible accommodations. "
    "Our budget is around $700-800 per night.",
    session=session_1,
)
print("🤖 Agent:", response)
```

```python
response = await travel_agent.run(
    "The Hotel Sacher sounds perfect! We're both vegetarian and I have a "
    "severe nut allergy. Can you note that for future trips?",
    session=session_1,
)
print("🤖 Agent:", response)
```

```python
# Verify what was stored
print("📋 Preference store contents:")
for uid, prefs in preference_store.items():
    print(f"\n  User: {uid}")
    for p in prefs:
        print(f"    - {p}")
```

### Scenario 2 — Sarah returns weeks later

Sarah starts a **brand-new thread** (simulating a new session). The working memory is empty, but the long-term preference store still has her information. The agent should retrieve it and use it to personalize recommendations.

```python
session_2 = travel_agent.create_session()  # New session — no working memory

response = await travel_agent.run(
    "Hi, my husband and I are planning another trip. Can you recommend a good hotel?",
    session=session_2,
)
print("🤖 Agent:", response)
print("\n💡 The agent retrieved Sarah's saved preferences from long-term memory "
      "even though this is a completely new conversation thread.")
```

```python
response = await travel_agent.run(
    "Great suggestions! For the Maui option, what activities would you recommend for the kids?",
    session=session_2,
)
print("🤖 Agent:", response)
```

## Summary

In this lesson you learned three types of agent memory and how to implement them with the Microsoft Agent Framework:

| Memory Type | MAF Mechanism | Lifetime |
|---|---|---|
| **Working** | `agent.create_session()` | Single conversation |
| **Short-term** | Accumulated context within a thread | Single task / session |
| **Long-term** | External store accessed via `@tool` functions | Across sessions |

### Key Takeaways
1. **`agent.create_session()`** provides working memory — the agent sees the full conversation history within a session.
2. **New sessions lose context** — without long-term memory the agent cannot recall past conversations.
3. **`@tool` functions** bridge the gap — they let the agent save and retrieve information from a persistent store.
4. **Personalization improves over time** — the more preferences are stored, the better the agent's recommendations.

### Real-World Applications
- **Customer Service**: Remember customer history and preferences
- **Personal Assistants**: Maintain context across days or weeks
- **Healthcare**: Track patient information and preferences
- **E-commerce**: Personalized shopping based on history

### Next Steps
- Replace the in-memory dict with a database or vector store (e.g. Azure AI Search)
- Add memory expiration for time-sensitive information
- Build multi-agent systems with shared memory
- Explore the Cognee notebook for knowledge-graph-backed memory