# Notebook: 09-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/09-metacognition/code_samples/09-python-agent-framework.ipynb

---

# Lesson 09 - Metacognition Design Pattern

## Setup

This notebook demonstrates the Metacognition design pattern using the Microsoft Agent Framework.

**Prerequisites:**
- Azure OpenAI deployment configured via environment variables
- Azure CLI authenticated (`az login`)

```python
! pip install agent-framework azure-ai-projects azure-identity -q
```

```python
import logging
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)

import os
import asyncio
from typing import Annotated

from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential
```

```python

provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
```

## What is Metacognition?

Metacognition is **thinking about thinking**. In the context of AI agents, it means building agents that can:

- **Self-reflect** on their own outputs and reasoning process
- **Detect errors** and recover gracefully instead of failing silently
- **Evaluate** whether their responses are complete and helpful
- **Adapt** their strategy when an initial approach doesn't work (e.g., falling back to a backup system)

A metacognitive agent doesn't just answer questions — it monitors its own performance and adjusts on the fly.

## Primary and Backup Tools

A common metacognition pattern is the **fallback strategy**. The agent tries a primary tool first; if it fails (e.g., a 404 error), the agent recognizes the failure and transparently switches to a backup tool.

This mirrors real-world systems where primary services may be unavailable and agents must self-diagnose the issue before choosing an alternative path.

Below we define two flight lookup tools:
- **Primary** — covers Paris, Tokyo, and Barcelona
- **Backup** — covers Berlin, Sydney, and New York City

```python
@tool(approval_mode="never_require")
def get_flight_times(
    destination: Annotated[str, "The destination city"]
) -> str:
    """Get available flight times for a destination (primary source)."""
    flights = {
        "Paris": "Departures: 08:00, 12:30, 17:45 — from $350",
        "Tokyo": "Departures: 11:00, 23:30 — from $890",
        "Barcelona": "Departures: 07:15, 14:00, 19:30 — from $280",
    }
    if destination in flights:
        return flights[destination]
    raise Exception(f"404: No flights found for {destination} in primary system")


@tool(approval_mode="never_require")
def get_flight_times_backup(
    destination: Annotated[str, "The destination city"]
) -> str:
    """Get available flight times from backup system (used when primary fails)."""
    backup_flights = {
        "Berlin": "Departures: 09:00, 16:00 — from $220",
        "Sydney": "Departures: 22:00 — from $1200",
        "New York City": "Departures: 06:00, 10:30, 15:00, 20:00 — from $450",
    }
    return backup_flights.get(
        destination,
        f"No flights found for {destination} in any system. Please try again later.",
    )
```

## Self-Reflecting Agent with Error Recovery

The agent below is instructed to try the primary flight system first, recognize failures, and transparently fall back to the backup system. After each response it briefly self-evaluates whether it fully answered the user's question.

```python
agent = await provider.create_agent(
    tools=[get_flight_times, get_flight_times_backup],
    name="FlightBookingAgent",
    instructions="""You are a flight booking agent with self-reflection capabilities.

When looking up flights:
1. Try the primary flight system first (get_flight_times)
2. If the primary system fails (404 error), acknowledge the error and try the backup system (get_flight_times_backup)
3. Always explain to the user what happened — be transparent about fallbacks
4. If both systems fail, apologize and suggest alternatives

After each response, briefly evaluate whether your answer was complete and helpful.""",
)

# Test with a destination in primary system
print("=== Test 1: Destination in primary system ===")
response = await agent.run(
    "What flights are available to Paris?",
    )
print(response)

# Test with a destination only in backup system
print("\n=== Test 2: Destination only in backup system ===")
response = await agent.run(
    "What flights are available to Berlin?",
    )
print(response)
```

## Self-Evaluation Pattern

Another facet of metacognition is **self-evaluation**: a separate agent (or the same agent in a second pass) reviews a response for completeness, accuracy, and helpfulness.

Below we create a `ResponseEvaluator` agent that scores travel-agent responses on three dimensions.

```python
evaluation_agent = await provider.create_agent(
    tools=[get_flight_times, get_flight_times_backup],
    name="ResponseEvaluator",
    instructions="""You are a quality evaluator for travel agent responses.
Given a travel question and the agent's response, evaluate:
1. Completeness: Did it answer all parts of the question? (1-5)
2. Accuracy: Is the information correct? (1-5)
3. Helpfulness: Would a traveler find this useful? (1-5)
Provide a brief evaluation with scores and one suggestion for improvement.""",
)

# Evaluate the agent's response from Test 1
eval_prompt = f"""Question: What flights are available to Paris?
Agent Response: {response}

Please evaluate the above response."""

evaluation = await evaluation_agent.run(eval_prompt)
print("=== Self-Evaluation ===")
print(evaluation)
```

## Summary

In this lesson you learned how to build **metacognitive agents** using the Microsoft Agent Framework:

- **Self-reflection**: Agents that monitor their own reasoning and transparently communicate what happened.
- **Error recovery with fallbacks**: A primary + backup tool pattern where the agent detects failures (e.g., 404 errors) and automatically tries an alternative source.
- **Self-evaluation**: A separate evaluator agent that scores responses for completeness, accuracy, and helpfulness.

These patterns make agents more robust, transparent, and trustworthy — critical qualities for production deployments.