# Notebook: 03-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/03-agentic-design-patterns/code_samples/03-python-agent-framework.ipynb

---

# Lesson 03 - Agentic Design Patterns

In this lesson, we explore three foundational design patterns for building effective AI agents:

1. **Clear Agent Instructions** — Crafting precise, role-defining prompts that guide agent behavior
2. **Structured Output with Pydantic Models** — Ensuring agents return predictable, validated data
3. **Single Responsibility Agents** — Designing focused agents that each do one thing well

We'll apply each pattern to a **travel destination recommender** scenario, progressively building a system that can suggest destinations, check availability, and handle logistics.

## Setup

```python
%pip install agent-framework azure-ai-projects azure-identity pydantic --quiet
```

```python
import logging
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)

import os
import asyncio
from typing import Annotated
from pydantic import BaseModel
from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential
```

## Pattern 1: Clear Agent Instructions

The most impactful pattern is also the simplest: writing clear, detailed instructions for your agent.

Good instructions define:
- **Who** the agent is (persona and tone)
- **What** it should do (step-by-step responsibilities)
- **How** it should behave (constraints and style)

Below, we create a travel concierge agent with explicit instructions that shape every response it produces.

```python
provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())

agent = await provider.create_agent(
    name="TravelConcierge",
    instructions="""You are a luxury travel concierge named Alex. Your role is to:
1. Understand the traveler's preferences (budget, climate, activities)
2. Check destination availability before making recommendations
3. Provide detailed, personalized travel suggestions
4. Always mention visa requirements and best travel seasons
Be warm, professional, and enthusiastic about travel.""",
)

response = await agent.run(
    "I'd love a week-long vacation somewhere with great food and history. Budget around $2500."
)
print(response)
```

## Pattern 2: Structured Output with Pydantic Models

Free-form text is useful for conversation, but downstream systems need structured data.
By pairing **Pydantic models** with a **tool function**, we can:

- Define an exact schema for the agent's output
- Validate responses automatically
- Integrate agent results into application logic reliably

We also introduce a tool that returns destination details so the agent grounds its recommendations in real data.

```python
class DestinationRecommendation(BaseModel):
    destination: str
    available: bool
    best_season: str
    highlights: list[str]
    estimated_budget_usd: int


class TravelRecommendations(BaseModel):
    recommendations: list[DestinationRecommendation]
    personalized_note: str


@tool(approval_mode="never_require")
def get_destination_details(destination: Annotated[str, "The destination to look up"]) -> str:
    """Get details about a vacation destination."""
    details = {
        "Barcelona": "Available. Best: May-Jun. Beach, architecture, nightlife. ~$2000/week",
        "Tokyo": "Available. Best: Mar-Apr. Culture, food, technology. ~$2500/week",
        "Cape Town": "Not available. Best: Nov-Mar. Nature, wine, adventure. ~$1800/week",
    }
    return details.get(destination, f"{destination}: No information available.")


structured_agent = await provider.create_agent(
    name="StructuredTravelExpert",
    instructions="You are a travel expert. Recommend destinations based on traveler preferences. Use the get_destination_details tool.",
    tools=[get_destination_details],
)

response = await structured_agent.run(
    "Recommend 3 destinations for a culture-loving traveler with a $2500 budget"
)

if response:
    print(response)
```

## Pattern 3: Single Responsibility Agents

Complex tasks benefit from splitting work across multiple focused agents, each with a single responsibility:

- A **Destination Expert** that knows about places and availability
- A **Logistics Planner** that handles flights, hotels, and itineraries

This mirrors the software engineering principle of *separation of concerns* — each agent is easier to test, maintain, and improve independently.

```python
destination_agent = await provider.create_agent(
    name="DestinationExpert",
    tools=[get_destination_details],
    instructions="""You are a destination research specialist. Your only job is to:
1. Evaluate destinations based on traveler preferences
2. Check availability using the provided tool
3. Return a short ranked list with pros/cons
Do NOT discuss flights, hotels, or logistics — another agent handles that.""",
)

logistics_agent = await provider.create_agent(
    name="LogisticsPlanner",
    instructions="""You are a travel logistics planner. Your only job is to:
1. Create a day-by-day itinerary for the chosen destination
2. Suggest flight and hotel options within the stated budget
3. Note visa requirements and travel insurance recommendations
Do NOT recommend destinations — another agent handles that.""",
)

# Step 1: Destination Expert picks the best options
dest_response = await destination_agent.run(
    "I want a week of culture and food for under $2500. Where should I go?"
)
print("=== Destination Expert ===")
print(dest_response)

# Step 2: Logistics Planner builds the trip plan
logistics_response = await logistics_agent.run(
    f"Plan a week-long trip based on this recommendation:\n{dest_response}"
)
print("\n=== Logistics Planner ===")
print(logistics_response)
```

## Summary

In this lesson we applied three agentic design patterns to a travel recommender scenario:

| Pattern | Key Idea | Benefit |
|---|---|---|
| **Clear Instructions** | Define persona, responsibilities, and constraints up front | Consistent, on-brand agent behavior |
| **Structured Output** | Use Pydantic models as the response format | Validated, machine-readable results |
| **Single Responsibility** | Give each agent one focused job | Easier to test, maintain, and compose |

These patterns compose naturally — you can combine clear instructions with structured output inside a single-responsibility agent to build robust, production-ready systems.