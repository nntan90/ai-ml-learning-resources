# Notebook: 08-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/08-multi-agent/code_samples/08-python-agent-framework.ipynb

---

# Lesson 08 - Multi-Agent Design Pattern

## Setup

```python
import logging
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)

%pip install agent-framework azure-ai-projects azure-identity --quiet

import os
import asyncio

from agent_framework import AgentResponseUpdate, WorkflowBuilder
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential
```

```python
provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
```

## Why Multi-Agent Systems?

Real-world tasks like trip planning involve many different kinds of expertise — logistics, local knowledge, budgeting, and more. A single agent trying to handle everything quickly becomes unwieldy.

Multi-agent systems solve this through **specialization**: each agent focuses on one area of expertise, producing higher-quality results than a generalist. They also improve **scalability** — you can add new agents (e.g., a flight specialist, a restaurant critic) without rewriting the existing workflow. The agents compose together through a structured pipeline, passing context from one to the next.

## Creating Specialized Agents

```python
planner_agent = await provider.create_agent(
    name="TravelPlanner",
    instructions="You are a travel planning specialist. Create detailed trip itineraries based on the traveler's preferences. Include daily schedules, must-see attractions, and logistical tips.",
)

concierge_agent = await provider.create_agent(
    name="TravelConcierge",
    instructions="You are a travel concierge who reviews and enhances trip plans. Review the plan for completeness, add local insider tips, suggest restaurants, and identify potential issues. Provide your feedback in a constructive format.",
)
```

## Building a Sequential Workflow

`WorkflowBuilder` lets you wire agents into a directed graph. Here we create a simple two-step pipeline: the **TravelPlanner** drafts the itinerary, then the **TravelConcierge** reviews and enhances it.

```python
workflow = WorkflowBuilder(start_executor=planner_agent) \
    .add_edge(planner_agent, concierge_agent) \
    .build()

last_author = None
events = workflow.run("Plan a 5-day trip to Paris for a food-loving couple on a $3000 budget.", stream=True)
async for event in events:
    if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
        update = event.data
        author = update.author_name
        if author != last_author:
            if last_author is not None:
                print()
            print(f"\n{'='*50}")
            print(f"🤖 {author}:")
            print(f"{'='*50}")
            last_author = author
        print(update.text, end="", flush=True)
```

## Adding More Agents to the Workflow

One of the biggest advantages of the multi-agent pattern is how easy it is to extend. Below we add a **BudgetReviewer** agent that checks the plan against the traveler's budget, flags items that might push costs over the limit, and suggests money-saving alternatives. The workflow now runs three agents in sequence:

```
TravelPlanner → TravelConcierge → BudgetReviewer
```

```python
budget_agent = await provider.create_agent(
    name="BudgetReviewer",
    instructions="You are a budget-conscious travel advisor. Review the proposed trip plan and concierge enhancements against the traveler's stated budget. Estimate costs for flights, hotels, meals, and activities. Flag anything that risks exceeding the budget and suggest cost-saving alternatives while preserving the trip's quality.",
)

extended_workflow = WorkflowBuilder(start_executor=planner_agent) \
    .add_edge(planner_agent, concierge_agent) \
    .add_edge(concierge_agent, budget_agent) \
    .build()

last_author = None
events = extended_workflow.run("Plan a 5-day trip to Paris for a food-loving couple on a $3000 budget.", stream=True)
async for event in events:
    if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
        update = event.data
        author = update.author_name
        if author != last_author:
            if last_author is not None:
                print()
            print(f"\n{'='*50}")
            print(f"🤖 {author}:")
            print(f"{'='*50}")
            last_author = author
        print(update.text, end="", flush=True)
```

## Summary

In this lesson you learned how to:

1. **Create specialized agents** — each with a focused role (planning, concierge, budget review).
2. **Wire agents into a sequential workflow** using `WorkflowBuilder` and `add_edge`.
3. **Stream output** from a multi-agent pipeline, tracking which agent is speaking.
4. **Extend a workflow** by adding new agents to the chain without modifying existing ones.

The multi-agent design pattern keeps each agent simple while producing richer, more thoroughly reviewed results than any single agent could achieve alone.