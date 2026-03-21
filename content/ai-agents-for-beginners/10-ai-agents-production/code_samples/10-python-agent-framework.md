# Notebook: 10-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/10-ai-agents-production/code_samples/10-python-agent-framework.ipynb

---

# Lesson 10 - AI Agents in Production

In this lesson you will learn **production patterns** for AI agents using the Microsoft Agent Framework (Python). We cover:

- **Observability** — adding timing and logging to agent interactions
- **Evaluation** — using an evaluator agent to score response quality
- **Cost management** — strategies for token optimization and model selection

The scenario is a **travel agent** that helps users plan trips, with monitoring and evaluation layered on top.

## Setup

```python
! pip install agent-framework azure-ai-projects azure-identity -U -q
```

```python
import logging
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)

import os
import asyncio
import time
from typing import Annotated

from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential
```

```python
# Create the Azure AI Foundry provider
provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())

```

## Production Considerations

Moving AI agents from prototypes to production requires careful attention to three pillars:

1. **Observability** — You need visibility into what the agent is doing, how long it takes, and which tools it calls. Without tracing and logging, debugging production issues is nearly impossible.

2. **Evaluation** — Automated quality checks ensure the agent's responses remain accurate, complete, and helpful over time. An evaluator agent can score responses against defined criteria.

3. **Cost Management** — Token usage directly impacts cost. Strategies like prompt optimization, model selection, and caching help keep expenses under control without sacrificing quality.

## Building an Observable Agent

We define travel tools and wrap the agent call with timing so we can monitor latency. In production you would integrate with OpenTelemetry or a similar tracing backend.

```python
@tool(approval_mode="never_require")
def get_flight_info(destination: Annotated[str, "The destination city"]) -> str:
    """Get flight information for a destination."""
    flights = {
        "Paris": "BA 304, 08:30-11:45, $350",
        "Tokyo": "JL 044, 11:00-07:00+1, $890",
        "Barcelona": "VY 7821, 07:15-10:30, $280",
    }
    return flights.get(destination, f"No flights found to {destination}")


@tool(approval_mode="never_require")
def get_activity_suggestions(destination: Annotated[str, "The destination city"]) -> str:
    """Get activity suggestions for a destination."""
    activities = {
        "Paris": "Louvre Museum, Eiffel Tower, Seine River Cruise, Montmartre walking tour",
        "Tokyo": "Senso-ji Temple, Tsukiji Market tour, Shibuya Crossing, teamLab Borderless",
        "Barcelona": "Sagrada Familia, Park Güell, La Boqueria Market, Gothic Quarter walk",
    }
    return activities.get(destination, f"No activities found for {destination}")
```

```python
agent = await provider.create_agent(
    tools=[get_flight_info, get_activity_suggestions],
    name="TravelAgent",
    instructions="You are a helpful travel agent. Use the available tools to help users plan their trips. Provide comprehensive, actionable travel advice.",
)

# Simple observability: track timing
start_time = time.time()
response = await agent.run(
    "I want to plan a day trip in Paris. What flights and activities do you recommend?",
    )
elapsed = time.time() - start_time
print(f"Response ({elapsed:.2f}s):\n{response}")
```

## Evaluation Patterns

A common production pattern is to use a second agent as an **evaluator**. The evaluator scores the primary agent's response against predefined criteria such as completeness, accuracy, and helpfulness.

This enables:
- Automated quality gates before responses reach users
- Regression detection when prompts or models change
- Continuous monitoring of agent performance over time

```python
evaluator = await provider.create_agent(
    name="ResponseEvaluator",
    instructions="""You evaluate travel agent responses on these criteria:
1. Completeness (1-5): Did it cover flights AND activities?
2. Accuracy (1-5): Is the information consistent?
3. Helpfulness (1-5): Would a traveler find this actionable?
4. Overall Score (1-5)
Provide scores and a brief explanation for each.""",
)

evaluation = await evaluator.run(f"Evaluate this travel agent response:\n\n{response}")
print(f"Evaluation:\n{evaluation}")
```

## Cost Management Strategies

Controlling costs is critical for production AI agents. Here are key strategies:

| Strategy | Description |
|---|---|
| **Prompt optimization** | Keep system instructions concise. Remove redundant context to reduce input tokens. |
| **Model selection** | Use smaller, cheaper models (e.g. GPT-4o-mini) for simple tasks like classification or extraction, and reserve larger models for complex reasoning. |
| **Caching** | Cache tool results and frequent queries to avoid redundant API calls. |
| **Token budgets** | Set `max_tokens` limits to prevent unexpectedly long responses. |
| **Batching** | Group multiple user queries into a single API call where possible. |

In practice, a tiered approach works well: route straightforward requests to a fast, inexpensive model and escalate only complex queries to a more capable one.

## Summary

In this lesson you learned how to:

1. **Add observability** to agent interactions with timing and logging, laying the groundwork for tracing and monitoring.
2. **Evaluate agent responses** automatically using an evaluator agent that scores completeness, accuracy, and helpfulness.
3. **Manage costs** through prompt optimization, model selection, caching, and token budgets.

These production patterns help ensure your AI agents are reliable, measurable, and cost-effective at scale.