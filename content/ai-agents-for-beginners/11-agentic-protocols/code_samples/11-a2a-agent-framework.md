# Notebook: 11-a2a-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/11-agentic-protocols/code_samples/11-a2a-agent-framework.ipynb

---

# Lesson 11 - Agent-to-Agent (A2A) Protocol

## Setup

```python
%pip install agent-framework azure-ai-projects azure-identity
```

```python
import os
import asyncio
from agent_framework import tool, AgentResponseUpdate, WorkflowBuilder
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential
```

```python

provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
```

## What is the A2A Protocol?

The **Agent-to-Agent (A2A) protocol** is an open standard that enables AI agents to communicate,
discover each other, and collaborate — even when they are built on different frameworks or hosted
by different services.

Key concepts:

- **Discovery** – Agents publish an *Agent Card* that describes their capabilities, making it
  easy for other agents (or orchestrators) to find the right specialist for a task.
- **Message Passing** – Agents exchange structured messages through a common protocol, so a
  request from one agent can be understood and fulfilled by another regardless of its internal
  implementation.
- **Task Lifecycle** – A2A defines states such as *submitted*, *working*, *completed*, and
  *failed*, giving the orchestrator full visibility into how a delegated task is progressing.

In this lesson we simulate A2A-style collaboration by wiring three specialized travel agents
into a workflow where each agent contributes its expertise and passes results to the next.

## Creating Specialized Travel Agents

```python
currency_agent = await provider.create_agent(
    name="CurrencyExchangeAgent",
    instructions="""You are a currency exchange specialist. You help travelers understand:
- Current exchange rates between currencies
- Best times to exchange money
- Tips for getting the best rates
When asked about a destination, provide relevant currency information.""",
)

activity_agent = await provider.create_agent(
    name="ActivityPlannerAgent",
    instructions="""You are a local activities specialist. You recommend:
- Must-see attractions and hidden gems
- Local experiences and cultural activities
- Restaurant and dining recommendations
Tailor suggestions to the traveler's interests.""",
)

travel_manager = await provider.create_agent(
    name="TravelManagerAgent",
    instructions="""You are a travel manager who coordinates between specialist agents.
When planning a trip:
1. Gather currency information from the currency specialist
2. Get activity recommendations from the activity planner
3. Synthesize everything into a cohesive travel brief
Present the final plan in an organized, easy-to-read format.""",
)
```

## Multi-Agent Collaboration via Workflow

We connect the three agents into a sequential workflow that mirrors A2A message passing:

1. **CurrencyExchangeAgent** receives the user request and produces currency guidance.
2. **ActivityPlannerAgent** receives the enriched context and adds activity recommendations.
3. **TravelManagerAgent** synthesizes both inputs into a final travel brief.

```python
workflow = WorkflowBuilder(start_executor=currency_agent) \
    .add_edge(currency_agent, activity_agent) \
    .add_edge(activity_agent, travel_manager) \
    .build()

last_author = None
events = workflow.run(
    "Plan a week-long trip to Tokyo. I love food, temples, and technology.",
    stream=True,
)
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

## Understanding A2A in Production

In a production environment the A2A protocol unlocks powerful cross-service scenarios:

| Capability | Description |
|---|---|
| **Cross-framework interop** | An agent built with one framework can delegate tasks to an agent built with any other A2A-compliant framework, enabling true cross-organization interoperability. |
| **Service boundaries** | Agents can live in separate microservices, cloud regions, or even different organisations while still collaborating seamlessly. |
| **Dynamic discovery** | An orchestrator can query an Agent Card registry at runtime to find the best-suited specialist for a given sub-task. |
| **Streaming & push notifications** | A2A supports Server-Sent Events (SSE) for real-time progress updates and push notifications for long-running tasks. |

The workflow we built above is a simplified, in-process version of this pattern. In a real
deployment each agent would expose an HTTP endpoint, publish an Agent Card, and communicate
via the A2A JSON-RPC protocol.

## Summary

In this lesson you learned:

1. **What the A2A protocol is** — an open standard for agent-to-agent discovery, messaging,
   and task management.
2. **How to create specialized agents** — a Currency Exchange agent, an Activity Planner agent,
   and a Travel Manager orchestrator.
3. **How to wire agents into a workflow** — using `WorkflowBuilder` to model sequential
   message passing between agents.
4. **How A2A works in production** — enabling cross-framework, cross-service collaboration
   with dynamic discovery and streaming updates.