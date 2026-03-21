# Notebook: 02-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/02-explore-agentic-frameworks/code_samples/02-python-agent-framework.ipynb

---

# Lesson 02 - Exploring Microsoft Agent Framework

The **Microsoft Agent Framework (MAF)** is a unified framework for building AI agents. It provides a clean, composable architecture with four core building blocks:

- **Client** – connects to an AI model endpoint and handles communication
- **Agent** – wraps a client with instructions and tool definitions
- **Tools** – extend agent capabilities with custom functions the model can call
- **Session** – maintains conversation history for multi-turn interactions

In this lesson, we'll build a **travel booking agent** that checks destination availability using these concepts.

## Setup

```python
# Install the Microsoft Agent Framework package
! pip install agent-framework azure-ai-projects -U -q
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

## Understanding the Agent Framework Architecture

The Microsoft Agent Framework follows a layered architecture:

```
Client  →  Agent  →  Tools
                  →  Session
```

1. **Client** – An `AzureAIProjectAgentProvider` connects to an Azure OpenAI deployment. It handles authentication, request formatting, and response parsing.
2. **Agent** – Created from the client via `provider.create_agent()`, the agent combines model access with instructions (system prompt) and tools.
3. **Tools** – Python functions decorated with `@tool` that the agent can invoke to perform actions or retrieve data.
4. **Session** – An `AgentSession` object (created via `agent.create_session()`) that stores conversation history, enabling multi-turn dialogue where the agent remembers prior context.

Let's build each layer step by step.

```python
# Create the client – this is the connection to the AI model
provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
```

## Adding Tools with the @tool Decorator

Tools let agents take actions beyond generating text. The `@tool` decorator converts a regular Python function into something the agent can call.

Key points:
- Use `Annotated[type, "description"]` so the model understands each parameter.
- The docstring becomes the tool description the model sees.
- `approval_mode="never_require"` means the tool runs automatically without user confirmation.

```python
@tool(approval_mode="never_require")
def check_destination_availability(
    destination: Annotated[str, "The destination to check availability for"]
) -> str:
    """Check if a vacation destination is currently available for booking."""
    available = {
        "Barcelona": True,
        "Tokyo": True,
        "Cape Town": False,
        "Vancouver": True,
        "Dubai": False,
    }
    is_available = available.get(destination, False)
    return f"{destination} is {'available' if is_available else 'not available'} for booking."
```

## Creating an Agent with Tools

Now we combine the client, instructions, and tools into an agent. The `instructions` act as the system prompt — they define the agent's persona and behaviour.

```python
agent = await provider.create_agent(
    name="TravelAvailabilityAgent",
    instructions=(
        "You are a travel booking agent. Help users check destination availability "
        "and make recommendations. Always check availability before recommending a destination."
    ),
    tools=[check_destination_availability],
)
```

## Multi-Turn Conversations with Sessions

An `AgentSession` (created via `agent.create_session()`) keeps track of all messages in a conversation. By passing the same session to each `agent.run()` call, the agent has access to the full conversation history and can refer back to earlier messages.

We pass `tools=[check_destination_availability]` so the agent can call our availability checker during each turn.

```python
session = agent.create_session()

# Turn 1: Ask about available destinations
response = await agent.run(
    "Which destinations do you have available?",
    session=session,
)
print(f"Agent: {response}")
```

```python
# Turn 2: Follow-up question — the agent remembers the conversation
response = await agent.run(
    "I'd like to go somewhere warm. What's available?",
    session=session,
)
print(f"Agent: {response}")
```

## Summary

In this lesson you explored the four pillars of the Microsoft Agent Framework:

| Concept | What You Learned |
|---------|------------------|
| **Client** | `AzureAIProjectAgentProvider` connects to Azure OpenAI with credential-based auth |
| **Agent** | `provider.create_agent()` bundles a model connection with instructions and a name |
| **Tools** | The `@tool` decorator exposes Python functions for the agent to call |
| **Session** | `agent.create_session()` maintains conversation history across multiple turns |

These building blocks compose together to create agents that can hold natural conversations, call external functions, and maintain context — the foundation for more advanced agentic patterns in later lessons.