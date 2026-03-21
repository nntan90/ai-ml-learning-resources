# Notebook: 01-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/01-intro-to-ai-agents/code_samples/01-python-agent-framework.ipynb

---

# Lesson 01 - Introduction to AI Agents

Welcome to the first lesson in the **AI Agents for Beginners** course!

An **AI agent** is a program that uses a large language model (LLM) as its reasoning engine and can take *actions* in the real world — calling APIs, querying databases, or running code — to accomplish a goal on behalf of a user.

In this notebook you will build your first agent: a **Travel Agent** that recommends vacation destinations. Along the way you will learn how to:

1. Connect to Azure AI Foundry Agent Service using the **Microsoft Agent Framework**.
2. Give the agent a **tool** — a plain Python function it can call.
3. Run the agent and inspect its response.
4. Stream the agent's response token-by-token.

## Setup

Before running this notebook, make sure you have:

1. **An Azure AI Foundry project** with a deployed chat model (e.g. `gpt-4o-mini`).
2. **Logged in with the Azure CLI** — run `az login` in your terminal.
3. **Set the required environment variables:**
   - `AZURE_AI_PROJECT_ENDPOINT` — your Azure AI Foundry project endpoint.
   - `AZURE_AI_MODEL_DEPLOYMENT_NAME` — the name of your deployed model.

The cell below installs the Python packages you need.

```python
%pip install agent-framework azure-ai-projects azure-identity -q
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

provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
```

## Creating Your First Agent

An agent needs two things:

- **Instructions** that tell it *who it is* and *how to behave* (a system prompt).
- **Tools** — Python functions decorated with `@tool` that the agent can call to retrieve information or perform actions.

Below we define a simple tool that returns a list of popular vacation destinations. The agent will use this tool when a user asks for travel recommendations.

```python
@tool(approval_mode="never_require")
def get_destinations() -> list[str]:
    """Get a list of popular vacation destinations."""
    return [
        "Barcelona",
        "Paris",
        "Berlin",
        "Tokyo",
        "Sydney",
        "New York City",
        "Cairo",
        "Cape Town",
        "Rio de Janeiro",
        "Bali",
    ]
```

```python
agent = await provider.create_agent(
    tools=[get_destinations],
    name="TravelAgent",
    instructions=(
        "You are a helpful travel agent. Help users find their perfect vacation "
        "destination based on their preferences. Use the get_destinations tool "
        "to see available destinations."
    ),
)

response = await agent.run(
    "I'm looking for a warm beach destination. What do you recommend?"
)
print(response)
```

## Streaming Responses

For a more interactive experience you can **stream** the agent's response. Instead of waiting for the full reply, the agent yields text chunks as they are generated. This is especially useful in chat interfaces where you want to display output in real time.

```python
async for chunk in agent.run(
    "Tell me about Tokyo as a travel destination", stream=True
):
    print(chunk, end="", flush=True)
```

## Summary

In this lesson you learned how to:

- **Create a provider** that connects to Azure AI Foundry Agent Service via `AzureAIProjectAgentProvider`.
- **Define a tool** using the `@tool` decorator so the agent can call your Python functions.
- **Run the agent** with a user message and print its response.
- **Stream responses** for real-time output.

In the next lesson we will explore agentic frameworks in more depth and learn how to give agents more powerful tools and multi-step reasoning capabilities.