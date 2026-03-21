# Notebook: 05-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/05-agentic-rag/code_samples/05-python-agent-framework.ipynb

---

# Lesson 05 - Agentic RAG

## Setup

This notebook demonstrates the Agentic RAG (Retrieval-Augmented Generation) pattern using the Microsoft Agent Framework.

**Prerequisites:**
- `AZURE_SEARCH_SERVICE_ENDPOINT` — your Azure AI Search service endpoint
- `AZURE_SEARCH_API_KEY` — your Azure AI Search API key
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

## What is Agentic RAG?

Traditional RAG follows a fixed pipeline: retrieve documents, then generate a response. **Agentic RAG** goes further by giving the agent autonomy to decide **when** and **how** to retrieve information.

With Agentic RAG, the agent can:
- **Decide** whether retrieval is needed before answering a question
- **Choose** which data source or tool to query
- **Evaluate** retrieved results and perform follow-up retrievals if the first attempt is insufficient
- **Combine** information from multiple retrieval steps into a coherent answer

This makes the agent more flexible and accurate compared to a static retrieve-then-generate pipeline.

## Creating a Search Tool

In Agentic RAG, external data sources are wrapped as **tools** that the agent can invoke on demand. This lets the agent treat retrieval as just another action it can take, rather than a mandatory step.

Below we define a travel knowledge base and expose it as a tool the agent can call to look up destination information.

```python
TRAVEL_KNOWLEDGE_BASE = {
    "Barcelona": "Barcelona is Spain's cosmopolitan capital of Catalonia. Best visited Mar-May or Sep-Nov. Known for Gaudí architecture, La Rambla, beaches. Average daily cost: $150-200.",
    "Tokyo": "Tokyo is Japan's capital, mixing ultramodern with traditional. Best visited Mar-Apr (cherry blossoms) or Oct-Nov. Known for Shibuya, temples, sushi. Average daily cost: $200-250.",
    "Paris": "Paris is France's capital and a global center for art, fashion, and culture. Best visited Apr-Jun or Sep-Oct. Known for Eiffel Tower, Louvre, cuisine. Average daily cost: $180-250.",
    "Cape Town": "Cape Town sits on South Africa's southwest tip. Best visited Nov-Mar. Known for Table Mountain, wine regions, wildlife. Average daily cost: $100-150.",
}


@tool(approval_mode="never_require")
def search_travel_knowledge(
    query: Annotated[str, "The search query about a travel destination"]
) -> str:
    """Search the travel knowledge base for destination information."""
    results = []
    for destination, info in TRAVEL_KNOWLEDGE_BASE.items():
        if query.lower() in destination.lower() or any(
            word in info.lower() for word in query.lower().split()
        ):
            results.append(f"**{destination}**: {info}")
    return (
        "\n\n".join(results)
        if results
        else "No matching destinations found in the knowledge base."
    )
```

## Building the RAG Agent

Now we create an agent that is instructed to **always retrieve information before answering**. The agent uses the `search_travel_knowledge` tool to ground its responses in the knowledge base rather than relying on its own training data.

```python
agent = await provider.create_agent(
    tools=[search_travel_knowledge],
    name="TravelRAGAgent",
    instructions="""You are a knowledgeable travel advisor. Before answering questions about destinations:
1. ALWAYS search the travel knowledge base first
2. Base your answers on retrieved information
3. If information is not in the knowledge base, say so clearly
4. Provide specific details like costs, best seasons, and highlights.""",
)

response = await agent.run(
    "I'm interested in visiting somewhere with great architecture. What destinations would you recommend?",
    )
print(response)
```

## Iterative Retrieval — The Maker-Checker Pattern

A key advantage of Agentic RAG is **iterative retrieval**. The agent can perform multiple rounds of search to verify, refine, or expand on its initial findings — similar to a "maker-checker" workflow:

1. **Maker step**: The agent retrieves initial information and drafts a response.
2. **Checker step**: The agent performs additional retrievals to verify details or fill gaps.

Below, the agent is asked a question that requires comparing multiple destinations, prompting it to search several times.

```python
checker_agent = await provider.create_agent(
    tools=[search_travel_knowledge],
    name="TravelRAGCheckerAgent",
    instructions="""You are a meticulous travel advisor who double-checks recommendations.
When answering travel questions:
1. Search for relevant destinations first
2. For each destination found, search again with the destination name to get full details
3. Compare the options using verified information
4. Present a final recommendation with specific costs, best travel times, and highlights
5. If any detail seems incomplete, search once more to confirm before responding.""",
)

response = await checker_agent.run(
    "I have a $175/day budget and want to travel in April. Which destinations fit my budget and timing?",
    )
print(response)
```

## Summary

In this lesson you learned how to build an **Agentic RAG** system using the Microsoft Agent Framework:

- **Agentic RAG** lets agents autonomously decide when to retrieve information, making retrieval dynamic rather than fixed.
- **Tools as data sources**: External knowledge bases (like Azure AI Search) are wrapped as tools the agent can invoke.
- **Iterative retrieval**: The maker-checker pattern enables the agent to perform multiple retrieval rounds — searching, verifying, and refining — before producing a final answer.

In production, you would replace the in-memory `TRAVEL_KNOWLEDGE_BASE` with a real Azure AI Search index to handle large-scale travel document retrieval.