# Notebook: 11-mcp-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/11-agentic-protocols/code_samples/11-mcp-agent-framework.ipynb

---

# Lesson 11 - Model Context Protocol (MCP)

The **Model Context Protocol (MCP)** is an open standard that enables agents to dynamically discover and use tools, resources, and data sources at runtime. Instead of hardcoding tools into an agent, MCP lets agents connect to external servers that expose capabilities on demand.

In this lesson, you'll learn:
- What MCP is and why it matters for agent systems
- How the MCP client-server architecture works
- How to build agents that use MCP-style tool discovery

## Setup

**Prerequisites:**
- Azure AI Foundry project with a deployed model
- Run `az login` for authentication

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

## What is the Model Context Protocol (MCP)?

MCP defines a standard way for AI agents to discover and interact with external tools and data sources:

- **MCP Server**: Exposes tools, resources, and prompts via a standard protocol
- **MCP Client**: The agent runtime that connects to servers and discovers available capabilities
- **Dynamic Discovery**: Agents don't need hardcoded tools — they discover what's available at runtime

This is powerful for building extensible agent systems where new capabilities can be added without modifying the agent code.

## How MCP Works

```
┌─────────────┐     discover      ┌─────────────────┐
│  MCP Client  │ ──────────────► │   MCP Server     │
│  (Agent)     │                  │  (Tool Provider) │
│              │ ◄────────────── │                   │
│              │   tool results   │  • list_tools()  │
│              │                  │  • call_tool()   │
└─────────────┘                  │  • resources     │
                                  └─────────────────┘
```

1. The agent (MCP client) connects to an MCP server
2. The server responds with a list of available tools and their schemas
3. The agent can then call any discovered tool during its reasoning
4. Results flow back through the same protocol

## Simulating MCP Tool Discovery

Since a real MCP server requires a running server process, we'll demonstrate the pattern using `@tool` functions that simulate what an MCP-connected accommodation service would provide.

In production, these tools would be discovered dynamically from an MCP server rather than defined locally.

```python
@tool(approval_mode="never_require")
def search_accommodations(
    location: Annotated[str, "The city to search for accommodations"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"],
    check_out: Annotated[str, "Check-out date (YYYY-MM-DD)"],
    guests: Annotated[int, "Number of guests"] = 2
) -> str:
    """Search for accommodations (simulating an MCP-connected Airbnb tool).
    In production, this would be discovered via MCP from an accommodation service."""
    listings = {
        "Tokyo": [
            {"name": "Shinjuku Modern Apartment", "price": 120, "rating": 4.8},
            {"name": "Traditional Ryokan in Asakusa", "price": 200, "rating": 4.9},
            {"name": "Shibuya Studio", "price": 85, "rating": 4.5},
        ],
        "Paris": [
            {"name": "Le Marais Charming Flat", "price": 150, "rating": 4.7},
            {"name": "Montmartre Artist Loft", "price": 110, "rating": 4.6},
        ],
        "Barcelona": [
            {"name": "Gothic Quarter Penthouse", "price": 130, "rating": 4.8},
            {"name": "Barceloneta Beach Flat", "price": 95, "rating": 4.4},
        ],
    }
    results = listings.get(location, [])
    if not results:
        return f"No accommodations found in {location}"
    output = f"Accommodations in {location} ({check_in} to {check_out}, {guests} guests):\n"
    for listing in results:
        output += f"  - {listing['name']}: ${listing['price']}/night (★{listing['rating']})\n"
    return output


@tool(approval_mode="never_require")
def get_local_experiences(
    location: Annotated[str, "The city to find experiences in"],
    interest: Annotated[str, "Type of experience (food, culture, adventure, etc.)"] = "all"
) -> str:
    """Get local experiences and activities (simulating an MCP-connected tourism tool)."""
    experiences = {
        "Tokyo": {
            "food": ["Tsukiji Market Tour ($45)", "Ramen Making Class ($60)", "Sake Tasting ($35)"],
            "culture": ["Tea Ceremony ($50)", "Samurai Museum ($15)", "Sumo Tournament ($80)"],
            "adventure": ["Mt. Fuji Day Trip ($120)", "Go-kart City Tour ($80)"],
        },
        "Paris": {
            "food": ["Wine & Cheese Tasting ($55)", "Cooking Class ($90)", "Market Tour ($40)"],
            "culture": ["Louvre Guided Tour ($35)", "Montmartre Art Walk ($25)"],
        },
    }
    city_exp = experiences.get(location, {})
    if not city_exp:
        return f"No experiences found in {location}"
    if interest != "all" and interest in city_exp:
        items = city_exp[interest]
        return f"{interest.title()} experiences in {location}:\n" + "\n".join(f"  - {e}" for e in items)
    output = f"All experiences in {location}:\n"
    for cat, items in city_exp.items():
        output += f"\n  {cat.title()}:\n"
        for item in items:
            output += f"    - {item}\n"
    return output
```

## Building an Agent with MCP-Style Tools

```python
agent = await provider.create_agent(
    tools=[search_accommodations, get_local_experiences],
    name="AccommodationAgent",
    instructions="""You are an accommodation and travel experiences specialist powered by MCP-connected services.

Help travelers find the perfect place to stay and things to do. When searching:
1. Use the search_accommodations tool to find listings
2. Use the get_local_experiences tool to suggest activities
3. Compare options and make personalized recommendations
4. Consider the traveler's budget, interests, and travel style""",
)

response = await agent.run(
    "I'm visiting Tokyo for 5 nights in April with my partner. We love traditional Japanese culture and food. "
    "Find us a place to stay and suggest some experiences.",
    )
print(response)
```

## MCP in Production

In a production environment, MCP enables powerful patterns:

- **Dynamic tool discovery**: Agents connect to MCP servers and discover tools at runtime
- **Decoupled architecture**: Tool providers can update independently of the agent
- **Cross-organization sharing**: Teams can expose capabilities via MCP servers that any agent can use
- **Microsoft Agent Framework support**: MAF has built-in MCP client support via the `mcp` integration

To use a real MCP server with MAF, you would connect via `hosted_mcp_tool()` or the MCP client integration.

**Learn more:**
- [MCP Specification](https://modelcontextprotocol.io/)
- [Microsoft Agent Framework MCP Support](https://github.com/microsoft/agent-framework/tree/main/python/samples/02-agents/mcp)

## Summary

In this lesson, you learned:
- **MCP** is an open standard for dynamic tool discovery between agents and tool providers
- The **client-server architecture** lets agents discover capabilities at runtime
- MCP enables **extensible, decoupled agent systems** where tools can be added without code changes
- Microsoft Agent Framework provides **built-in MCP support** for production use