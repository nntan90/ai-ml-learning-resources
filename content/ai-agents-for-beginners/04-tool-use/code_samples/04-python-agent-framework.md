# Notebook: 04-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/04-tool-use/code_samples/04-python-agent-framework.ipynb

---

# Lesson 04 - Tool Use Design Pattern

In this lesson you will learn the **Tool Use** design pattern for AI agents using the Microsoft Agent Framework (Python). We cover:

- Defining function tools with the `@tool` decorator and typed parameters
- Providing tool schemas so the model knows what each tool does
- Controlling tool execution with `approval_mode`
- Returning **structured output** via Pydantic models and `response_format`

The scenario is a **travel booking agent** that can look up destinations, check availability, and retrieve flight information.

## Setup

```python
! pip install agent-framework azure-ai-projects azure-identity -U -q
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

```python
# Create the Azure AI Foundry provider
provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
```

## Defining Tools with the @tool Decorator

The `@tool` decorator turns a plain Python function into a tool that an agent can call.
Key points:

- The **docstring** becomes the tool description the model sees.
- **Type annotations** (including `Annotated` with descriptions) define the tool schema.
- `approval_mode` controls whether the user must approve each call before it executes.

```python
@tool(approval_mode="never_require")
def get_destinations() -> list[str]:
    """Get available vacation destinations."""
    return ["Barcelona", "Paris", "Berlin", "Tokyo", "Sydney", "New York City"]


@tool(approval_mode="never_require")
def check_availability(
    destination: Annotated[str, "The destination to check"],
) -> str:
    """Check booking availability for a destination."""
    availability = {
        "Barcelona": "Available - 3 spots left",
        "Paris": "Available",
        "Berlin": "Sold out",
        "Tokyo": "Available - 1 spot left",
        "Sydney": "Available",
        "New York City": "Available",
    }
    return availability.get(destination, "Unknown destination")


@tool(approval_mode="never_require")
def get_flight_info(
    origin: Annotated[str, "Origin airport code"],
    destination: Annotated[str, "Destination airport code"],
) -> str:
    """Get flight information between two cities."""
    flights = {
        "LHR-BCN": "BA 2042, Departs 08:30, Arrives 11:45, $350",
        "LHR-CDG": "AF 1081, Departs 09:15, Arrives 11:30, $280",
        "LHR-NRT": "JL 044, Departs 11:00, Arrives 07:00+1, $890",
    }
    return flights.get(
        f"{origin}-{destination}",
        f"No direct flights from {origin} to {destination}",
    )
```

## Creating an Agent with Multiple Tools

Pass all three tools to the client so the model can invoke whichever ones it needs to answer the user's question.

```python
travel_tools = [get_destinations, check_availability, get_flight_info]

agent = await provider.create_agent(
    name="TravelToolAgent",
    instructions="You are a travel agent. Use the available tools to answer questions about destinations, availability, and flights.",
    tools=travel_tools,
)

response = await agent.run(
    "What destinations do you have? Which ones are still available?"
)
print(response)
```

## Structured Output with Tools

By setting `response_format` to a Pydantic model, the agent is forced to return a well-typed JSON object instead of free-form text. This is useful when downstream code needs to consume the result programmatically.

```python
class BookingRecommendation(BaseModel):
    destination: str
    available: bool
    flight_details: str
    estimated_cost: int


class TravelPlan(BaseModel):
    recommendations: list[BookingRecommendation]


structured_agent = await provider.create_agent(
    name="StructuredTravelAgent",
    instructions=(
        "You are a travel agent. Use the available tools to find destinations, "
        "check availability, and get flight info. Return structured results."
    ),
    tools=[get_destinations, check_availability, get_flight_info],
)

response = await structured_agent.run(
    "I want to fly from London Heathrow to somewhere warm in Europe. "
    "Check what's available."
)
if response:
    print(response)
```

## Tool Approval Patterns

The `approval_mode` parameter on `@tool` controls whether tool calls require human approval before executing:

| Mode | Behaviour |
|---|---|
| `"never_require"` | Tool runs automatically — no user confirmation needed. |
| `"always_require"` | Every call must be approved by the user before it executes. |

Use `"always_require"` for tools that have side-effects (e.g. booking a flight, charging a credit card) so a human stays in the loop.

```python
@tool(approval_mode="always_require")
def book_flight(
    origin: Annotated[str, "Origin airport code"],
    destination: Annotated[str, "Destination airport code"],
    passenger_name: Annotated[str, "Full name of the passenger"],
) -> str:
    """Book a flight for a passenger. Requires approval before executing."""
    return (
        f"Flight booked from {origin} to {destination} "
        f"for {passenger_name}. Confirmation #TRV-2024-{hash(passenger_name) % 10000:04d}"
    )


print("Tool name:", book_flight.name)
print("Approval mode:", book_flight.approval_mode)
```

## Summary

In this lesson you learned how to:

1. **Define tools** using the `@tool` decorator with typed parameters and docstrings that serve as the tool schema.
2. **Compose multiple tools** so the agent can call them in sequence to answer complex queries.
3. **Return structured output** by passing a Pydantic model as `response_format`.
4. **Control tool approval** with `approval_mode` to keep a human in the loop for sensitive operations.

These patterns form the foundation for building reliable, production-ready agents that can interact with external systems safely.