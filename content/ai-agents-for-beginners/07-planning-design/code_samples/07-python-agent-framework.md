# Notebook: 07-python-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/07-planning-design/code_samples/07-python-agent-framework.ipynb

---

# Lesson 07 - Planning Design Pattern

This notebook demonstrates the **Planning Design Pattern** for AI agents using the Microsoft Agent Framework.
You will learn how to break complex travel requests into structured subtasks, assign them to specialist agents,
and execute the resulting plan — all using structured output powered by Pydantic models.

## Setup

```python
! pip install agent-framework azure-ai-projects azure-identity -U -q
```

```python
import logging
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)

import os, asyncio
from typing import Annotated
from pydantic import BaseModel
from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential
```

```python

provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
```

## Task Decomposition

Task decomposition is the core of the planning design pattern. Instead of asking a single agent to
handle a complex request end-to-end, we break the problem into smaller, well-defined **subtasks**.
Each subtask is assigned to a specialist agent (e.g., flights, hotels, activities) with clear
priorities and dependency ordering.

This approach provides several benefits:
- **Clarity**: each subtask has a single responsibility.
- **Parallelism**: independent subtasks can run concurrently.
- **Reliability**: failures are isolated to individual subtasks.
- **Budget tracking**: costs are estimated per subtask and rolled up.

```python
class TravelSubTask(BaseModel):
    task_id: int
    description: str
    assigned_agent: str  # "flight_agent", "hotel_agent", "activity_agent"
    priority: str  # "high", "medium", "low"
    dependencies: list[int] = []


class TravelPlan(BaseModel):
    destination: str
    trip_duration_days: int
    subtasks: list[TravelSubTask]
    total_estimated_budget_usd: int
    notes: str
```

## Creating a Planning Agent with Structured Output

The planning agent acts as a **front desk coordinator**. Given a high-level travel request it
produces a structured `TravelPlan` — decomposing the request into subtasks, setting priorities,
and identifying dependencies so that a concierge or execution layer can carry out the work.

```python
planning_agent = await provider.create_agent(
    name="TravelPlanner",
    instructions="""You are a travel planning agent. When given a travel request:
1. Break it into specific subtasks (flights, hotels, activities, logistics)
2. Assign each subtask to the appropriate specialist agent
3. Set priorities and identify dependencies between tasks
4. Estimate the total budget""",
)

result = await planning_agent.run(
    "Plan a 7-day trip to Paris for a couple interested in art, cuisine, and history. Budget around $5000.",
    )
if result:
    plan = result
    print(f"Destination: {plan.destination}")
    print(f"Duration: {plan.trip_duration_days} days")
    print(f"Budget: ${plan.total_estimated_budget_usd}")
    print(f"\nSubtasks:")
    for task in plan.subtasks:
        print(f"  [{task.priority}] {task.task_id}. {task.description} → {task.assigned_agent}")
```

## Executing a Plan with Specialist Tools

Once the front desk agent has produced a structured plan, the **concierge agent** executes it.
Each specialist tool handles one category of subtask (flights, hotels, activities). The concierge
iterates through the plan's subtasks in dependency order and dispatches each one to the
appropriate tool.

```python
@tool
def book_flight(
    destination: Annotated[str, "The destination city"],
    departure_date: Annotated[str, "Departure date (YYYY-MM-DD)"],
    return_date: Annotated[str, "Return date (YYYY-MM-DD)"],
) -> str:
    """Search and book flights for the trip."""
    return f"Flight booked to {destination}: {departure_date} → {return_date}, confirmation #FLT-{hash(destination) % 10000:04d}"


@tool
def reserve_hotel(
    city: Annotated[str, "The city for the hotel"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"],
    check_out: Annotated[str, "Check-out date (YYYY-MM-DD)"],
    guests: Annotated[int, "Number of guests"],
) -> str:
    """Reserve a hotel room in the destination city."""
    return f"Hotel reserved in {city}: {check_in} to {check_out} for {guests} guests, confirmation #HTL-{hash(city) % 10000:04d}"


@tool
def book_activity(
    activity_name: Annotated[str, "Name of the activity or tour"],
    date: Annotated[str, "Date of the activity (YYYY-MM-DD)"],
    participants: Annotated[int, "Number of participants"],
) -> str:
    """Book a tour, museum visit, or other activity."""
    return f"Activity booked: {activity_name} on {date} for {participants} people, confirmation #ACT-{hash(activity_name) % 10000:04d}"


# Concierge agent that executes the plan using specialist tools
concierge_agent = await provider.create_agent(
    name="Concierge",
    instructions="""You are a travel concierge executing a structured travel plan.
Use the available tools to fulfil each subtask. Work through the subtasks in order,
respecting dependencies. Summarise the results when finished.""",
    tools=[book_flight, reserve_hotel, book_activity],
)

# Build a prompt from the plan produced above
if result.value:
    subtask_lines = "\n".join(
        f"- [{t.priority}] {t.task_id}. {t.description} (agent: {t.assigned_agent}, deps: {t.dependencies})"
        for t in plan.subtasks
    )
    execution_prompt = (
        f"Execute the following travel plan for {plan.destination} "
        f"({plan.trip_duration_days} days, ${plan.total_estimated_budget_usd} budget):\n"
        f"{subtask_lines}"
    )

    exec_response = await concierge_agent.run(execution_prompt)
    print(exec_response)
```

## Summary

In this lesson you learned the **Planning Design Pattern** for AI agents:

1. **Task Decomposition** — A front desk planning agent breaks a complex travel request into
   structured subtasks using Pydantic models, assigning each to a specialist agent with priorities
   and dependencies.
2. **Structured Output** — By passing a `response_format` the agent returns a validated
   `TravelPlan` object instead of free-form text, making downstream processing reliable.
3. **Plan Execution** — A concierge agent iterates through the subtasks using specialist tools
   (`book_flight`, `reserve_hotel`, `book_activity`) to carry out the plan and report results.

This pattern separates *what to do* (planning) from *how to do it* (execution), making agents
more modular, testable, and easier to extend.