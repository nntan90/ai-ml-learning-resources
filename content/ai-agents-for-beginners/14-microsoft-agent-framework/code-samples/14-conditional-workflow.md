# Notebook: 14-conditional-workflow

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/14-microsoft-agent-framework/code-samples/14-conditional-workflow.ipynb

---

```python
import asyncio
import json
import os
from typing import Annotated, Any, Never

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    ai_function,
    executor,
)

# 🤖 GitHub Models or OpenAI client integration
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from IPython.display import HTML, display
from pydantic import BaseModel

print("✅ All imports successful!")
```

## Step 1: Define Pydantic Models for Structured Outputs

These models define the **schema** that agents will return. Using `response_format` with Pydantic ensures:
- ✅ Type-safe data extraction
- ✅ Automatic validation
- ✅ No parsing errors from free-text responses
- ✅ Easy conditional routing based on fields

```python
class BookingCheckResult(BaseModel):
    """Result from checking hotel availability at a destination."""

    destination: str
    has_availability: bool
    message: str


class AlternativeResult(BaseModel):
    """Suggested alternative destination when no rooms available."""

    alternative_destination: str
    reason: str


class BookingConfirmation(BaseModel):
    """Booking suggestion when rooms are available."""

    destination: str
    action: str
    message: str


print("✅ Pydantic models defined:")
print("   - BookingCheckResult (availability check)")
print("   - AlternativeResult (alternative suggestion)")
print("   - BookingConfirmation (booking confirmation)")
```

## Step 2: Create the Hotel Booking Tool

This tool is what the **availability_agent** will call to check if rooms are available. We use the `@ai_function` decorator to:
- Convert a Python function into an AI-callable tool
- Automatically generate JSON schema for the LLM
- Handle parameter validation
- Enable automatic invocation by agents

For this demo:
- **Stockholm, Seattle, Tokyo, London, Amsterdam** → Have rooms ✅
- **All other cities** → No rooms ❌

```python
@ai_function(description="Check hotel room availability for a destination city")
def hotel_booking(destination: Annotated[str, "The destination city to check for hotel rooms"]) -> str:
    """
    Simulates checking hotel room availability.
    
    Returns JSON string with availability status.
    """
    display(
        HTML(f"""
        <div style='padding: 15px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin: 10px 0;'>
            <strong>🔍 Tool Invoked:</strong> hotel_booking("{destination}")
        </div>
    """)
    )

    # Simulate availability check
    cities_with_rooms = ["stockholm", "seattle", "tokyo", "london", "amsterdam"]
    has_rooms = destination.lower() in cities_with_rooms

    result = {"has_availability": has_rooms, "destination": destination}

    return json.dumps(result)


print("✅ hotel_booking tool created with @ai_function decorator")
```

## Step 3: Define Condition Functions for Routing

These functions inspect the agent's response and determine which path to take in the workflow.

**Key Pattern:**
1. Check if the message is `AgentExecutorResponse`
2. Parse the structured output (Pydantic model)
3. Return `True` or `False` to control routing

The workflow will evaluate these conditions on **edges** to decide which executor to invoke next.

```python
def has_availability_condition(message: Any) -> bool:
    """
    Condition for routing when hotels ARE available.
    
    Returns True if the destination has hotel rooms.
    """
    if not isinstance(message, AgentExecutorResponse):
        return True  # Default to True if unexpected type

    try:
        result = BookingCheckResult.model_validate_json(message.agent_run_response.text)

        display(
            HTML(f"""
            <div style='padding: 12px; background: #c8e6c9; border-left: 4px solid #4caf50; border-radius: 4px; margin: 10px 0;'>
                <strong>✅ Condition Check:</strong> has_availability = <strong>{result.has_availability}</strong> for {result.destination}
            </div>
        """)
        )

        return result.has_availability
    except Exception as e:
        display(
            HTML(f"""
            <div style='padding: 12px; background: #ffcdd2; border-left: 4px solid #f44336; border-radius: 4px; margin: 10px 0;'>
                <strong>⚠️  Error:</strong> {str(e)}
            </div>
        """)
        )
        return False


def no_availability_condition(message: Any) -> bool:
    """
    Condition for routing when hotels are NOT available.
    
    Returns True if the destination has no hotel rooms.
    """
    if not isinstance(message, AgentExecutorResponse):
        return False

    try:
        result = BookingCheckResult.model_validate_json(message.agent_run_response.text)

        display(
            HTML(f"""
            <div style='padding: 12px; background: #ffecb3; border-left: 4px solid #ff9800; border-radius: 4px; margin: 10px 0;'>
                <strong>❌ Condition Check:</strong> no_availability for {result.destination}
            </div>
        """)
        )

        return not result.has_availability
    except Exception as e:
        return False


print("✅ Condition functions defined:")
print("   - has_availability_condition (routes when rooms exist)")
print("   - no_availability_condition (routes when no rooms)")
```

## Step 4: Create Custom Display Executor

Executors are workflow components that perform transformations or side effects. We use the `@executor` decorator to create a custom executor that displays the final result.

**Key Concepts:**
- `@executor(id="...")` - Registers a function as a workflow executor
- `WorkflowContext[Never, str]` - Type hints for input/output
- `ctx.yield_output(...)` - Yields the final workflow result

```python
@executor(id="display_result")
async def display_result(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    """
    Display the final result as workflow output.
    
    This executor receives the final agent response and yields it as the workflow output.
    """
    display(
        HTML("""
        <div style='padding: 15px; background: #f3e5f5; border-left: 4px solid #9c27b0; border-radius: 4px; margin: 10px 0;'>
            <strong>📤 Display Executor:</strong> Yielding workflow output
        </div>
    """)
    )

    await ctx.yield_output(response.agent_run_response.text)


print("✅ display_result executor created with @executor decorator")
```

## Step 5: Load Environment Variables

Configure the LLM client. This example works with:
- **GitHub Models** (Free tier with GitHub token)
- **Azure OpenAI**
- **OpenAI**

```python
# Load environment variables
load_dotenv()

# Check for GitHub Models or OpenAI
chat_client = OpenAIChatClient(base_url=os.environ.get(
    "GITHUB_ENDPOINT"), api_key=os.environ.get("GITHUB_TOKEN"), model_id="gpt-4o")
```

## Step 6: Create AI Agents with Structured Outputs

We create **three specialized agents**, each wrapped in an `AgentExecutor`:

1. **availability_agent** - Checks hotel availability using the tool
2. **alternative_agent** - Suggests alternative cities (when no rooms)
3. **booking_agent** - Encourages booking (when rooms available)

**Key Features:**
- `tools=[hotel_booking]` - Provides the tool to the agent
- `response_format=PydanticModel` - Forces structured JSON output
- `AgentExecutor(..., id="...")` - Wraps agent for workflow use

```python
# Agent 1: Check availability with tool
availability_agent = AgentExecutor(
    chat_client.create_agent(
        instructions=(
            "You are a hotel booking assistant that checks room availability. "
            "Use the hotel_booking tool to check if rooms are available at the destination. "
            "Return JSON with fields: destination (string), has_availability (bool), and message (string). "
            "The message should summarize the availability status."
        ),
        tools=[hotel_booking],
        response_format=BookingCheckResult,
    ),
    id="availability_agent",
)

# Agent 2: Suggest alternative (when no rooms)
alternative_agent = AgentExecutor(
    chat_client.create_agent(
        instructions=(
            "You are a helpful travel assistant. When a user cannot find hotels in their requested city, "
            "suggest an alternative nearby city that has availability. "
            "Return JSON with fields: alternative_destination (string) and reason (string). "
            "Make your suggestion sound appealing and helpful."
        ),
        response_format=AlternativeResult,
    ),
    id="alternative_agent",
)

# Agent 3: Suggest booking (when rooms available)
booking_agent = AgentExecutor(
    chat_client.create_agent(
        instructions=(
            "You are a booking assistant. The user has found available hotel rooms. "
            "Encourage them to book by highlighting the destination's appeal. "
            "Return JSON with fields: destination (string), action (string), and message (string). "
            "The action should be 'book_now' and message should be encouraging."
        ),
        response_format=BookingConfirmation,
    ),
    id="booking_agent",
)

display(
    HTML("""
    <div style='padding: 15px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin: 10px 0;'>
        <strong>✅ Created 3 Agents:</strong>
        <ul style='margin: 10px 0 0 0;'>
            <li><strong>availability_agent</strong> - Checks availability with hotel_booking tool</li>
            <li><strong>alternative_agent</strong> - Suggests alternative cities</li>
            <li><strong>booking_agent</strong> - Encourages booking</li>
        </ul>
    </div>
""")
)
```

## Step 7: Build the Workflow with Conditional Edges

Now we use `WorkflowBuilder` to construct the graph with conditional routing:

**Workflow Structure:**
```
availability_agent (START)
        ↓
   Evaluate conditions
        ↙         ↘
[no_availability]  [has_availability]
        ↓              ↓
alternative_agent  booking_agent
        ↓              ↓
    display_result ←───┘
```

**Key Methods:**
- `.set_start_executor(...)` - Sets the entry point
- `.add_edge(from, to, condition=...)` - Adds conditional edge
- `.build()` - Finalizes the workflow

```python
# Build the workflow with conditional routing
workflow = (
    WorkflowBuilder()
    .set_start_executor(availability_agent)
    # NO AVAILABILITY PATH
    .add_edge(availability_agent, alternative_agent, condition=no_availability_condition)
    .add_edge(alternative_agent, display_result)
    # HAS AVAILABILITY PATH
    .add_edge(availability_agent, booking_agent, condition=has_availability_condition)
    .add_edge(booking_agent, display_result)
    .build()
)

display(
    HTML("""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; margin: 10px 0;'>
        <h3 style='margin: 0 0 15px 0;'>✅ Workflow Built Successfully!</h3>
        <p style='margin: 0; line-height: 1.6;'>
            <strong>Conditional Routing:</strong><br>
            • If <strong>NO availability</strong> → alternative_agent → display_result<br>
            • If <strong>availability</strong> → booking_agent → display_result
        </p>
    </div>
""")
)
```

## Step 8: Run Test Case 1 - City WITHOUT Availability (Paris)

Let's test the **no availability** path by requesting hotels in Paris (which has no rooms in our simulation).

```python
display(
    HTML("""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>🧪 TEST CASE 1: Paris (No Availability)</h3>
        <p style='margin: 0;'>Expected workflow path: availability_agent → alternative_agent → display_result</p>
    </div>
""")
)

# Create request for Paris
request_paris = AgentExecutorRequest(
    messages=[ChatMessage(Role.USER, text="I want to book a hotel in Paris")], should_respond=True
)

# Run the workflow
events_paris = await workflow.run(request_paris)
outputs_paris = events_paris.get_outputs()

# Display results
if outputs_paris:
    result_paris = AlternativeResult.model_validate_json(outputs_paris[0])

    display(
        HTML(f"""
        <div style='padding: 25px; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); border-radius: 12px; box-shadow: 0 4px 12px rgba(255,165,0,0.3); margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0; color: #333;'>🏆 WORKFLOW RESULT (Paris)</h3>
            <div style='background: white; padding: 20px; border-radius: 8px;'>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Status:</strong> ❌ No rooms in Paris</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Alternative Suggestion:</strong> 🏨 {result_paris.alternative_destination}</p>
                <p style='margin: 0; font-size: 14px; color: #666;'><strong>Reason:</strong> {result_paris.reason}</p>
            </div>
        </div>
    """)
    )
```

## Step 9: Run Test Case 2 - City WITH Availability (Stockholm)

Now let's test the **availability** path by requesting hotels in Stockholm (which has rooms in our simulation).

```python
display(
    HTML("""
    <div style='padding: 20px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #1b5e20;'>🧪 TEST CASE 2: Stockholm (Has Availability)</h3>
        <p style='margin: 0;'>Expected workflow path: availability_agent → booking_agent → display_result</p>
    </div>
""")
)

# Create request for Stockholm
request_stockholm = AgentExecutorRequest(
    messages=[ChatMessage(Role.USER, text="I want to book a hotel in Stockholm")], should_respond=True
)

# Run the workflow
events_stockholm = await workflow.run(request_stockholm)
outputs_stockholm = events_stockholm.get_outputs()

# Display results
if outputs_stockholm:
    result_stockholm = BookingConfirmation.model_validate_json(outputs_stockholm[0])

    display(
        HTML(f"""
        <div style='padding: 25px; background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%); color: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(76,175,80,0.3); margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0;'>🏆 WORKFLOW RESULT (Stockholm)</h3>
            <div style='background: white; color: #333; padding: 20px; border-radius: 8px;'>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Status:</strong> ✅ Rooms Available!</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Destination:</strong> 🏨 {result_stockholm.destination}</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Action:</strong> {result_stockholm.action}</p>
                <p style='margin: 0; font-size: 14px; color: #666;'><strong>Message:</strong> {result_stockholm.message}</p>
            </div>
        </div>
    """)
    )
```

## Key Takeaways and Next Steps

### ✅ What You've Learned:

1. **WorkflowBuilder Pattern**
   - Use `.set_start_executor()` to define entry point
   - Use `.add_edge(from, to, condition=...)` for conditional routing
   - Call `.build()` to finalize the workflow

2. **Conditional Routing**
   - Condition functions inspect `AgentExecutorResponse`
   - Parse structured outputs to make routing decisions
   - Return `True` to activate an edge, `False` to skip it

3. **Tool Integration**
   - Use `@ai_function` to convert Python functions into AI tools
   - Agents call tools automatically when needed
   - Tools return JSON that agents can parse

4. **Structured Outputs**
   - Use Pydantic models for type-safe data extraction
   - Set `response_format=MyModel` when creating agents
   - Parse responses with `Model.model_validate_json()`

5. **Custom Executors**
   - Use `@executor(id="...")` to create workflow components
   - Executors can transform data or perform side effects
   - Use `ctx.yield_output()` to produce workflow results

### 🚀 Real-World Applications:

- **Travel Booking**: Check availability, suggest alternatives, compare options
- **Customer Service**: Route based on issue type, sentiment, priority
- **E-commerce**: Check inventory, suggest alternatives, process orders
- **Content Moderation**: Route based on toxicity scores, user flags
- **Approval Workflows**: Route based on amount, user role, risk level
- **Multi-stage Processing**: Route based on data quality, completeness

### 📚 Next Steps:

- Add more complex conditions (multiple criteria)
- Implement loops with workflow state management
- Add sub-workflows for reusable components
- Integrate with real APIs (hotel booking, inventory systems)
- Add error handling and fallback paths
- Visualize workflows with the built-in visualization tools