# Notebook: 14-middleware

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/14-microsoft-agent-framework/code-samples/14-middleware.ipynb

---

# Hotel Booking with Priority Member Middleware

This notebook demonstrates **function-based middleware** using the Microsoft Agent Framework. We build upon the conditional workflow example by adding a middleware layer that gives priority members special privileges.

## What You'll Learn:
1. **Function-Based Middleware**: Intercept and modify function results
2. **Context Access**: Read and modify `context.result` after execution
3. **Business Logic Implementation**: Priority member benefits
4. **Result Override**: Change function outcomes based on user status
5. **Same Workflow, Different Outcomes**: Middleware-driven behavior changes

## Workflow Architecture with Middleware:

```
User Input: "I want to book a hotel in Paris"
                    ↓
        [availability_agent]
        - Calls hotel_booking tool
        - 🌟 priority_check middleware intercepts
        - Checks user membership status
        - IF priority + no rooms → Override to available!
        - Returns BookingCheckResult
                    ↓
        Conditional Routing
           /                    \
    [has_availability]    [no_availability]
          ↓                      ↓
    [booking_agent]        [alternative_agent]
    (Priority override!)   (Regular users)
          ↓                      ↓
       [display_result executor]
```

## Key Difference from Conditional Workflow:

**Without Middleware** (14-conditional-workflow.ipynb):
- Paris has no rooms → Route to alternative_agent

**With Middleware** (this notebook):
- Regular user + Paris → No rooms → Route to alternative_agent
- Priority user + Paris → 🌟 Middleware overrides! → Available → Route to booking_agent

## Prerequisites:
- Microsoft Agent Framework installed
- Understanding of conditional workflows (see 14-conditional-workflow.ipynb)
- GitHub token or OpenAI API key
- Basic understanding of middleware patterns

```python
import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Never

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    FunctionInvocationContext,
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

These models define the **schema** that agents will return. We've added a `priority_override` field to track when middleware modifies the availability result.

```python
class BookingCheckResult(BaseModel):
    """Result from checking hotel availability at a destination."""

    destination: str
    has_availability: bool
    message: str
    priority_override: bool = False  # 🆕 NEW! Tracks if middleware overrode the result


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
print("   - BookingCheckResult (availability check with priority_override)")
print("   - AlternativeResult (alternative suggestion)")
print("   - BookingConfirmation (booking confirmation)")
```

## Step 2: Define Priority Members Database

For this demo, we'll simulate a priority membership database. In production, this would query a real database or API.

**Priority Members:**
- `alice@example.com` - VIP member
- `bob@example.com` - Premium member  
- `priority_user` - Test account

```python
# Simulated priority members database
PRIORITY_MEMBERS = {
    "alice@example.com",
    "bob@example.com",
    "priority_user",
}

# Global variable to track current user (in real app, use proper session management)
current_user_id = "regular_user"  # Default: regular user


def set_user(user_id: str):
    """Set the current user for the session."""
    global current_user_id
    current_user_id = user_id
    is_priority = user_id in PRIORITY_MEMBERS
    status = "🌟 PRIORITY MEMBER" if is_priority else "👤 Regular User"

    display(
        HTML(f"""
        <div style='padding: 15px; background: {"linear-gradient(135deg, #FFD700 0%, #FFA500 100%)" if is_priority else "#e3f2fd"}; 
                    border-left: 4px solid {"#FF6B35" if is_priority else "#2196f3"}; border-radius: 4px; margin: 10px 0;'>
            <strong>👤 Current User Set:</strong> {user_id}<br>
            <strong>Status:</strong> {status}
        </div>
    """)
    )


print("✅ Priority members database created")
print(f"   Priority members: {len(PRIORITY_MEMBERS)} users")
```

## Step 3: Create the Hotel Booking Tool

Same as the conditional workflow, but now it will be intercepted by middleware!

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

## Step 4: 🌟 Create Priority Check Middleware (THE KEY FEATURE!)

This is the **core functionality** of this notebook. The middleware:

1. **Intercepts** the hotel_booking function call
2. **Executes** the function normally by calling `next(context)`
3. **Inspects** the result in `context.result`
4. **Overrides** the result if user is priority and no rooms available
5. **Returns** the modified result back to the agent

**Key Pattern:**
```python
async def my_middleware(context, next):
    await next(context)  # Execute function
    # Now context.result contains the function's output
    if some_condition:
        context.result = new_value  # Override!
```

```python
async def priority_check_middleware(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    """
    Function middleware that overrides hotel_booking results for priority members.
    
    Workflow:
    1. Let the function execute normally
    2. Check if user is a priority member
    3. If priority + no availability → Override to make rooms available!
    4. Agent will then route to booking path instead of alternative path
    """
    function_name = context.function.name

    display(
        HTML(f"""
        <div style='padding: 12px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 4px; margin: 10px 0;'>
            <strong>🔄 Middleware:</strong> Intercepting {function_name}...
        </div>
    """)
    )

    # Execute the original function
    await next(context)

    # Now inspect and potentially modify the result
    if context.result and function_name == "hotel_booking":
        result_data = json.loads(context.result)
        destination = result_data.get("destination", "")
        has_availability = result_data.get("has_availability", False)

        # Check if user is priority member
        is_priority = current_user_id in PRIORITY_MEMBERS

        # Override logic: Priority member + no availability → Make available!
        if is_priority and not has_availability:
            display(
                HTML(f"""
                <div style='padding: 20px; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
                            border-radius: 8px; margin: 10px 0; box-shadow: 0 4px 12px rgba(255,165,0,0.4);'>
                    <h3 style='margin: 0 0 10px 0; color: #333;'>🌟 PRIORITY OVERRIDE ACTIVATED! 🌟</h3>
                    <p style='margin: 0; color: #555; line-height: 1.6;'>
                        <strong>User:</strong> {current_user_id}<br>
                        <strong>Status:</strong> VIP Priority Member<br>
                        <strong>Action:</strong> Overriding "No Availability" for {destination}<br>
                        <strong>Result:</strong> ✅ Rooms now available for priority booking!
                    </p>
                </div>
            """)
            )

            # Override the result!
            result_data["has_availability"] = True
            result_data["priority_override"] = True
            context.result = json.dumps(result_data)

        elif not has_availability:
            display(
                HTML(f"""
                <div style='padding: 12px; background: #ffebee; border-left: 4px solid #f44336; border-radius: 4px; margin: 10px 0;'>
                    <strong>ℹ️ Middleware:</strong> No priority override (user: {current_user_id})
                </div>
            """)
            )


print("✅ priority_check_middleware created")
print("   - Intercepts hotel_booking function")
print("   - Overrides availability for priority members")
```

## Step 5: Define Condition Functions for Routing

Same condition functions as the conditional workflow - they inspect the structured output to determine routing.

```python
def has_availability_condition(message: Any) -> bool:
    """Condition for routing when hotels ARE available (including priority overrides!)."""
    if not isinstance(message, AgentExecutorResponse):
        return True

    try:
        result = BookingCheckResult.model_validate_json(message.agent_run_response.text)

        # Check if this was a priority override
        override_indicator = " 🌟" if result.priority_override else ""

        display(
            HTML(f"""
            <div style='padding: 12px; background: #c8e6c9; border-left: 4px solid #4caf50; border-radius: 4px; margin: 10px 0;'>
                <strong>✅ Condition Check:</strong> has_availability = <strong>{result.has_availability}</strong> for {result.destination}{override_indicator}
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
    """Condition for routing when hotels are NOT available."""
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
    except Exception:
        return False


print("✅ Condition functions defined")
```

## Step 6: Create Custom Display Executor

Same executor as before - displays the final workflow output.

```python
@executor(id="display_result")
async def display_result(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    """Display the final result as workflow output."""
    display(
        HTML("""
        <div style='padding: 15px; background: #f3e5f5; border-left: 4px solid #9c27b0; border-radius: 4px; margin: 10px 0;'>
            <strong>📤 Display Executor:</strong> Yielding workflow output
        </div>
    """)
    )

    await ctx.yield_output(response.agent_run_response.text)


print("✅ display_result executor created")
```

## Step 7: Load Environment Variables

Configure the LLM client (GitHub Models or OpenAI).

```python
# Load environment variables
load_dotenv()

# Check for GitHub Models or OpenAI
chat_client = OpenAIChatClient(base_url=os.environ.get(
    "GITHUB_ENDPOINT"), api_key=os.environ.get("GITHUB_TOKEN"), model_id="gpt-4o")

```

## Step 8: Create AI Agents with Middleware

**KEY DIFFERENCE:** When creating the availability_agent, we pass the `middleware` parameter!

This is how we inject the priority_check_middleware into the agent's function invocation pipeline.

```python
# Agent 1: Check availability with tool + middleware
availability_agent = AgentExecutor(
    chat_client.create_agent(
        instructions=(
            "You are a hotel booking assistant that checks room availability. "
            "Use the hotel_booking tool to check if rooms are available at the destination. "
            "Return JSON with fields: destination (string), has_availability (bool), message (string), "
            "and priority_override (bool, true if priority member got special access). "
            "The message should summarize the availability status and mention if priority override occurred."
        ),
        tools=[hotel_booking],
        response_format=BookingCheckResult,
        middleware=[priority_check_middleware],  # 🌟 MIDDLEWARE INJECTION!
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
            "If priority_override is true in the input, mention that they received priority member access. "
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
            <li><strong>availability_agent</strong> - WITH priority_check_middleware 🌟</li>
            <li><strong>alternative_agent</strong> - Suggests alternative cities</li>
            <li><strong>booking_agent</strong> - Encourages booking</li>
        </ul>
    </div>
""")
)
```

## Step 9: Build the Workflow

Same workflow structure as before - conditional routing based on availability.

```python
# Build the workflow with conditional routing
workflow = (
    WorkflowBuilder()
    .set_start_executor(availability_agent)
    # NO AVAILABILITY PATH
    .add_edge(availability_agent, alternative_agent, condition=no_availability_condition)
    .add_edge(alternative_agent, display_result)
    # HAS AVAILABILITY PATH (can be triggered by middleware override!)
    .add_edge(availability_agent, booking_agent, condition=has_availability_condition)
    .add_edge(booking_agent, display_result)
    .build()
)

display(
    HTML("""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; margin: 10px 0;'>
        <h3 style='margin: 0 0 15px 0;'>✅ Workflow Built Successfully!</h3>
        <p style='margin: 0; line-height: 1.6;'>
            <strong>Conditional Routing (Middleware-Aware):</strong><br>
            • If <strong>NO availability</strong> → alternative_agent → display_result<br>
            • If <strong>availability</strong> (or 🌟 <strong>priority override</strong>) → booking_agent → display_result
        </p>
    </div>
""")
)
```

## Step 10: Test Case 1 - Regular User in Paris (No Override)

A regular user tries to book Paris → No rooms → Routes to alternative_agent

```python
# Set as regular user
set_user("regular_user")

display(
    HTML("""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>🧪 TEST CASE 1: Regular User + Paris</h3>
        <p style='margin: 0;'><strong>Expected:</strong> No rooms → No middleware override → Alternative suggestion</p>
    </div>
""")
)

# Create request
request_regular = AgentExecutorRequest(
    messages=[ChatMessage(Role.USER, text="I want to book a hotel in Paris")], should_respond=True
)

# Run workflow
events_regular = await workflow.run(request_regular)
outputs_regular = events_regular.get_outputs()

# Display results
if outputs_regular:
    result_regular = AlternativeResult.model_validate_json(outputs_regular[0])

    display(
        HTML(f"""
        <div style='padding: 25px; background: #fff; border: 2px solid #ff9800; border-radius: 12px; margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0; color: #e65100;'>📊 RESULT (Regular User)</h3>
            <div style='background: #fff3e0; padding: 20px; border-radius: 8px;'>
                <p style='margin: 0 0 10px 0;'><strong>Status:</strong> ❌ No rooms in Paris</p>
                <p style='margin: 0 0 10px 0;'><strong>Middleware:</strong> No priority override (regular user)</p>
                <p style='margin: 0 0 10px 0;'><strong>Alternative:</strong> 🏨 {result_regular.alternative_destination}</p>
                <p style='margin: 0;'><strong>Reason:</strong> {result_regular.reason}</p>
            </div>
        </div>
    """)
    )
```

## Step 11: Test Case 2 - 🌟 Priority User in Paris (WITH Override!)

A priority member tries to book Paris → No rooms initially → 🌟 Middleware overrides! → Routes to booking_agent

**This is the key demonstration of middleware power!**

```python
# Set as priority user
set_user("priority_user")

display(
    HTML("""
    <div style='padding: 20px; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #333;'>🧪 TEST CASE 2: 🌟 Priority User + Paris</h3>
        <p style='margin: 0; color: #555;'><strong>Expected:</strong> No rooms → 🌟 MIDDLEWARE OVERRIDE → Rooms available → Booking suggestion!</p>
    </div>
""")
)

# Create request
request_priority = AgentExecutorRequest(
    messages=[ChatMessage(Role.USER, text="I want to book a hotel in Paris")], should_respond=True
)

# Run workflow
events_priority = await workflow.run(request_priority)
outputs_priority = events_priority.get_outputs()

# Display results
if outputs_priority:
    result_priority = BookingConfirmation.model_validate_json(outputs_priority[0])

    display(
        HTML(f"""
        <div style='padding: 25px; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); border-radius: 12px; 
                    box-shadow: 0 8px 16px rgba(255,165,0,0.4); margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0; color: #333;'>🏆 RESULT (Priority Member) 🌟</h3>
            <div style='background: white; padding: 20px; border-radius: 8px;'>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Status:</strong> ✅ Rooms Available (Priority Override!)</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Middleware:</strong> 🌟 OVERRIDE ACTIVATED!</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Destination:</strong> 🏨 {result_priority.destination}</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Action:</strong> {result_priority.action}</p>
                <p style='margin: 0; font-size: 14px; color: #666;'><strong>Message:</strong> {result_priority.message}</p>
                <div style='margin-top: 15px; padding: 15px; background: #fff3cd; border-radius: 6px; border-left: 4px solid #FF6B35;'>
                    <strong>💡 What Just Happened:</strong><br>
                    1. hotel_booking tool returned "no availability"<br>
                    2. priority_check_middleware intercepted the result<br>
                    3. Middleware checked user status: priority_user ✅<br>
                    4. Middleware OVERRODE the result to "available"<br>
                    5. Workflow routed to booking_agent instead of alternative_agent!
                </div>
            </div>
        </div>
    """)
    )
```

## Step 12: Test Case 3 - Priority User in Stockholm (Already Available)

Priority user tries Stockholm → Rooms available → No override needed → Routes to booking_agent

This shows that middleware only acts when needed!

```python
# Priority user is still set from previous test

display(
    HTML("""
    <div style='padding: 20px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #1b5e20;'>🧪 TEST CASE 3: Priority User + Stockholm</h3>
        <p style='margin: 0;'><strong>Expected:</strong> Rooms available → No override needed → Booking suggestion</p>
    </div>
""")
)

# Create request
request_stockholm = AgentExecutorRequest(
    messages=[ChatMessage(Role.USER, text="I want to book a hotel in Stockholm")], should_respond=True
)

# Run workflow
events_stockholm = await workflow.run(request_stockholm)
outputs_stockholm = events_stockholm.get_outputs()

# Display results
if outputs_stockholm:
    result_stockholm = BookingConfirmation.model_validate_json(outputs_stockholm[0])

    display(
        HTML(f"""
        <div style='padding: 25px; background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%); color: white; border-radius: 12px; 
                    box-shadow: 0 4px 12px rgba(76,175,80,0.3); margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0;'>🏆 RESULT (Priority User - No Override Needed)</h3>
            <div style='background: white; color: #333; padding: 20px; border-radius: 8px;'>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Status:</strong> ✅ Rooms Available (Natural)</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Middleware:</strong> No override needed</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Destination:</strong> 🏨 {result_stockholm.destination}</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Action:</strong> {result_stockholm.action}</p>
                <p style='margin: 0; font-size: 14px; color: #666;'><strong>Message:</strong> {result_stockholm.message}</p>
                <div style='margin-top: 15px; padding: 15px; background: #e8f5e9; border-radius: 6px; border-left: 4px solid #4caf50;'>
                    <strong>💡 Middleware Behavior:</strong><br>
                    • hotel_booking returned "available" naturally<br>
                    • Middleware saw available = true → No override needed<br>
                    • Workflow proceeded normally to booking_agent
                </div>
            </div>
        </div>
    """)
    )
```

## Key Takeaways and Middleware Concepts

### ✅ What You've Learned:

#### **1. Function-Based Middleware Pattern**

Middleware intercepts function calls using a simple async function:

```python
async def my_middleware(
    context: FunctionInvocationContext,
    next: Callable,
) -> None:
    # Before function execution
    print("Intercepting...")
    
    # Execute the function
    await next(context)
    
    # After function execution - inspect result
    if context.result:
        # Modify result if needed
        context.result = modified_value
```

#### **2. Context Access and Result Override**

- `context.function` - Access the function being called
- `context.arguments` - Read function arguments
- `context.kwargs` - Access additional parameters
- `await next(context)` - Execute the function
- `context.result` - Read/modify the function's output

#### **3. Business Logic Implementation**

Our middleware implements priority member benefits:
- **Regular users**: No modifications, standard workflow
- **Priority users**: Override "no availability" → "available"
- **Conditional logic**: Only overrides when needed

#### **4. Same Workflow, Different Outcomes**

The power of middleware:
- ✅ No changes to the workflow structure
- ✅ No changes to the tool function
- ✅ No changes to conditional routing logic
- ✅ Just middleware → Different behavior!

### 🚀 Real-World Applications:

1. **VIP/Premium Features**
   - Override rate limits for premium users
   - Provide priority access to resources
   - Unlock premium features dynamically

2. **A/B Testing**
   - Route users to different implementations
   - Test new features with specific users
   - Gradual feature rollouts

3. **Security & Compliance**
   - Audit function calls
   - Block sensitive operations
   - Enforce business rules

4. **Performance Optimization**
   - Cache results for specific users
   - Skip expensive operations when possible
   - Dynamic resource allocation

5. **Error Handling & Retry**
   - Catch and handle errors gracefully
   - Implement retry logic
   - Fallback to alternative implementations

6. **Logging & Monitoring**
   - Track function execution times
   - Log parameters and results
   - Monitor usage patterns

### 🔑 Key Differences from Decorators:

| Feature | Decorator | Middleware |
|---------|-----------|------------|
| **Scope** | Single function | All functions in agent |
| **Flexibility** | Fixed at definition | Dynamic at runtime |
| **Context** | Limited | Full agent context |
| **Composition** | Multiple decorators | Middleware pipeline |
| **Agent-Aware** | No | Yes (access to agent state) |

### 📚 When to Use Middleware:

✅ **Use middleware when:**
- You need to modify behavior based on user/session state
- You want to apply logic to multiple functions
- You need access to agent-level context
- You're implementing cross-cutting concerns (logging, auth, etc.)

❌ **Don't use middleware when:**
- Simple input validation (use Pydantic)
- Function-specific logic (keep in function)
- One-time modifications (just change the function)

### 🎓 Advanced Patterns:

```python
# Multiple middleware (execution order matters!)
middleware=[
    logging_middleware,      # Logs first
    auth_middleware,         # Then checks auth
    cache_middleware,        # Then checks cache
    rate_limit_middleware,   # Then rate limits
    priority_check_middleware  # Finally priority check
]

# Conditional middleware execution
async def conditional_middleware(context, next):
    if should_execute(context):
        await next(context)
        # Modify result
    else:
        # Skip execution entirely
        context.result = cached_value
```

### 🔗 Related Concepts:

- **Agent Middleware**: Intercepts agent.run() calls
- **Function Middleware**: Intercepts tool function calls (what we used!)
- **Middleware Pipeline**: Chain of middleware executing in order
- **Context Propagation**: Pass state through middleware chain