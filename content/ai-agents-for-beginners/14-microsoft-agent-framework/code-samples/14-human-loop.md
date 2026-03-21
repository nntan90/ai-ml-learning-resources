# Notebook: 14-human-loop

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/14-microsoft-agent-framework/code-samples/14-human-loop.ipynb

---

# Human-in-the-Loop Workflow with Microsoft Agent Framework

## 🎯 Learning Objectives

In this notebook, you'll learn how to implement **human-in-the-loop** workflows using Microsoft Agent Framework's `RequestInfoExecutor`. This powerful pattern allows you to pause AI workflows to gather human input, making your agents interactive and giving humans control over critical decisions.

## 🔄 What is Human-in-the-Loop?

**Human-in-the-loop (HITL)** is a design pattern where AI agents pause execution to request human input before continuing. This is essential for:

- ✅ **Critical decisions** - Get human approval before taking important actions
- ✅ **Ambiguous situations** - Let humans clarify when AI is uncertain
- ✅ **User preferences** - Ask users to choose between multiple options
- ✅ **Compliance & safety** - Ensure human oversight for regulated operations
- ✅ **Interactive experiences** - Build conversational agents that respond to user input

## 🏗️ How It Works in Microsoft Agent Framework

The framework provides three key components for HITL:

1. **`RequestInfoExecutor`** - A special executor that pauses the workflow and emits a `RequestInfoEvent`
2. **`RequestInfoMessage`** - Base class for typed request payloads sent to humans
3. **`RequestResponse`** - Correlates human responses with original requests using `request_id`

**Workflow Pattern:**
```
Agent detects need for input
    ↓
Sends message to RequestInfoExecutor
    ↓
Workflow pauses & emits RequestInfoEvent
    ↓
Application collects human input (console, UI, etc.)
    ↓
Application sends RequestResponse via send_responses_streaming()
    ↓
Workflow resumes with human input
```

## 🏨 Our Example: Hotel Booking with User Confirmation

We'll build upon the conditional workflow by adding human confirmation **before** suggesting alternative destinations:

1. User requests a destination (e.g., "Paris")
2. `availability_agent` checks if rooms are available
3. **If no rooms** → `confirmation_agent` asks "Would you like to see alternatives?"
4. Workflow **pauses** using `RequestInfoExecutor`
5. **Human responds** "yes" or "no" via console input
6. `decision_manager` routes based on response:
   - **Yes** → Show alternative destinations
   - **No** → Cancel booking request
7. Display final result

This demonstrates how to give users control over the agent's suggestions!

---

Let's get started! 🚀

## Step 1: Import Required Libraries

We import the standard Agent Framework components plus **human-in-the-loop specific classes**:
- `RequestInfoExecutor` - Executor that pauses workflow for human input
- `RequestInfoEvent` - Event emitted when human input is requested
- `RequestInfoMessage` - Base class for typed request payloads
- `RequestResponse` - Correlates human responses with requests
- `WorkflowOutputEvent` - Event for detecting workflow outputs

```python
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Annotated, Any, Never

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Executor,
    RequestInfoEvent,          # NEW: Event when human input is requested
    RequestInfoExecutor,       # NEW: Executor that gathers human input
    RequestInfoMessage,        # NEW: Base class for request payloads
    RequestResponse,           # NEW: Correlates response with request
    Role,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,       # NEW: Event for workflow outputs
    WorkflowRunState,          # NEW: Enum of workflow run states
    WorkflowStatusEvent,       # NEW: Event for run state changes
    ai_function,
    executor,
    handler,                   # NEW: Decorator for executor methods
)

# 🤖 GitHub Models or OpenAI client integration
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from IPython.display import HTML, display
from pydantic import BaseModel

print("✅ All imports successful!")
print("🔄 Human-in-the-loop components loaded: RequestInfoExecutor, RequestInfoEvent, RequestResponse")
```

## Step 2: Define Pydantic Models for Structured Outputs

These models define the **schema** that agents will return. We keep all models from the conditional workflow and add:

**New for Human-in-the-Loop:**
- `HumanFeedbackRequest` - Subclass of `RequestInfoMessage` that defines the request payload sent to humans
  - Contains `prompt` (question to ask) and `destination` (context about the unavailable city)

```python
# Existing models from conditional workflow
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


# NEW: Pydantic model for agent's response format
class ConfirmationQuestion(BaseModel):
    """
    Pydantic model used by confirmation_agent's response_format.
    This is what the agent will output as JSON.
    """
    question: str  # The question to ask the user
    destination: str  # The unavailable destination for context


# NEW: Dataclass for RequestInfoExecutor
@dataclass
class HumanFeedbackRequest(RequestInfoMessage):
    """
    Request sent to RequestInfoExecutor asking if user wants alternatives.
    
    MUST be a dataclass subclassing RequestInfoMessage for type compatibility.
    This is what gets sent to the RequestInfoExecutor.
    """
    prompt: str = ""  # The question to ask the user
    destination: str = ""  # The unavailable destination for context


print("✅ Pydantic models defined:")
print("   - BookingCheckResult (availability check)")
print("   - AlternativeResult (alternative suggestion)")
print("   - BookingConfirmation (booking confirmation)")
print("   - ConfirmationQuestion (agent response format) 🆕")
print("   - HumanFeedbackRequest (RequestInfoMessage for HITL) 🆕")
```

## Step 3: Create the Hotel Booking Tool

Same tool from the conditional workflow - checks if rooms are available in the destination.

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

## Step 4: Define Condition Functions for Routing

We need **four condition functions** for our human-in-the-loop workflow:

**From conditional workflow:**
1. `has_availability_condition` - Routes when hotels ARE available
2. `no_availability_condition` - Routes when hotels are NOT available

**New for human-in-the-loop:**
3. `user_wants_alternatives_condition` - Routes when user says "yes" to alternatives
4. `user_declines_alternatives_condition` - Routes when user says "no" to alternatives

```python
# Existing condition functions from conditional workflow
def has_availability_condition(message: Any) -> bool:
    """Condition for routing when hotels ARE available."""
    if not isinstance(message, AgentExecutorResponse):
        return True

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
        display(HTML(f"""<div style='padding: 12px; background: #ffcdd2; border-left: 4px solid #f44336; border-radius: 4px; margin: 10px 0;'><strong>⚠️  Error:</strong> {str(e)}</div>"""))
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
    except Exception as e:
        return False


# NEW: Condition functions for human-in-the-loop routing
def user_wants_alternatives_condition(message: Any) -> bool:
    """
    Condition for routing when user WANTS to see alternatives.
    
    Checks the AgentExecutorRequest sent by decision_manager.
    """
    # Check if it's an AgentExecutorRequest (what decision_manager sends)
    if isinstance(message, AgentExecutorRequest):
        # Check the message text to determine user's choice
        if message.messages and len(message.messages) > 0:
            msg_text = message.messages[0].text.lower()
            wants_alternatives = "wants to see alternative" in msg_text or "want to see alternative" in msg_text
            
            display(
                HTML(f"""
                <div style='padding: 12px; background: #e1f5fe; border-left: 4px solid #0288d1; border-radius: 4px; margin: 10px 0;'>
                    <strong>🔍 User Decision:</strong> User wants alternatives = <strong>{wants_alternatives}</strong>
                </div>
            """)
            )
            
            return wants_alternatives
    
    return False
def user_declines_alternatives_condition(message: Any) -> bool:
    """
    Condition for routing when user DECLINES alternatives.
    
    Checks the AgentExecutorRequest sent by decision_manager.
    """
    # Check if it's an AgentExecutorRequest (what decision_manager sends)
    if isinstance(message, AgentExecutorRequest):
        # Check the message text to determine user's choice
        if message.messages and len(message.messages) > 0:
            msg_text = message.messages[0].text.lower()
            declined = "declined" in msg_text or "has declined" in msg_text
            
            display(
                HTML(f"""
                <div style='padding: 12px; background: #fce4ec; border-left: 4px solid #c2185b; border-radius: 4px; margin: 10px 0;'>
                    <strong>🚫 User Decision:</strong> User declined alternatives = <strong>{declined}</strong>
                </div>
            """)
            )
            
            return declined
    
    return False
print("✅ Condition functions defined:")
print("   - has_availability_condition (routes when rooms exist)")
print("   - no_availability_condition (routes when no rooms)")
print("   - user_wants_alternatives_condition (routes when user says yes) 🆕")
print("   - user_declines_alternatives_condition (routes when user says no) 🆕")
```

## Step 5: Create the Decision Manager Executor

This is the **core of the human-in-the-loop pattern**! The `DecisionManager` is a custom `Executor` that:

1. **Receives human feedback** via `RequestResponse` objects
2. **Processes the user's decision** (yes/no)
3. **Routes the workflow** by sending messages to appropriate agents

Key features:
- Uses `@handler` decorator to expose methods as workflow steps
- Receives `RequestResponse[HumanFeedbackRequest, str]` containing both the original request and user's answer
- Yields simple "yes" or "no" messages that trigger our condition functions

```python
class DecisionManager(Executor):
    """
    Coordinates workflow routing based on human feedback.
    
    This executor receives RequestResponse objects from the RequestInfoExecutor
    and makes routing decisions by sending simple messages that trigger
    condition functions.
    """

    def __init__(self, id: str | None = None):
        super().__init__(id=id or "decision_manager")

    @handler
    async def on_human_feedback(
        self,
        feedback: RequestResponse[HumanFeedbackRequest, str],
        ctx: WorkflowContext[AgentExecutorRequest],
    ) -> None:
        """
        Process human feedback and let the workflow route based on conditions.
        
        The RequestResponse contains:
        - feedback.data: The user's string reply (e.g., "yes" or "no")
        - feedback.original_request: The HumanFeedbackRequest with context
        
        This handler just displays feedback and passes the RequestResponse through.
        The routing is done by condition functions on the edges.
        """
        user_reply = (feedback.data or "").strip().lower()
        destination = getattr(feedback.original_request, "destination", "unknown")

        display(
            HTML(f"""
            <div style='padding: 15px; background: #f3e5f5; border-left: 4px solid #9c27b0; border-radius: 4px; margin: 10px 0;'>
                <strong>🎯 Decision Manager:</strong> Processing user reply: <strong>"{user_reply}"</strong> for {destination}
            </div>
        """)
        )

        if user_reply == "yes":
            display(
                HTML("""
                <div style='padding: 12px; background: #c8e6c9; border-left: 4px solid #4caf50; border-radius: 4px; margin: 10px 0;'>
                    <strong>➡️  Routing:</strong> User wants alternatives → Will route to alternative_agent
                </div>
            """)
            )
            # Create and send a message for the alternative_agent
            user_msg = ChatMessage(
                Role.USER,
                text=f"The user wants to see alternative destinations near {destination}. Please suggest one.",
            )
            await ctx.send_message(AgentExecutorRequest(messages=[user_msg], should_respond=True))
        
        elif user_reply == "no":
            display(
                HTML("""
                <div style='padding: 12px; background: #ffcdd2; border-left: 4px solid #f44336; border-radius: 4px; margin: 10px 0;'>
                    <strong>🚫 Routing:</strong> User declined alternatives → Will route to cancellation_agent
                </div>
            """)
            )
            # Create and send a message for the cancellation_agent
            user_msg = ChatMessage(
                Role.USER,
                text="The user has declined to see alternatives. Please acknowledge their decision.",
            )
            await ctx.send_message(AgentExecutorRequest(messages=[user_msg], should_respond=True))
        
        else:
            # Handle unexpected input - treat as decline
            display(
                HTML(f"""
                <div style='padding: 12px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 4px; margin: 10px 0;'>
                    <strong>⚠️  Warning:</strong> Unexpected input "{user_reply}" - treating as decline
                </div>
            """)
            )
            user_msg = ChatMessage(
                Role.USER,
                text="The user has declined to see alternatives. Please acknowledge their decision.",
            )
            await ctx.send_message(AgentExecutorRequest(messages=[user_msg], should_respond=True))


print("✅ DecisionManager executor created with @handler method for human feedback")
```

## Step 6: Create Custom Display Executor

Same display executor from conditional workflow - yields final results as workflow output.

```python
@executor(id="prepare_human_request")
async def prepare_human_request(
    response: AgentExecutorResponse, 
    ctx: WorkflowContext[HumanFeedbackRequest]
) -> None:
    """
    Transform agent response into HumanFeedbackRequest for RequestInfoExecutor.
    
    This executor bridges the type gap between:
    - confirmation_agent outputs AgentExecutorResponse with ConfirmationQuestion JSON
    - request_info_executor expects HumanFeedbackRequest (RequestInfoMessage dataclass)
    """
    display(
        HTML("""
        <div style='padding: 12px; background: #e1f5fe; border-left: 4px solid #0288d1; border-radius: 4px; margin: 10px 0;'>
            <strong>🔄 Transform:</strong> Converting ConfirmationQuestion to HumanFeedbackRequest
        </div>
    """)
    )
    
    # Parse the agent's Pydantic output (ConfirmationQuestion)
    confirmation = ConfirmationQuestion.model_validate_json(response.agent_run_response.text)
    
    # Convert to HumanFeedbackRequest dataclass for RequestInfoExecutor
    feedback_request = HumanFeedbackRequest(
        prompt=confirmation.question,
        destination=confirmation.destination
    )
    
    # Send the properly typed RequestInfoMessage to the RequestInfoExecutor
    await ctx.send_message(feedback_request)


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


print("✅ prepare_human_request executor created with @executor decorator")
print("✅ display_result executor created with @executor decorator")
```

## Step 7: Load Environment Variables

Configure the LLM client (GitHub Models, Azure OpenAI, or OpenAI).

```python
# Load environment variables
load_dotenv()

# Check for GitHub Models or OpenAI
chat_client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"), 
    api_key=os.environ.get("GITHUB_TOKEN"), 
    model_id="gpt-4o"
)

print("✅ Chat client configured with GitHub Models")
```

## Step 8: Create AI Agents and Executors

We create **six workflow components**:

**Agents (wrapped in AgentExecutor):**
1. **availability_agent** - Checks hotel availability using the tool
2. **confirmation_agent** - 🆕 Prepares the human confirmation request
3. **alternative_agent** - Suggests alternative cities (when user says yes)
4. **booking_agent** - Encourages booking (when rooms available)
5. **cancellation_agent** - 🆕 Handles cancellation message (when user says no)

**Special Executors:**
6. **request_info_executor** - 🆕 `RequestInfoExecutor` that pauses workflow for human input
7. **decision_manager** - 🆕 Custom executor that routes based on human response (already defined above)

```python
# Agent 1: Check availability with tool (same as conditional workflow)
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

# Agent 2: NEW - Prepare human confirmation request
confirmation_agent = AgentExecutor(
    chat_client.create_agent(
        instructions=(
            "You are a helpful assistant. The user's requested destination has no available hotel rooms. "
            "Create a polite message asking if they would like to see alternative destinations nearby. "
            "Return a JSON with: destination (the unavailable city), and question (a friendly yes/no question). "
            "Keep the question concise and friendly."
        ),
        response_format=ConfirmationQuestion,  # Use Pydantic model for agent output
    ),
    id="confirmation_agent",
)

# Agent 3: Suggest alternative (when user says yes)
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

# Agent 4: Suggest booking (when rooms available)
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

# Agent 5: NEW - Handle cancellation when user declines alternatives
class CancellationMessage(BaseModel):
    """Message when user declines alternatives."""
    status: str
    message: str

cancellation_agent = AgentExecutor(
    chat_client.create_agent(
        instructions=(
            "You are a helpful assistant. The user has declined to see alternative hotel destinations. "
            "Create a polite cancellation message. "
            "Return JSON with: status (should be 'cancelled'), and message (a friendly acknowledgment). "
            "Keep the message brief and understanding."
        ),
        response_format=CancellationMessage,
    ),
    id="cancellation_agent",
)

# NEW: RequestInfoExecutor - pauses workflow to gather human input
request_info_executor = RequestInfoExecutor(id="request_info")

# NEW: DecisionManager instance - routes based on human feedback
decision_manager = DecisionManager(id="decision_manager")

display(
    HTML("""
    <div style='padding: 15px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin: 10px 0;'>
        <strong>✅ Created Workflow Components:</strong>
        <ul style='margin: 10px 0 0 0;'>
            <li><strong>availability_agent</strong> - Checks availability with hotel_booking tool</li>
            <li><strong>confirmation_agent</strong> 🆕 - Prepares human confirmation request</li>
            <li><strong>alternative_agent</strong> - Suggests alternative cities</li>
            <li><strong>booking_agent</strong> - Encourages booking</li>
            <li><strong>cancellation_agent</strong> 🆕 - Handles user declining alternatives</li>
            <li><strong>request_info_executor</strong> 🆕 - Pauses workflow for human input</li>
            <li><strong>decision_manager</strong> 🆕 - Routes based on human response</li>
        </ul>
    </div>
""")
)
```

## Step 9: Build the Workflow with Human-in-the-Loop

Now we construct the workflow graph with **conditional routing** including the human-in-the-loop path:

**Workflow Structure:**
```
availability_agent (START)
        ↓
   Evaluate conditions
        ↙                    ↘
[no_availability]        [has_availability]
        ↓                        ↓
confirmation_agent          booking_agent
        ↓                        ↓
prepare_human_request      display_result
        ↓
request_info_executor (PAUSE)
        ↓
decision_manager
   ↙         ↘
[yes]        [no]
   ↓           ↓
alternative  cancellation
   ↓           ↓
display_result
```

**Key Edges:**
- `availability_agent → confirmation_agent` (when no rooms)
- `confirmation_agent → prepare_human_request` (transform type)
- `prepare_human_request → request_info_executor` (pause for human)
- `request_info_executor → decision_manager` (always - provides RequestResponse)
- `decision_manager → alternative_agent` (when user says "yes")
- `decision_manager → cancellation_agent` (when user says "no")
- `availability_agent → booking_agent` (when rooms available)
- All paths end at `display_result`

```python
# Build the workflow with human-in-the-loop routing
workflow = (
    WorkflowBuilder()
    .set_start_executor(availability_agent)
    
    # NO AVAILABILITY PATH (with human-in-the-loop)
    .add_edge(availability_agent, confirmation_agent, condition=no_availability_condition)
    .add_edge(confirmation_agent, prepare_human_request)  # Transform to HumanFeedbackRequest
    .add_edge(prepare_human_request, request_info_executor)  # Send to RequestInfoExecutor
    .add_edge(request_info_executor, decision_manager)    # Always goes to decision manager
    
    # Decision manager routes based on user response
    .add_edge(decision_manager, alternative_agent, condition=user_wants_alternatives_condition)
    .add_edge(decision_manager, cancellation_agent, condition=user_declines_alternatives_condition)
    .add_edge(alternative_agent, display_result)
    .add_edge(cancellation_agent, display_result)
    
    # HAS AVAILABILITY PATH (no human input needed)
    .add_edge(availability_agent, booking_agent, condition=has_availability_condition)
    .add_edge(booking_agent, display_result)
    
    .build()
)

display(
    HTML("""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; margin: 10px 0;'>
        <h3 style='margin: 0 0 15px 0;'>✅ Workflow Built Successfully!</h3>
        <p style='margin: 0; line-height: 1.6;'>
            <strong>Human-in-the-Loop Routing:</strong><br>
            • If <strong>NO availability</strong> → confirmation_agent → prepare_human_request → request_info_executor → <strong>PAUSE FOR HUMAN</strong> → decision_manager<br>
            &nbsp;&nbsp;• If user says <strong>YES</strong> → alternative_agent → display_result<br>
            &nbsp;&nbsp;• If user says <strong>NO</strong> → cancellation_agent → display_result<br>
            • If <strong>availability</strong> → booking_agent → display_result (no human input needed)
        </p>
    </div>
""")
)
```

## Step 10: Run Test Case 1 - City WITHOUT Availability (Paris with Human Confirmation)

This test demonstrates the **full human-in-the-loop cycle**:

1. Request hotel in Paris
2. availability_agent checks → No rooms
3. confirmation_agent creates human-facing question
4. request_info_executor **pauses workflow** and emits `RequestInfoEvent`
5. **Application detects event and prompts user in console**
6. User types "yes" or "no"
7. Application sends response via `send_responses_streaming()`
8. decision_manager routes based on response
9. Final result displayed

**Key Pattern:**
- Use `workflow.run_stream()` for first iteration
- Use `workflow.send_responses_streaming(pending_responses)` for subsequent iterations
- Listen for `RequestInfoEvent` to detect when human input is needed
- Listen for `WorkflowOutputEvent` to capture final results

```python
display(
    HTML("""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>🧪 TEST CASE 1: Paris (No Availability - Human-in-the-Loop)</h3>
        <p style='margin: 0;'>Expected workflow path: availability_agent → confirmation_agent → request_info_executor → <strong>PAUSE</strong> → decision_manager → (depends on user input)</p>
    </div>
""")
)

# Create request for Paris
request_paris = AgentExecutorRequest(
    messages=[ChatMessage(Role.USER, text="I want to book a hotel in Paris")], 
    should_respond=True
)

# Human-in-the-loop execution pattern
pending_responses: dict[str, str] | None = None
completed = False
workflow_output: str | None = None

print("\n🔄 Starting human-in-the-loop workflow...")
print("=" * 60)

while not completed:
    # First iteration uses run_stream with the request
    # Subsequent iterations use send_responses_streaming with collected human responses
    if pending_responses:
        print(f"\n📤 Sending human responses: {pending_responses}")
        stream = workflow.send_responses_streaming(pending_responses)
        pending_responses = None  # Clear immediately after sending
    else:
        print(f"\n🚀 Starting workflow with request: 'I want to book a hotel in Paris'")
        stream = workflow.run_stream(request_paris)
    
    # Collect all events from this iteration
    events = [event async for event in stream]
    
    # Process events
    requests: list[tuple[str, str]] = []  # (request_id, prompt)
    
    for event in events:
        # Check for human input requests
        if isinstance(event, RequestInfoEvent) and isinstance(event.data, HumanFeedbackRequest):
            print(f"\n⏸️  WORKFLOW PAUSED - Human input requested!")
            print(f"   Request ID: {event.request_id}")
            print(f"   Destination: {event.data.destination}")
            requests.append((event.request_id, event.data.prompt))
        
        # Check for workflow outputs
        elif isinstance(event, WorkflowOutputEvent):
            workflow_output = str(event.data)
            completed = True
            print(f"\n✅ Workflow completed with output!")
    
    # If we have human requests, prompt the user
    if requests and not completed:
        responses: dict[str, str] = {}
        for req_id, prompt in requests:
            print(f"\n{'='*60}")
            print(f"💬 QUESTION FOR YOU:")
            print(f"   {prompt}")
            print(f"{'='*60}")
            
            # Get user input (in notebook, this will pause execution)
            answer = input("👉 Enter 'yes' or 'no': ").strip().lower()
            
            print(f"\n📝 You answered: {answer}")
            responses[req_id] = answer
        
        pending_responses = responses

print(f"\n{'='*60}")
print(f"🏆 FINAL WORKFLOW OUTPUT:")
print(f"{'='*60}")

# Display final result
if workflow_output:
    # Try to parse as JSON for pretty display
    try:
        result_data = json.loads(workflow_output)
        if "alternative_destination" in result_data:
            result_obj = AlternativeResult.model_validate_json(workflow_output)
            display(
                HTML(f"""
                <div style='padding: 25px; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); border-radius: 12px; box-shadow: 0 4px 12px rgba(255,165,0,0.3); margin: 20px 0;'>
                    <h3 style='margin: 0 0 15px 0; color: #333;'>🏆 WORKFLOW RESULT</h3>
                    <div style='background: white; padding: 20px; border-radius: 8px;'>
                        <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Status:</strong> ❌ No rooms in Paris</p>
                        <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>User Decision:</strong> ✅ Accepted alternatives</p>
                        <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Alternative Suggestion:</strong> 🏨 {result_obj.alternative_destination}</p>
                        <p style='margin: 0; font-size: 14px; color: #666;'><strong>Reason:</strong> {result_obj.reason}</p>
                    </div>
                </div>
            """)
            )
        else:
            # User declined
            display(
                HTML(f"""
                <div style='padding: 25px; background: linear-gradient(135deg, #f44336 0%, #e91e63 100%); color: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(244,67,54,0.3); margin: 20px 0;'>
                    <h3 style='margin: 0 0 15px 0;'>🏆 WORKFLOW RESULT</h3>
                    <div style='background: white; color: #333; padding: 20px; border-radius: 8px;'>
                        <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Status:</strong> ❌ No rooms in Paris</p>
                        <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>User Decision:</strong> 🚫 Declined alternatives</p>
                        <p style='margin: 0; font-size: 14px; color: #666;'><strong>Result:</strong> Booking request cancelled</p>
                    </div>
                </div>
            """)
            )
    except:
        print(workflow_output)
```

## Step 11: Run Test Case 2 - City WITH Availability (Stockholm - No Human Input Needed)

This test demonstrates the **direct path** when rooms are available:

1. Request hotel in Stockholm
2. availability_agent checks → Rooms available ✅
3. booking_agent suggests booking
4. display_result shows confirmation
5. **No human input required!**

The workflow bypasses the human-in-the-loop path entirely when rooms are available.

```python
display(
    HTML("""
    <div style='padding: 20px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #1b5e20;'>🧪 TEST CASE 2: Stockholm (Has Availability - No Human Input)</h3>
        <p style='margin: 0;'>Expected workflow path: availability_agent → booking_agent → display_result (direct, no pause)</p>
    </div>
""")
)

# Create request for Stockholm
request_stockholm = AgentExecutorRequest(
    messages=[ChatMessage(Role.USER, text="I want to book a hotel in Stockholm")], 
    should_respond=True
)

# Run the workflow (should complete without human input)
events_stockholm = await workflow.run(request_stockholm)
outputs_stockholm = events_stockholm.get_outputs()

# Display results
if outputs_stockholm:
    result_stockholm = BookingConfirmation.model_validate_json(outputs_stockholm[0])

    display(
        HTML(f"""
        <div style='padding: 25px; background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%); color: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(76,175,80,0.3); margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0;'>🏆 WORKFLOW RESULT (Stockholm - No Human Input)</h3>
            <div style='background: white; color: #333; padding: 20px; border-radius: 8px;'>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Status:</strong> ✅ Rooms Available!</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Destination:</strong> 🏨 {result_stockholm.destination}</p>
                <p style='margin: 0 0 10px 0; font-size: 16px;'><strong>Action:</strong> {result_stockholm.action}</p>
                <p style='margin: 0 0 10px 0; font-size: 14px; color: #666;'><strong>Message:</strong> {result_stockholm.message}</p>
                <p style='margin: 10px 0 0 0; font-size: 12px; color: #999; font-style: italic;'>Note: No human input was requested because rooms were available!</p>
            </div>
        </div>
    """)
    )
```

## Key Takeaways and Human-in-the-Loop Best Practices

### ✅ What You've Learned:

#### 1. **RequestInfoExecutor Pattern**
The human-in-the-loop pattern in Microsoft Agent Framework uses three key components:
- `RequestInfoExecutor` - Pauses workflow and emits events
- `RequestInfoMessage` - Base class for typed request payloads (subclass this!)
- `RequestResponse` - Correlates human responses with original requests

**Critical Understanding:**
- `RequestInfoExecutor` does NOT collect input itself - it only pauses the workflow
- Your application code must listen for `RequestInfoEvent` and collect input
- You must call `send_responses_streaming()` with a dict mapping `request_id` to user's answer

#### 2. **Streaming Execution Pattern**
```python
# First iteration
stream = workflow.run_stream(initial_request)

# Subsequent iterations (after human input)
stream = workflow.send_responses_streaming(pending_responses)

# Always process events
events = [event async for event in stream]
```

#### 3. **Event-Driven Architecture**
Listen for specific events to control workflow:
- `RequestInfoEvent` - Human input is needed (workflow paused)
- `WorkflowOutputEvent` - Final result is available (workflow complete)
- `WorkflowStatusEvent` - State changes (IN_PROGRESS, IDLE_WITH_PENDING_REQUESTS, etc.)

#### 4. **Custom Executors with @handler**
The `DecisionManager` demonstrates how to create executors that:
- Use `@handler` decorator to expose methods as workflow steps
- Receive typed messages (e.g., `RequestResponse[HumanFeedbackRequest, str]`)
- Route workflow by sending messages to other executors
- Access context via `WorkflowContext`

#### 5. **Conditional Routing with Human Decisions**
You can create condition functions that evaluate human responses:
```python
def user_wants_alternatives_condition(message: Any) -> bool:
    response_text = message.agent_run_response.text.lower()
    return response_text == "yes"
```

### 🎯 Real-World Applications:

1. **Approval Workflows**
   - Get manager approval before processing expense reports
   - Require human review before sending automated emails
   - Confirm high-value transactions before execution

2. **Content Moderation**
   - Flag questionable content for human review
   - Ask moderators to make final decision on edge cases
   - Escalate to humans when AI confidence is low

3. **Customer Service**
   - Let AI handle routine questions automatically
   - Escalate complex issues to human agents
   - Ask customer if they want to speak to a human

4. **Data Processing**
   - Ask humans to resolve ambiguous data entries
   - Confirm AI interpretations of unclear documents
   - Let users choose between multiple valid interpretations

5. **Safety-Critical Systems**
   - Require human confirmation before irreversible actions
   - Get approval before accessing sensitive data
   - Confirm decisions in regulated industries (healthcare, finance)

6. **Interactive Agents**
   - Build conversational bots that ask follow-up questions
   - Create wizards that guide users through complex processes
   - Design agents that collaborate with humans step-by-step

### 🔄 Comparison: With vs Without Human-in-the-Loop

| Feature | Conditional Workflow | Human-in-the-Loop Workflow |
|---------|---------------------|---------------------------|
| **Execution** | Single `workflow.run()` | Loop with `run_stream()` + `send_responses_streaming()` |
| **User Input** | None (fully automated) | Interactive prompts via `input()` or UI |
| **Components** | Agents + Executors | + RequestInfoExecutor + DecisionManager |
| **Events** | AgentExecutorResponse only | RequestInfoEvent, WorkflowOutputEvent, etc. |
| **Pausing** | No pausing | Workflow pauses at RequestInfoExecutor |
| **Human Control** | No human control | Humans make key decisions |
| **Use Case** | Automation | Collaboration & oversight |

### 🚀 Advanced Patterns:

#### Multiple Human Decision Points
You can have multiple `RequestInfoExecutor` nodes in the same workflow:
```python
.add_edge(agent1, request_info_1)  # First human decision
.add_edge(decision_manager_1, agent2)
.add_edge(agent2, request_info_2)  # Second human decision
.add_edge(decision_manager_2, final_agent)
```

#### Timeout Handling
Implement timeouts for human responses:
```python
import asyncio

try:
    answer = await asyncio.wait_for(
        asyncio.to_thread(input, "Enter yes/no: "),
        timeout=60.0
    )
except asyncio.TimeoutError:
    answer = "no"  # Default to safe option
```

#### Rich UI Integration
Instead of `input()`, integrate with web UI, Slack, Teams, etc.:
```python
if isinstance(event, RequestInfoEvent):
    # Send notification to user's preferred channel
    await slack_client.send_message(
        user_id=current_user,
        text=event.data.prompt,
        request_id=event.request_id
    )
```

#### Conditional Human-in-the-Loop
Only ask for human input in specific situations:
```python
def needs_human_approval_condition(message: Any) -> bool:
    # Only route to human if confidence is low or value is high
    if result.confidence < 0.7 or result.value > 10000:
        return True
    return False
```

### ⚠️ Best Practices:

1. **Always Subclass RequestInfoMessage**
   - Provides type safety and validation
   - Enables rich context for UI rendering
   - Clarifies intent of each request type

2. **Use Descriptive Prompts**
   - Include context about what you're asking
   - Explain consequences of each choice
   - Keep questions simple and clear

3. **Handle Unexpected Input**
   - Validate user responses
   - Provide defaults for invalid input
   - Give clear error messages

4. **Track Request IDs**
   - Use the correlation between request_id and responses
   - Don't try to manage state manually

5. **Design for Non-Blocking**
   - Don't block threads waiting for input
   - Use async patterns throughout
   - Support concurrent workflow instances

### 📚 Related Concepts:

- **Agent Middleware** - Intercept agent calls (previous notebook)
- **Workflow State Management** - Persist workflow state between runs
- **Multi-Agent Collaboration** - Combine human-in-the-loop with agent teams
- **Event-Driven Architectures** - Build reactive systems with events

---

### 🎓 Congratulations!

You've mastered human-in-the-loop workflows with Microsoft Agent Framework! You now know how to:
- ✅ Pause workflows to gather human input
- ✅ Use RequestInfoExecutor and RequestInfoMessage
- ✅ Handle streaming execution with events
- ✅ Create custom executors with @handler
- ✅ Route workflows based on human decisions
- ✅ Build interactive AI agents that collaborate with humans

**This is a critical pattern for building trustworthy, controllable AI systems!** 🚀