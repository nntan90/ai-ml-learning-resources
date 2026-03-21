# Notebook: 14-handoff

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/14-microsoft-agent-framework/code-samples/14-handoff.ipynb

---

# Travel Customer Support with Handoff Orchestration

This notebook demonstrates **handoff orchestration** using the Microsoft Agent Framework. We'll build a travel customer support system where agents can transfer control to specialists based on the customer's needs.

## What You'll Learn:
1. **Handoff Orchestration**: Dynamic agent routing based on context and expertise
2. **HandoffBuilder**: High-level API for building handoff workflows
3. **Specialist Routing**: Agents can hand off to other agents dynamically
4. **Multi-turn Conversations**: Seamless context preservation across handoffs
5. **Customer Support Flow**: Real-world application of agent handoffs

## Prerequisites:
- Microsoft Agent Framework installed
- GitHub token or OpenAI API key configured
- Understanding of basic agent concepts

```python
import asyncio
import json
import os
from collections.abc import AsyncIterable
from typing import Any, cast

from agent_framework import (
    ChatMessage,
    HandoffBuilder,
    HandoffUserInputRequest,
    RequestInfoEvent,
    WorkflowEvent,
    WorkflowOutputEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
)

# GitHub Models or OpenAI client integration
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from IPython.display import HTML, display
from pydantic import BaseModel
```

## Step 1: Define Pydantic Models for Structured Outputs

These models define the schema that each specialized agent will return. This ensures consistent and parseable responses from all agents.

```python
class FlightBookingResult(BaseModel):
    """Flight booking confirmation from the booking agent."""

    destination: str
    departure_date: str
    return_date: str
    booking_reference: str
    passenger_name: str
    flight_details: str
    total_cost: str
    status: str


class DisputeResult(BaseModel):
    """Dispute resolution result from the disputes agent."""

    dispute_type: str
    original_booking: str
    refund_amount: str
    refund_method: str
    processing_time: str
    reference_number: str
    status: str


class TripCheckResult(BaseModel):
    """Trip confirmation result from the trip check agent."""

    trip_reference: str
    destination: str
    travel_dates: str
    confirmation_status: str
    special_notes: str
    contact_info: str
```

## Step 2: Load Environment Variables


```python
# Load environment variables
load_dotenv()

# Check for GitHub Models or OpenAI
chat_client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),
    api_key=os.environ.get("GITHUB_TOKEN"),
    model_id="gpt-4o"
)
```

## Step 3: Create Four Specialized Travel Support Agents

Each agent has specific expertise and can hand off to appropriate specialists based on customer needs.

```python
# Agent 1: Customer Support Agent (Main triage agent)
customer_support_agent = chat_client.create_agent(
    instructions=(
        "You are a friendly customer support agent for a travel company. "
        "Assess customer requests and route them to the appropriate specialist: "
        "- For flight bookings or reservations: call handoff_to_booking_agent "
        "- For refunds, disputes, or billing issues: call handoff_to_disputes_agent "
        "- For trip confirmations or travel plan checks: call handoff_to_trip_check_agent "
        "Be welcoming and ensure customers feel heard before routing them."
    ),
    name="customer_support_agent",
)


# Agent 2: Booking Agent (Flight booking specialist)
booking_agent = chat_client.create_agent(
    instructions=(
        "You are a flight booking specialist. Handle all flight reservations and bookings. "
        "When a customer wants to book a flight, collect their destination, travel dates, "
        "and confirm the booking. Always provide a booking reference number. "
        "Return structured JSON with booking details. "
        "The flight is always confirmed as booked regardless of destination."
    ),
    name="booking_agent",
    response_format=FlightBookingResult,
)

# Agent 3: Disputes Agent (Refund and billing specialist)
disputes_agent = chat_client.create_agent(
    instructions=(
        "You are a disputes and refunds specialist. Handle customer complaints, "
        "refund requests, and billing disputes. Always approve refunds and provide "
        "a reference number. Process refunds back to the original payment method. "
        "Return structured JSON with refund details. "
        "All refund requests are approved and processed immediately."
    ),
    name="disputes_agent",
    response_format=DisputeResult,
)
# Agent 4: Trip Check Agent (Travel confirmation specialist)
trip_check_agent = chat_client.create_agent(
    instructions=(
        "You are a travel confirmation specialist. Verify and confirm customer "
        "travel plans, check itineraries, and provide travel status updates. "
        "Always confirm that travel plans are in order and provide reassurance. "
        "Return structured JSON with confirmation details. "
        "All travel plans are confirmed as valid and ready."
    ),
    name="trip_check_agent",
    response_format=TripCheckResult,
)



```

## Step 4: Build the Handoff Workflow

The HandoffBuilder creates a workflow where the customer support agent can dynamically hand off to specialists based on customer needs.


```python
workflow = (
    HandoffBuilder(
        name="travel_support_handoff",
        participants=[customer_support_agent, booking_agent, disputes_agent, trip_check_agent],
    )
    .set_coordinator(customer_support_agent)  # Main agent that receives initial requests
    .add_handoff(customer_support_agent, [booking_agent, disputes_agent, trip_check_agent])
    .with_termination_condition(
        lambda conv: sum(1 for msg in conv if msg.role.value == "user") > 
    )  # Stop after 3 user messages
    .build()
)

display(HTML("""
<div style='padding: 20px; background: linear-gradient(135deg, #ff7043 0%, #ff5722 100%); color: white; border-radius: 8px; margin: 10px 0;'>
    <h3 style='margin: 0 0 15px 0;'>Handoff Workflow Built Successfully!</h3>
    <p style='margin: 0; line-height: 1.6;'>
        <strong>Handoff Flow:</strong><br>
        • User Request → <strong>Customer Support Agent</strong> (triage)<br>
        • Support Agent → <strong>Specialist Agent</strong> (dynamic handoff)<br>
        • Specialist → <strong>Resolution</strong> (expert handling)<br>
        • System → <strong>User Response</strong> (final result)
    </p>
</div>
"""))


```

## Step 5: Helper Functions for Event Processing

These functions help us process workflow events and handle user input requests during the handoff process.

```python
async def drain_events(stream: AsyncIterable[WorkflowEvent]) -> list[WorkflowEvent]:
    """Collect all events from an async stream into a list."""
    return [event async for event in stream]


def handle_workflow_events(events: list[WorkflowEvent]) -> list[RequestInfoEvent]:
    """Process workflow events and extract pending user input requests."""
    requests: list[RequestInfoEvent] = []
    
    for event in events:
        if isinstance(event, WorkflowStatusEvent) and event.state in {
            WorkflowRunState.IDLE,
            WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
        }:
           print(f"[Workflow Status] {event.state.name}")

        elif isinstance(event, WorkflowOutputEvent):
            conversation = cast(list[ChatMessage], event.data)
            if isinstance(conversation, list):
                print("\n=== Final Conversation ===")
                for message in conversation:
                    # Filter out messages with no text (tool calls)
                    if not message.text.strip():
                        continue
                    speaker = message.author_name or message.role.value
                    print(f"- {speaker}: {message.text}")
                print("==========================")

        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                print_handoff_request(event.data)
                requests.append(event)

    return requests


def print_handoff_request(request: HandoffUserInputRequest) -> None:
    """Display a user input request with conversation context."""
    print("\n=== User Input Requested ===")
    # Filter out messages with no text for cleaner display
    messages_with_text = [
        msg for msg in request.conversation if msg.text.strip()]
    print(f"Last {len(messages_with_text)} messages in conversation:")
    for message in messages_with_text[-3:]:  # Show last 3 for brevity
        speaker = message.author_name or message.role.value
        text = message.text[:100] + \
            "..." if len(message.text) > 100 else message.text
        print(f"  {speaker}: {text}")
    print("============================")


print("Helper functions defined for event processing")
```

## Step 6: Test Case 1 - Flight Booking Request

Let's test our handoff workflow with a flight booking request. The customer support agent should hand off to the booking agent.


```python
async def test_booking_handoff():
    """Test handoff workflow for flight booking requests."""

    display(HTML("""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>Test Case 1: Flight Booking Request</h3>
        <p style='margin: 0;'><strong>Expected Flow:</strong> Customer Support → Booking Agent</p>
    </div>
    """))

    # Start the workflow
    print("[User]: I want to book a flight to Paris for next month")
    events = await drain_events(
        workflow.run_stream("I want to book a flight to Paris for next month")
    )
    pending_requests = handle_workflow_events(events)

    # Handle any additional user input requests
    scripted_responses = [
        "I'd like to travel from New York to Paris on December 15th and return on December 22nd.",
        "Yes, please confirm the booking under the name John Smith."
    ]

    response_index = 0
    while pending_requests and response_index < len(scripted_responses):
        user_response = scripted_responses[response_index]
        print(f"\n[User]: {user_response}")

        responses = {req.request_id: user_response for req in pending_requests}
        events = await drain_events(workflow.send_responses_streaming(responses))
        pending_requests = handle_workflow_events(events)

        response_index += 1

    # Extract and display the final booking result
    if events:
        for event in events:
            if isinstance(event, WorkflowOutputEvent):
                conversation = cast(list[ChatMessage], event.data)
                for message in conversation:
                    if message.author_name == "booking_agent" and message.text.strip():
                        try:
                            booking_data = FlightBookingResult.model_validate_json(
                                message.text)
                            display_booking_result(booking_data)
                        except Exception as e:
                            print(f"Could not parse booking result: {e}")


def display_booking_result(booking: FlightBookingResult):
    """Display flight booking result in a formatted section."""

    display(HTML(f"""
    <div style='padding: 20px; background: #e8f5e9; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4caf50;'>
        <h3 style='margin: 0 0 15px 0; color: #2e7d32;'>✈️ Flight Booking Confirmed</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;'>
            <div>
                <strong style='color: #333;'>Booking Reference:</strong> {booking.booking_reference}<br>
                <strong style='color: #333;'>Passenger:</strong> {booking.passenger_name}<br>
                <strong style='color: #333;'>Status:</strong> <span style='color: #4caf50; font-weight: bold;'>{booking.status}</span>
            </div>
            <div>
                <strong style='color: #333;'>Destination:</strong> {booking.destination}<br>
                <strong style='color: #333;'>Total Cost:</strong> {booking.total_cost}<br>
                <strong style='color: #333;'>Departure:</strong> {booking.departure_date}
            </div>
        </div>
        <div style='margin-bottom: 10px;'>
            <strong style='color: #333;'>Flight Details:</strong> {booking.flight_details}
        </div>
        <div style='background: rgba(76,175,80,0.1); padding: 10px; border-radius: 4px; margin-top: 10px;'>
            <strong style='color: #2e7d32;'>✅ Success:</strong> Flight booking completed through handoff to booking specialist
        </div>
    </div>
    """))


# Run the booking test
await test_booking_handoff()
```

## Step 7: Test Case 2 - Dispute/Refund Request

Let's test our handoff workflow with a refund request. The customer support agent should hand off to the disputes agent.

```python
async def test_dispute_handoff():
    """Test handoff workflow for dispute/refund requests."""

    display(HTML("""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>Test Case 2: Refund Request</h3>
        <p style='margin: 0;'><strong>Expected Flow:</strong> Customer Support → Disputes Agent</p>
    </div>
    """))

    # Start the workflow
    print("[User]: I need to cancel my flight and get a refund")
    events = await drain_events(
        workflow.run_stream("I need to cancel my flight and get a refund")
    )
    pending_requests = handle_workflow_events(events)

    # Handle any additional user input requests
    scripted_responses = [
        "My booking reference is FL12345. I can't travel due to a family emergency.",
        "Yes, please process the full refund back to my credit card."
    ]

    response_index = 0
    while pending_requests and response_index < len(scripted_responses):
        user_response = scripted_responses[response_index]
        print(f"\n[User]: {user_response}")

        responses = {req.request_id: user_response for req in pending_requests}
        events = await drain_events(workflow.send_responses_streaming(responses))
        pending_requests = handle_workflow_events(events)

        response_index += 1

    # Extract and display the final dispute result
    if events:
        for event in events:
            if isinstance(event, WorkflowOutputEvent):
                conversation = cast(list[ChatMessage], event.data)
                for message in conversation:
                    if message.author_name == "disputes_agent" and message.text.strip():
                        try:
                            dispute_data = DisputeResult.model_validate_json(
                                message.text)
                            display_dispute_result(dispute_data)
                        except Exception as e:
                            print(f"Could not parse dispute result: {e}")


def display_dispute_result(dispute: DisputeResult):
    """Display dispute resolution result in a formatted section."""

    display(HTML(f"""
    <div style='padding: 20px; background: #fff3e0; border-radius: 8px; margin: 15px 0; border-left: 4px solid #ff9800;'>
        <h3 style='margin: 0 0 15px 0; color: #f57c00;'>💰 Refund Processed</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;'>
            <div>
                <strong style='color: #333;'>Reference Number:</strong> {dispute.reference_number}<br>
                <strong style='color: #333;'>Dispute Type:</strong> {dispute.dispute_type}<br>
                <strong style='color: #333;'>Status:</strong> <span style='color: #ff9800; font-weight: bold;'>{dispute.status}</span>
            </div>
            <div>
                <strong style='color: #333;'>Refund Amount:</strong> {dispute.refund_amount}<br>
                <strong style='color: #333;'>Refund Method:</strong> {dispute.refund_method}<br>
                <strong style='color: #333;'>Processing Time:</strong> {dispute.processing_time}
            </div>
        </div>
        <div style='margin-bottom: 10px;'>
            <strong style='color: #333;'>Original Booking:</strong> {dispute.original_booking}
        </div>
        <div style='background: rgba(255,152,0,0.1); padding: 10px; border-radius: 4px; margin-top: 10px;'>
            <strong style='color: #f57c00;'>✅ Success:</strong> Refund processed through handoff to disputes specialist
        </div>
    </div>
    """))

    # Run the dispute test
await test_dispute_handoff()
```

## Step 8: Test Case 3 - Trip Confirmation Request

Let's test our handoff workflow with a trip confirmation request. The customer support agent should hand off to the trip check agent.

```python
async def test_trip_check_handoff():
    """Test handoff workflow for trip confirmation requests."""

    display(HTML("""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>Test Case 3: Trip Confirmation</h3>
        <p style='margin: 0;'><strong>Expected Flow:</strong> Customer Support → Trip Check Agent</p>
    </div>
    """))

    # Start the workflow
    print("[User]: Can you confirm my travel plans are all set?")
    events = await drain_events(
        workflow.run_stream("Can you confirm my travel plans are all set?")
    )
    pending_requests = handle_workflow_events(events)

    # Handle any additional user input requests
    scripted_responses = [
        "I'm traveling to London next week. My confirmation number is TR98765.",
        "Perfect, thank you for checking everything is ready!"
    ]

    response_index = 0
    while pending_requests and response_index < len(scripted_responses):
        user_response = scripted_responses[response_index]
        print(f"\n[User]: {user_response}")
        
        responses = {req.request_id: user_response for req in pending_requests}
        events = await drain_events(workflow.send_responses_streaming(responses))
        pending_requests = handle_workflow_events(events)
        
        response_index += 1
    # Extract and display the final trip check result
    if events:
        for event in events:
            if isinstance(event, WorkflowOutputEvent):
                conversation = cast(list[ChatMessage], event.data)
                for message in conversation:
                    if message.author_name == "trip_check_agent" and message.text.strip():
                        try:
                            trip_data = TripCheckResult.model_validate_json(
                                message.text)
                            display_trip_check_result(trip_data)
                        except Exception as e:
                            print(f"Could not parse trip check result: {e}")


def display_trip_check_result(trip: TripCheckResult):
    """Display trip confirmation result in a formatted section."""

    display(HTML(f"""
    <div style='padding: 20px; background: #f3e5f5; border-radius: 8px; margin: 15px 0; border-left: 4px solid #9c27b0;'>
        <h3 style='margin: 0 0 15px 0; color: #7b1fa2;'>🎯 Trip Confirmed</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;'>
            <div>
                <strong style='color: #333;'>Trip Reference:</strong> {trip.trip_reference}<br>
                <strong style='color: #333;'>Destination:</strong> {trip.destination}<br>
                <strong style='color: #333;'>Status:</strong> <span style='color: #9c27b0; font-weight: bold;'>{trip.confirmation_status}</span>
            </div>
            <div>
                <strong style='color: #333;'>Travel Dates:</strong> {trip.travel_dates}<br>
                <strong style='color: #333;'>Contact Info:</strong> {trip.contact_info}
            </div>
        </div>
        <div style='margin-bottom: 10px;'>
            <strong style='color: #333;'>Special Notes:</strong> {trip.special_notes}
        </div>
        <div style='background: rgba(156,39,176,0.1); padding: 10px; border-radius: 4px; margin-top: 10px;'>
            <strong style='color: #7b1fa2;'>✅ Success:</strong> Trip confirmed through handoff to trip check specialist
        </div>
    </div>
    """))


# Run the trip check test
await test_trip_check_handoff()

   
```

## Step 9: Workflow Analysis - Understanding Handoff Flow


```python
async def analyze_handoff_patterns():
    """Analyze different handoff patterns and routing decisions."""

    display(HTML("""
    <div style='padding: 20px; background: #f3e5f5; border-left: 4px solid #9c27b0; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #7b1fa2;'>Handoff Pattern Analysis</h3>
        <p style='margin: 0;'>Testing different request types to show routing decisions...</p>
    </div>
    """))

    test_requests = [
        "I want to book a round-trip flight to Tokyo",
        "I need a refund for my cancelled flight",
        "Please check if my travel itinerary is confirmed",
        "Can you help me with a billing dispute?"
    ]

    for i, request in enumerate(test_requests, 1):
        print(f"\n--- Test Request {i} ---")
        print(f"User: {request}")

        # Run workflow and capture routing decision
        events = await drain_events(workflow.run_stream(request))

        # Analyze which agent was activated
        for event in events:
            if isinstance(event, WorkflowOutputEvent):
                conversation = cast(list[ChatMessage], event.data)
                for message in conversation:
                    if message.author_name == "customer_support_agent":
                        print(f"Support Agent: {message.text[:100]}...")
                    elif message.author_name in ["booking_agent", "disputes_agent", "trip_check_agent"]:
                        agent_type = {
                            "booking_agent": "🛫 BOOKING SPECIALIST",
                            "disputes_agent": "💰 DISPUTES SPECIALIST",
                            "trip_check_agent": "🎯 TRIP CHECK SPECIALIST"
                        }[message.author_name]
                        print(f"Routed to: {agent_type}")
                        break
                break
    display(HTML("""
    <div style='padding: 25px; background: linear-gradient(135deg, #9c27b0 0%, #673ab7 100%); color: white; border-radius: 12px; 
                box-shadow: 0 4px 12px rgba(156,39,176,0.4); margin: 20px 0;'>
        <h2 style='margin: 0 0 20px 0;'>Handoff Analysis Results</h2>
        <div style='background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px;'>
            <h4 style='margin: 0 0 10px 0;'>Key Observations</h4>
            <ul style='margin: 0; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Dynamic Routing:</strong> Customer support agent analyzes request intent</li>
                <li><strong>Context Preservation:</strong> Full conversation history maintained</li>
                <li><strong>Specialist Focus:</strong> Each agent handles their expertise area</li>
                <li><strong>Seamless Handoff:</strong> Users don't need to repeat information</li>
            </ul>
        </div>
    </div>
    """))

    # Run the analysis
await analyze_handoff_patterns()
```