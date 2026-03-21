# Notebook: 14-concurrent

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/14-microsoft-agent-framework/code-samples/14-concurrent.ipynb

---

# Travel Recommendations with Concurrent Orchestration

This notebook demonstrates **concurrent orchestration** using the Microsoft Agent Framework. We'll build a travel recommendation system with three specialized agents that work in parallel to provide comprehensive travel insights.

## What You'll Learn:
1. **Concurrent Orchestration**: Running multiple agents in parallel (fan-out/fan-in pattern)
2. **ConcurrentBuilder**: High-level API for building concurrent workflows
3. **Travel Recommendations**: Three specialized agents working together
4. **Default Aggregation**: Combining multiple agent responses
5. **Performance Benefits**: Parallel execution vs sequential processing


## Three Specialized Agents:

1. **Attractions Agent**: Tourist attractions, activities, landmarks
2. **Dining Agent**: Local cuisine, restaurants, food experiences
3. **History Agent**: Historical facts, cultural significance, context

```python
import asyncio
import json
import os
from typing import Any, cast

from agent_framework import (
    ChatMessage,
    ConcurrentBuilder,
    WorkflowOutputEvent,
)

# GitHub Models or OpenAI client integration
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from IPython.display import HTML, display
from pydantic import BaseModel

print("All imports successful!")
```

## Step 1: Define Pydantic Models for Structured Outputs

These models define the schema that each specialized agent will return. This ensures consistent and parseable responses from all agents.

## Step 1: Define Pydantic Models for Structured Outputs

These models define the schema that each specialized agent will return. This ensures consistent and parseable responses from all agents.

```python
class AttractionsRecommendation(BaseModel):
    """Tourist attractions and activities recommendations."""

    destination: str
    top_attractions: list[str]
    activities: list[str]
    best_time_to_visit: str
    transportation_tips: str  


class DiningRecommendation(BaseModel):
    """Food and dining recommendations."""

    destination: str
    local_cuisine: str
    must_try_dishes: list[str]
    recommended_restaurants: list[str]
    food_experiences: list[str]
    dining_etiquette: str


class HistoryRecommendation(BaseModel):
    """Historical and cultural information."""

    destination: str
    historical_significance: str
    cultural_highlights: list[str]
    important_periods: list[str]
    cultural_experiences: list[str]
    interesting_facts: list[str]
```

## Step 2: Load Environment Variables

Configure the LLM client (GitHub Models or OpenAI) following the same pattern as the middleware notebook.

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

## Step 3: Create Three Specialized Travel Agents


```python
# Agent 1: Tourist Attractions Expert
attractions_agent = chat_client.create_agent(
    instructions=(
        "You are a tourism expert specializing in attractions and activities. "
        "When given a travel destination, provide comprehensive recommendations for "
        "tourist attractions, activities, best times to visit, and transportation tips. "
        "Focus on popular landmarks, unique experiences, and practical travel advice. "
        "Return structured JSON with the specified fields."
    ),
    name="attractions_agent",
    response_format=AttractionsRecommendation,
)

# Agent 2: Food and Dining Expert
dining_agent = chat_client.create_agent(
    instructions=(
        "You are a culinary expert specializing in local food and dining experiences. "
        "When given a travel destination, provide recommendations for local cuisine, "
        "must-try dishes, recommended restaurants, and unique food experiences. "
        "Include dining etiquette and cultural food customs. "
        "Return structured JSON with the specified fields."
    ),
    name="dining_agent",
    response_format=DiningRecommendation,
)


# Agent 3: History and Culture Expert
history_agent = chat_client.create_agent(
    instructions=(
        "You are a historian and cultural expert. "
        "When given a travel destination, provide historical context, cultural significance, "
        "important historical periods, cultural experiences, and interesting facts. "
        "Focus on helping travelers understand the cultural heritage and historical importance. "
        "Return structured JSON with the specified fields."
    ),
    name="history_agent",
    response_format=HistoryRecommendation,
)
```

# Step 4: Build the Concurrent Workflow

The ConcurrentBuilder creates a workflow that:
1. **Dispatches** the same input to all three agents simultaneously (fan-out)
2. **Runs agents** in parallel for better performance
3. **Aggregates** all responses into a single output (fan-in)
4. **Returns** combined ChatMessage list from all agents

```python
# Build the concurrent workflow using ConcurrentBuilder
workflow = (
    ConcurrentBuilder()
    .participants([attractions_agent, dining_agent, history_agent])
    .build()
)

display(HTML("""
<div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; margin: 10px 0;'>
    <h3 style='margin: 0 0 15px 0;'>Concurrent Workflow Built Successfully!</h3>
    <p style='margin: 0; line-height: 1.6;'>
        <strong>Architecture:</strong><br>
        • Input → <strong>Dispatcher</strong> (fan-out)<br>
        • <strong>3 Agents</strong> run in parallel (attractions, dining, history)<br>
        • <strong>Aggregator</strong> combines results (fan-in)<br>
        • Output → Combined travel recommendations
    </p>
</div>
"""))
```

## Step 5: Test Case 1 - Tokyo Travel Recommendations

Let's test our concurrent workflow with Tokyo as the destination. All three agents will work simultaneously to provide comprehensive travel recommendations.

```python
async def display_travel_recommendations(destination: str):
    """Run the concurrent workflow and display formatted results."""

    display(HTML(f"""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>Processing Travel Recommendations for {destination}</h3>
        <p style='margin: 0;'><strong>Status:</strong> Running 3 agents concurrently...</p>
    </div>
    """))

    # Run the workflow
    events = await workflow.run(f"I want comprehensive travel recommendations for {destination}")
    outputs = events.get_outputs()

    if outputs:
        # Get the aggregated messages from all agents
        messages: list[ChatMessage] = outputs[0]
        # Separate messages by agent (skip user message)
        agent_responses = [msg for msg in messages if msg.author_name in [
            "attractions_agent", "dining_agent", "history_agent"]]

        # Display results
        display(HTML(f"""
        <div style='padding: 25px; background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%); color: white; border-radius: 12px; 
                    box-shadow: 0 4px 12px rgba(76,175,80,0.3); margin: 20px 0;'>
            <h2 style='margin: 0 0 20px 0;'>Complete Travel Guide for {destination}</h2>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>Generated by 3 concurrent agents</p>
        </div>
        """))
        # Process each agent's response
        for msg in agent_responses:
            agent_name = msg.author_name

            try:
                # Parse the JSON response
                if agent_name == "attractions_agent":
                    data = AttractionsRecommendation.model_validate_json(
                        msg.text)
                    display_attractions_section(data)
                elif agent_name == "dining_agent":
                    data = DiningRecommendation.model_validate_json(msg.text)
                    display_dining_section(data)
                elif agent_name == "history_agent":
                    data = HistoryRecommendation.model_validate_json(msg.text)
                    display_history_section(data)
            except Exception as e:
                display(HTML(f"""
                <div style='padding: 15px; background: #ffcdd2; border-left: 4px solid #f44336; border-radius: 4px; margin: 10px 0;'>
                    <strong>Error parsing {agent_name} response:</strong> {str(e)}
                    <details><summary>Raw response</summary>{msg.text}</details>
                </div>
                """))
def display_attractions_section(data: AttractionsRecommendation):
    """Display attractions recommendations in a formatted section."""
    attractions_list = "".join([f"<li>{attraction}</li>" for attraction in data.top_attractions])
    activities_list = "".join([f"<li>{activity}</li>" for activity in data.activities])
    
    display(HTML(f"""
    <div style='padding: 20px; background: #e3f2fd; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196f3;'>
        <h3 style='margin: 0 0 15px 0; color: #1976d2;'>🏛️ Tourist Attractions & Activities</h3>
        <div style='margin-bottom: 15px;'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Top Attractions:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{attractions_list}</ul>
        </div>
        <div style='margin-bottom: 15px;'>
        <h4 style='margin: 0 0 8px 0; color: #333;'>Recommended Activities:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{activities_list}</ul>
        </div>
        <div style='margin-bottom: 10px;'>
            <strong style='color: #333;'>Best Time to Visit:</strong> {data.best_time_to_visit}
        </div>
        <div>
            <strong style='color: #333;'>Transportation Tips:</strong> {data.transportation_tips}
        </div>
    </div>
    """))


def display_dining_section(data: DiningRecommendation):
    """Display dining recommendations in a formatted section."""
    dishes_list = "".join(
        [f"<li>{dish}</li>" for dish in data.must_try_dishes])
    restaurants_list = "".join(
        [f"<li>{restaurant}</li>" for restaurant in data.recommended_restaurants])
    experiences_list = "".join(
        [f"<li>{exp}</li>" for exp in data.food_experiences])

    display(HTML(f"""
    <div style='padding: 20px; background: #fff3e0; border-radius: 8px; margin: 15px 0; border-left: 4px solid #ff9800;'>
        <h3 style='margin: 0 0 15px 0; color: #f57c00;'>🍜 Food & Dining Experiences</h3>
        <div style='margin-bottom: 15px;'>
            <strong style='color: #333;'>Local Cuisine:</strong> {data.local_cuisine}
        </div>
        <div style='margin-bottom: 15px;'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Must-Try Dishes:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{dishes_list}</ul>
        </div>
        <div style='margin-bottom: 15px;'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Recommended Restaurants:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{restaurants_list}</ul>
        </div>
        <div style='margin-bottom: 15px;'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Food Experiences:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{experiences_list}</ul>
        </div>
        <div>
            <strong style='color: #333;'>Dining Etiquette:</strong> {data.dining_etiquette}
        </div>
    </div>
    """))


def display_history_section(data: HistoryRecommendation):
    """Display history recommendations in a formatted section."""
    highlights_list = "".join(
        [f"<li>{highlight}</li>" for highlight in data.cultural_highlights])
    periods_list = "".join(
        [f"<li>{period}</li>" for period in data.important_periods])
    experiences_list = "".join(
        [f"<li>{exp}</li>" for exp in data.cultural_experiences])
    facts_list = "".join(
        [f"<li>{fact}</li>" for fact in data.interesting_facts])

    display(HTML(f"""
    <div style='padding: 20px; background: #f3e5f5; border-radius: 8px; margin: 15px 0; border-left: 4px solid #9c27b0;'>
        <h3 style='margin: 0 0 15px 0; color: #7b1fa2;'>📚 History & Culture</h3>
        <div style='margin-bottom: 15px;'>
            <strong style='color: #333;'>Historical Significance:</strong> {data.historical_significance}
        </div>
        <div style='margin-bottom: 15px;'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Cultural Highlights:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{highlights_list}</ul>
        </div>
        <div style='margin-bottom: 15px;'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Important Historical Periods:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{periods_list}</ul>
        </div>
        <div style='margin-bottom: 15px;'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Cultural Experiences:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{experiences_list}</ul>
        </div>
        <div>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Interesting Facts:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{facts_list}</ul>
        </div>
    </div>
    """))

    # Test with Tokyo
await display_travel_recommendations("Tokyo")
```

# Step 6: Test Case 2 - Paris Travel Recommendations

```python
await display_travel_recommendations("Paris")
```

## Step 7: Performance Analysis - Concurrent vs Sequential

Let's measure the performance difference between concurrent and sequential execution to demonstrate the benefits of concurrent orchestration.



```python
import time
from agent_framework import SequentialBuilder


async def measure_concurrent_performance(destination: str):
    """Measure concurrent execution time."""
    start_time = time.time()

    events = await workflow.run(f"I want travel recommendations for {destination}")
    outputs = events.get_outputs()

    end_time = time.time()
    return end_time - start_time, len(outputs[0]) if outputs else 0


async def measure_sequential_performance(destination: str):
    """Measure sequential execution time."""
    # Build sequential workflow for comparison
    sequential_workflow = (
        SequentialBuilder()
        .participants([attractions_agent, dining_agent, history_agent])
        .build()
    )
    start_time = time.time()

    events = await sequential_workflow.run(f"I want travel recommendations for {destination}")
    outputs = events.get_outputs()

    end_time = time.time()
    return end_time - start_time, len(outputs[0]) if outputs else 0


async def performance_comparison():
    """Compare concurrent vs sequential performance."""
    test_destination = "Barcelona"

    display(HTML("""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>Performance Comparison Test</h3>
        <p style='margin: 0;'>Testing with destination: <strong>Barcelona</strong></p>
    </div>
    """))

    # Test concurrent execution
    print("Running concurrent workflow...")
    concurrent_time, concurrent_msgs = await measure_concurrent_performance(test_destination)

# Test sequential execution
    print("Running sequential workflow...")
    sequential_time, sequential_msgs = await measure_sequential_performance(test_destination)

    # Calculate performance improvement
    improvement = ((sequential_time - concurrent_time) / sequential_time) * 100

    display(HTML(f"""
    <div style='padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; 
                box-shadow: 0 4px 12px rgba(102,126,234,0.4); margin: 20px 0;'>
        <h2 style='margin: 0 0 20px 0;'>Performance Results</h2>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;'>
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                <h4 style='margin: 0 0 10px 0;'>⚡ Concurrent Execution</h4>
                <p style='margin: 0; font-size: 24px; font-weight: bold;'>{concurrent_time:.2f}s</p>
                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>{concurrent_msgs} messages</p>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                <h4 style='margin: 0 0 10px 0;'>🔄 Sequential Execution</h4>
                <p style='margin: 0; font-size: 24px; font-weight: bold;'>{sequential_time:.2f}s</p>
                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>{sequential_msgs} messages</p>
            </div>
        </div>
        <div style='background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px;'>
            <h4 style='margin: 0 0 10px 0;'>Performance Improvement</h4>
            <p style='margin: 0; font-size: 20px; font-weight: bold;'>{improvement:.1f}% faster</p>
            <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
                Saved {sequential_time - concurrent_time:.2f} seconds with concurrent execution
            </p>
        </div>
    </div>
    """))
# Run performance comparison
await performance_comparison()
```