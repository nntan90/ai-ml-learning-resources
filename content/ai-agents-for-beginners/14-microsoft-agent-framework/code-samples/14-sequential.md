# Notebook: 14-sequential

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/14-microsoft-agent-framework/code-samples/14-sequential.ipynb

---

## Two Sequential Agents:

1. **Front Desk Agent**: Makes initial attraction recommendations for the city
2. **Concierge Agent**: Reviews and rates the front desk recommendation based on popularity

## Key Benefits of Sequential Orchestration:

- **Iterative Refinement**: Second agent improves upon first agent's work
- **Specialization**: Each agent has a specific role in the process
- **Quality Control**: Built-in review and validation step
- **Clear Information Flow**: Structured handoff between agents

## Prerequisites:
- Microsoft Agent Framework installed
- GitHub token or OpenAI API key configured
- Understanding of basic agent concepts

```python
import asyncio
import json
import os
from typing import Any, cast

from agent_framework import (
    ChatMessage,
    SequentialBuilder,
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

These models define the schema that each agent will return. The front desk agent provides a recommendation, and the concierge agent provides a review and rating.

```python
class AttractionRecommendation(BaseModel):
    """Attraction recommendation from the front desk agent."""

    city: str
    attraction_name: str
    description: str
    category: str  # e.g., "museum", "landmark", "park", "entertainment"
    recommended_duration: str  # e.g., "2-3 hours", "half day"
    why_recommended: str
    best_time_to_visit: str


class AttractionReview(BaseModel):
    """Expert review and rating from the concierge agent."""

    attraction_name: str
    city: str
    popularity_score: int  # 1-10 scale
    popularity_reasoning: str
    visitor_rating: float  # 1.0-5.0 scale
    pros: list[str]
    cons: list[str]
    concierge_recommendation: str
    alternative_suggestions: list[str]
```

## Step 2: Load Environment Variables

Configure the LLM client (GitHub Models or OpenAI) following the same pattern as the concurrent notebook.

```python
# Load environment variables
load_dotenv()

# Check for GitHub Models or OpenAI
chat_client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),
    api_key=os.environ.get("GITHUB_TOKEN"),
    model_id="gpt-4o"
)

print("Chat client configured successfully!")
```

## Step 3: Create Two Sequential Agents

Each agent has a specific role in the sequential workflow. The front desk agent makes recommendations, and the concierge agent reviews and rates them.


```python
# Agent 1: Front Desk Agent (Makes initial recommendations)
front_desk_agent = chat_client.create_agent(
    instructions=(
        "You are a knowledgeable hotel front desk agent who specializes in local attractions. "
        "When a guest asks about attractions in a city, provide a single, well-researched recommendation "
        "for a popular tourist attraction. Focus on giving practical information including what makes "
        "this attraction special, how long to spend there, and the best time to visit. "
        "Be helpful and enthusiastic about your recommendation. "
        "Return structured JSON with the specified fields."
    ),
    name="front_desk_agent",
    response_format=AttractionRecommendation,
)

# Agent 2: Concierge Agent (Reviews and rates recommendations)
concierge_agent = chat_client.create_agent(
    instructions=(
        "You are an expert concierge with extensive knowledge of tourist attractions worldwide. "
        "You will receive an attraction recommendation and must provide an expert review and rating. "
        "Evaluate the recommendation based on the attraction's popularity, visitor satisfaction, "
        "and overall quality. Provide a popularity score (1-10), visitor rating (1.0-5.0), "
        "list pros and cons, and give your professional assessment. "
        "Also suggest alternative attractions if appropriate. "
        "Return structured JSON with the specified fields."
    ),
    name="concierge_agent",
    response_format=AttractionReview,
)


```

## Step 4: Build the Sequential Workflow

The SequentialBuilder creates a workflow where:
1. **Front Desk Agent** receives user input and makes a recommendation
2. **Concierge Agent** receives the front desk recommendation and provides expert review
3. **Output** contains both the original recommendation and the expert review

```python
# Build the sequential workflow using SequentialBuilder
workflow = (
    SequentialBuilder()
    .participants([front_desk_agent, concierge_agent])
    .build()
)

display(HTML("""
<div style='padding: 20px; background: linear-gradient(135deg, #ff7043 0%, #ff5722 100%); color: white; border-radius: 8px; margin: 10px 0;'>
    <h3 style='margin: 0 0 15px 0;'>Sequential Workflow Built Successfully!</h3>
    <p style='margin: 0; line-height: 1.6;'>
        <strong>Flow:</strong><br>
        • User Input → <strong>Front Desk Agent</strong> (recommendation)<br>
        • Front Desk Output → <strong>Concierge Agent</strong> (review & rating)<br>
        • Final Output → Combined recommendation + expert review
    </p>
</div>
"""))
```

```python
async def display_attraction_recommendation(city: str):
    """Run the sequential workflow and display formatted results."""

    display(HTML(f"""
    <div style='padding: 20px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #e65100;'>Processing Attraction Recommendation for {city}</h3>
        <p style='margin: 0;'><strong>Status:</strong> Running sequential workflow...</p>
    </div>
    """))

    # Run the workflow
    events = await workflow.run(f"I want to visit an attraction in {city}")
    outputs = events.get_outputs()

    if outputs:
        # Get the final conversation from both agents
        messages: list[ChatMessage] = outputs[0]

        # Find front desk and concierge responses
        front_desk_response = None
        concierge_response = None

        for msg in messages:
            if msg.author_name == "front_desk_agent":
                front_desk_response = msg.text
            elif msg.author_name == "concierge_agent":
                concierge_response = msg.text
# Display results
        display(HTML(f"""
        <div style='padding: 25px; background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%); color: white; border-radius: 12px; 
                    box-shadow: 0 4px 12px rgba(76,175,80,0.3); margin: 20px 0;'>
            <h2 style='margin: 0 0 20px 0;'>Attraction Recommendation for {city}</h2>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>Generated by sequential agent workflow</p>
        </div>
        """))
        
        # Process and display responses
        if front_desk_response:
            try:
                recommendation_data = AttractionRecommendation.model_validate_json(front_desk_response)
                display_front_desk_section(recommendation_data)
            except Exception as e:
                display(HTML(f"""
                <div style='padding: 15px; background: #ffcdd2; border-left: 4px solid #f44336; border-radius: 4px; margin: 10px 0;'>
                    <strong>Error parsing front desk response:</strong> {str(e)}
                    <details><summary>Raw response</summary>{front_desk_response}</details>
                </div>
                 """))
        
        if concierge_response:
            try:
                review_data = AttractionReview.model_validate_json(concierge_response)
                display_concierge_section(review_data)
            except Exception as e:
                display(HTML(f"""
                <div style='padding: 15px; background: #ffcdd2; border-left: 4px solid #f44336; border-radius: 4px; margin: 10px 0;'>
                    <strong>Error parsing concierge response:</strong> {str(e)}
                    <details><summary>Raw response</summary>{concierge_response}</details>
                </div>
                """))


def display_front_desk_section(data: AttractionRecommendation):
    """Display front desk recommendation in a formatted section."""
    
    display(HTML(f"""
    <div style='padding: 20px; background: #e3f2fd; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196f3;'>
        <h3 style='margin: 0 0 15px 0; color: #1976d2;'>🏨 Front Desk Recommendation</h3>
        <div style='margin-bottom: 15px;'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>{data.attraction_name}</h4>
            <span style='background: #2196f3; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;'>{data.category}</span>
        </div>
        <div style='margin-bottom: 15px;'>
            <strong style='color: #333;'>Description:</strong> {data.description}
        </div>
        <div style='margin-bottom: 10px;'>
            <strong style='color: #333;'>Why Recommended:</strong> {data.why_recommended}
        </div>
        <div style='margin-bottom: 10px;'>
            <strong style='color: #333;'>Recommended Duration:</strong> {data.recommended_duration}
        </div>
        <div>
            <strong style='color: #333;'>Best Time to Visit:</strong> {data.best_time_to_visit}
        </div>
    </div>
    """))
def display_concierge_section(data: AttractionReview):
    """Display concierge review in a formatted section."""
    
    # Create star rating display
    star_rating = "⭐" * int(data.visitor_rating) + "☆" * (5 - int(data.visitor_rating))
    
    # Create popularity bar
    popularity_bar = "🟩" * data.popularity_score + "⬜" * (10 - data.popularity_score)
    
    pros_list = "".join([f"<li style='color: #4caf50;'>✓ {pro}</li>" for pro in data.pros])
    cons_list = "".join([f"<li style='color: #f44336;'>✗ {con}</li>" for con in data.cons])
    alternatives_list = "".join([f"<li>{alt}</li>" for alt in data.alternative_suggestions])
    
    display(HTML(f"""
    <div style='padding: 20px; background: #fff3e0; border-radius: 8px; margin: 15px 0; border-left: 4px solid #ff9800;'>
        <h3 style='margin: 0 0 15px 0; color: #f57c00;'>🎩 Concierge Expert Review</h3>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;'>
            <div style='background: rgba(255,152,0,0.1); padding: 15px; border-radius: 8px;'>
                <h4 style='margin: 0 0 8px 0; color: #333;'>Popularity Score</h4>
                <div style='font-size: 24px; font-weight: bold; color: #f57c00;'>{data.popularity_score}/10</div>
                <div style='font-size: 12px; margin-top: 5px;'>{popularity_bar}</div>
            </div>
            <div style='background: rgba(255,152,0,0.1); padding: 15px; border-radius: 8px;'>
                <h4 style='margin: 0 0 8px 0; color: #333;'>Visitor Rating</h4>
                <div style='font-size: 20px; font-weight: bold; color: #f57c00;'>{data.visitor_rating}/5.0</div>
                <div style='font-size: 16px; margin-top: 5px;'>{star_rating}</div>
            </div>
        </div>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #333;'>Popularity Reasoning:</strong> {data.popularity_reasoning}
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;'>
            <div>
                <h4 style='margin: 0 0 8px 0; color: #333;'>Pros:</h4>
                <ul style='margin: 0; padding-left: 20px;'>{pros_list}</ul>
            </div>
            <div>
                <h4 style='margin: 0 0 8px 0; color: #333;'>Cons:</h4>
                <ul style='margin: 0; padding-left: 20px;'>{cons_list}</ul>
            </div>
        </div>
        <div style='margin-bottom: 15px;'>
            <strong style='color: #333;'>Concierge Recommendation:</strong> {data.concierge_recommendation}
        </div>
        
        <div>
            <h4 style='margin: 0 0 8px 0; color: #333;'>Alternative Suggestions:</h4>
            <ul style='margin: 0; padding-left: 20px; color: #555;'>{alternatives_list}</ul>
        </div>
    </div>
    """))
    # Test with Paris
await display_attraction_recommendation("Stockholm")

```

## Step 8: Workflow Analysis - Understanding Sequential Flow

Let's examine how information flows between agents and analyze the conversation history.

```python
async def analyze_sequential_flow(city: str):
    """Analyze the sequential flow between agents."""

    display(HTML(f"""
    <div style='padding: 20px; background: #f3e5f5; border-left: 4px solid #9c27b0; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #7b1fa2;'>Sequential Flow Analysis for {city}</h3>
        <p style='margin: 0;'>Examining agent interactions and information handoff...</p>
    </div>
    """))

    # Run the workflow
    events = await workflow.run(f"I want to visit an attraction in {city}")
    outputs = events.get_outputs()

    if outputs:
        messages: list[ChatMessage] = outputs[0]
        display(HTML(f"""
        <div style='padding: 25px; background: #f3e5f5; border-radius: 12px; margin: 20px 0;'>
            <h2 style='margin: 0 0 20px 0; color: #7b1fa2;'>Conversation Flow Analysis</h2>
        </div>
        """))

        # Display each message in the sequence
        for i, msg in enumerate(messages, 1):
            role_color = {
                "user": "#2196f3",
                "front_desk_agent": "#4caf50",
                "concierge_agent": "#ff9800"
            }.get(msg.author_name or "user", "#666666")

            role_name = {
                "user": "👤 User",
                "front_desk_agent": "🏨 Front Desk Agent",
                "concierge_agent": "🎩 Concierge Agent"
            }.get(msg.author_name or "user", "Unknown")

            # Truncate long messages for flow analysis
            content_preview = msg.text[:200] + \
                "..." if len(msg.text) > 200 else msg.text
            display(HTML(f"""
            <div style='padding: 15px; background: white; border-left: 4px solid {role_color}; border-radius: 4px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <span style='font-weight: bold; color: {role_color}; margin-right: 10px;'>Step {i}:</span>
                    <span style='font-weight: bold; color: {role_color};'>{role_name}</span>
                </div>
                <div style='color: #555; font-size: 14px; line-height: 1.4;'>
                    {content_preview}
                </div>
            </div>
            """))

        # Analyze the flow
        display(HTML(f"""
        <div style='padding: 20px; background: linear-gradient(135deg, #9c27b0 0%, #673ab7 100%); color: white; border-radius: 8px; margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0;'>Flow Analysis Summary</h3>
            <ul style='margin: 0; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Total Messages:</strong> {len(messages)}</li>
                <li><strong>Agents Involved:</strong> 2 (Front Desk + Concierge)</li>
                <li><strong>Flow Pattern:</strong> Linear sequential (User → Agent 1 → Agent 2)</li>
                <li><strong>Information Handoff:</strong> Front desk recommendation becomes concierge input</li>
                <li><strong>Output Quality:</strong> Enhanced through expert review and rating</li>
            </ul>
        </div>
        """))

        # Analyze the flow for Barcelona
await analyze_sequential_flow("Barcelona")
            
```