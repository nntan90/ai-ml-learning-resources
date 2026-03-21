# Notebook: 12-chat_summarization

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/12-context-engineering/code_samples/12-chat_summarization.ipynb

---

# Lesson 12 - Chat History Reduction with Agent Scratchpad

This notebook demonstrates how to manage context in long conversations using Microsoft Agent Framework. As conversations grow, the token count increases — eventually exceeding the model's context window. We address this with a **context summarization pattern** and an **agent scratchpad** for persistent memory.

## What You'll Learn:
1. **Why Context Management Matters**: Understanding token limits and context windows
2. **Context-Aware Agents**: Building agents that manage their own conversation context
3. **Context Summarization Pattern**: Using tools to condense conversation history
4. **Agent Scratchpad**: Persistent memory that survives context reduction

## Prerequisites:
- Azure OpenAI setup with environment variables configured
- Understanding of basic agent concepts from previous lessons

## Setup

```python
%pip install agent-framework azure-ai-projects azure-identity --quiet
```

```python
import os
import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential
```

```python
# Load environment variables and create Azure AI Foundry provider
load_dotenv()

provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())

print("✅ Azure AI Foundry provider configured")

```

## Why Context Management Matters

Every LLM has a finite **context window** — the maximum number of tokens it can process in a single request. As a multi-turn conversation progresses:

- **Token count grows linearly** with each user message and assistant reply.
- **Prompt tokens dominate cost** because the entire history is re-sent every turn.
- Eventually the conversation **exceeds the context window** and the model either truncates or errors.

### Strategies for Managing Context

| Strategy | How It Works | Trade-off |
|---|---|---|
| **Truncation** | Drop oldest messages | Loses early context |
| **Summarization** | Condense older messages into a summary | Some detail lost, but key points retained |
| **Scratchpad / External Memory** | Store key facts outside the conversation | Requires tool calls, but survives any reduction |

In this notebook we combine **summarization** with a **scratchpad tool** so the agent can maintain continuity even when conversation history is condensed.

## Creating a Context-Aware Agent

```python
agent = await provider.create_agent(
    name="ContextAwareAgent",
    instructions="""You are a helpful travel planning assistant with excellent memory management.
When conversations get long:
1. Summarize previous context into key points
2. Track user preferences mentioned earlier
3. Reference previous decisions without repeating full details
Always maintain continuity while being concise.""",
)

print("🤖 Context-aware travel planning agent created")
```

## Simulating a Long Conversation

Let's walk through a multi-turn conversation to see how context accumulates. The agent should retain key details (preferences, budget, travel dates) across turns and demonstrate continuity.

```python
session = agent.create_session()

# Turn 1 - Initial preferences
response = await agent.run("I'm planning a trip to Japan. I love sushi, temples, and photography.", session=session)
print(f"Turn 1: {response}\n")

# Turn 2 - More details
response = await agent.run("My budget is $3000 and I'll be traveling solo for 10 days in April.", session=session)
print(f"Turn 2: {response}\n")

# Turn 3 - Test context retention
response = await agent.run("Based on everything I've told you so far, what's the one thing you'd recommend I not miss?", session=session)
print(f"Turn 3: {response}\n")
```

Notice how the agent retains context from earlier turns — it knows about Japan, sushi, temples, photography, the $3000 budget, solo travel, and the April timeframe. In a short conversation this works well, but as the conversation grows the full history becomes expensive to re-send.

Let's continue the conversation with more turns to see context accumulation:

```python
# Turn 4 - Expand the conversation
response = await agent.run("What about accommodation? I prefer traditional Japanese inns.", session=session)
print(f"Turn 4: {response}\n")

# Turn 5 - Change of plans
response = await agent.run("Actually, I've changed my mind about the dates. I'll go in October instead for the autumn colors.", session=session)
print(f"Turn 5: {response}\n")

# Turn 6 - Test retention after change
response = await agent.run("Summarize my complete travel plan so far — destination, budget, duration, interests, accommodation, and timing.", session=session)
print(f"Turn 6: {response}\n")
```

## Context Summarization Pattern

As the conversation grows, we can use a **summarization tool** to condense accumulated context into a compact format. The agent calls this tool to record key preferences so that even if older messages are dropped, the essential information is preserved.

This pattern is the building block for more sophisticated history reduction:
1. The agent identifies key facts from the conversation
2. It calls the summarization tool to persist them
3. Older messages can be safely removed because the summary captures what matters

Below we define a `summarize_preferences` tool that the agent can call to record a compact summary of what it has learned.

```python
@tool(approval_mode="never_require")
def summarize_preferences(conversation_notes: str) -> str:
    """Summarize accumulated user preferences into a compact format."""
    return f"[SUMMARY] User preferences recorded: {conversation_notes}"


# Create an enhanced agent with the summarization tool
summarizing_agent = await provider.create_agent(
    name="SummarizingTravelAgent",
    instructions="""You are a helpful travel planning assistant that actively manages conversation context.

CONTEXT MANAGEMENT RULES:
1. After gathering several user preferences, call summarize_preferences() to record a compact summary
2. When the user asks you to recall details, reference your recorded summaries
3. Keep responses concise — avoid restating the entire history

PLANNING PROCESS:
1. Gather user preferences (destination, budget, dates, interests)
2. Summarize preferences using the tool
3. Create recommendations based on the summary
4. Update the summary when preferences change""",
    tools=[summarize_preferences],
)

print("🤖 Summarizing travel agent created with context tools")
```

```python
# Demonstrate the summarization pattern
summary_session = summarizing_agent.create_session()

# Provide a batch of preferences
response = await summarizing_agent.run(
    "I want to visit Greece. I love seafood, history, and island hopping. "
    "Budget is $4000 for two weeks. Traveling with my partner in June. "
    "Please record these preferences using your summarization tool.",
    session=summary_session,
)
print(f"Agent: {response}\n")

# Ask the agent to use the recorded context
response = await summarizing_agent.run(
    "Now, based on what you've recorded, suggest the top 3 islands we should visit.",
    session=summary_session,
)
print(f"Agent: {response}\n")
```

## Summary

In this lesson you learned how to manage context in long-running agent conversations using Microsoft Agent Framework:

### Key Concepts
- **Context windows are finite** — every token in the conversation history costs money and counts toward the limit.
- **Summarization tools** let the agent condense accumulated context into compact summaries, reducing token usage while preserving essential information.
- **Agent scratchpads** provide persistent external memory that survives any conversation reduction.

### What You Built
- A **context-aware agent** that maintains continuity across multi-turn conversations
- A **summarization tool** (`summarize_preferences`) that records key user details in a compact format
- A **multi-turn conversation** demonstrating context retention and change handling

### Real-World Applications
- **Customer Service Bots**: Remember preferences across long support sessions
- **Personal Assistants**: Track ongoing projects without re-explaining context
- **Educational Tutors**: Maintain student progress across many interactions

### Next Steps
- Implement a full scratchpad tool with file-based persistence
- Add automatic history truncation after summarization
- Combine with vector databases for semantic memory search
- Build agents that can resume conversations days later with full context