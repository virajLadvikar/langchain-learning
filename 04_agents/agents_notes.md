# LangChain Agents 🤖

An **Agent** is a language model calling tools in a loop until a given task is complete, governed by a **harness** (prompts and tools).

> 💡 **Core Formula:**  
> **Agent = Model + Harness**  
> The job of the harness is to get the model the right context and tools at the right time.

---

## 📑 Contents

1. [Overview & Core Agent Loop](#1-overview--core-agent-loop)
2. [Quickstart (`create_agent`)](#2-quickstart-create_agent)
3. [Core Components](#3-core-components)
   - [Model](#a-model)
   - [Tools](#b-tools)
   - [System Prompt](#c-system-prompt)
   - [Structured Output (`response_format`)](#d-structured-output-response_format)
4. [Quick Revision Reference](#4-quick-revision-reference)

---

## 1. Overview & Core Agent Loop

An agent executes in an iterative feedback loop:

```text
       ┌──────────────┐
       │     User     │
       └──────┬───────┘
              │ 1. User Prompt
              ▼
    ┌────────────────────┐
 ┌──►    Chat Model      │ ◄── Evaluates conversation & decides actions
 │  └─────────┬──────────┘
 │            │ 2. Tool Calls Required?
 │     ┌──────┴──────┐
 │    NO            YES
 │     │             │ 3. Execute Tool
 │     │             ▼
 │     │      ┌──────────────┐
 │     │      │ Tool Engine  │
 │     │      └──────┬───────┘
 │     │             │ 4. ToolMessage (Results)
 │     │             │
 │     └─────────────┴── Updates Conversation
 │
 └────── 5. Loops until model outputs final response (No tool calls)
              │
              ▼
       ┌──────────────┐
       │ Final Output │
       └──────────────┘
```

---

## 2. Quickstart (`create_agent`)

`create_agent` is the universal, highly configurable factory function for building LangChain agents.

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for online information."""
    return f"Results for: {query}"

# Initialize agent with model and tools
agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[search],
    system_prompt="You are a helpful research assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Search for latest AI news."}]
})

print(result["messages"][-1].content)
```

---

## 3. Core Components

### A. Model
Pass either a provider string (`"provider:model"`) or an initialized model instance:
```python
agent = create_agent(model="google_genai:gemini-2.5-flash", tools=tools)
# or OpenAI: model="openai:gpt-4o"
# or Anthropic: model="anthropic:claude-3-7-sonnet-20250219"
```

### B. Tools
Pass Python callables decorated with `@tool` or LangChain tool objects:
```python
@tool
def calculate_tax(amount: float, rate: float = 0.18) -> str:
    """Calculate tax on a given amount."""
    return f"Tax: {amount * rate:.2f}"

agent = create_agent(model="google_genai:gemini-2.5-flash", tools=[calculate_tax])
```

### C. System Prompt
Guides the agent's persona, boundaries, and decision strategy:
```python
agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=tools,
    system_prompt="You are a strict financial auditor. Verify all numbers before responding."
)
```

### D. Structured Output (`response_format`)
Force the agent to conclude with a strictly validated Pydantic model:
```python
from pydantic import BaseModel, Field

class ResearchSummary(BaseModel):
    summary: str = Field(description="Brief summary of findings")
    key_points: list[str] = Field(description="Key takeaway bullet points")
    confidence_score: float = Field(description="Confidence between 0 and 1")

agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[search],
    response_format=ResearchSummary
)

result = agent.invoke({"messages": [{"role": "user", "content": "Research quantum computing."}]})
print(result["structured_response"])
# ResearchSummary(summary="...", key_points=[...], confidence_score=0.95)
```

---

## 4. Quick Revision Reference

| Feature / Concept | Purpose |
| :--- | :--- |
| **`create_agent()`** | Universal agent builder combining model, tools, and system prompt |
| **`Agent = Model + Harness`** | Core design principle: model reasoning paired with tools and prompts |
| **`system_prompt`** | Defines persona, behavior rules, and task constraints |
| **`tools`** | List of `@tool` functions the agent can invoke |
| **`response_format`** | Enforces structured output validated against Pydantic models |

---

### 🔑 Core Takeaway

> **LangChain Agents turn static models into autonomous problem solvers by looping model reasoning with tool execution until a task is completed.**
