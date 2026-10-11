# LangChain Tools 🔧

LangChain **Tools** are callable functions that allow agents to interact with external systems, query databases, execute code, fetch live information, and trigger actions in the real world.

The model decides **when to invoke a tool** and **what input arguments to provide** based on the conversation context.

---

## 📑 Contents

1. [Creating Tools (`@tool`)](#1-creating-tools-tool)
2. [Input Schema & Validation (Pydantic / JSON Schema)](#2-input-schema--validation-pydantic--json-schema)
3. [Custom Name & Description](#3-custom-name--description)
4. [Tool Runtime & Context Access (`ToolRuntime`)](#4-tool-runtime--context-access-toolruntime)
5. [Tool Return Values & Types](#5-tool-return-values--types)
   - [String Return](#a-return-a-string)
   - [Structured Object (dict)](#b-return-an-object)
   - [Multimodal Content Blocks](#c-return-multimodal-content)
   - [Command (State Updates)](#d-return-a-command-state-mutation)
   - [Direct Tool Returns (`return_direct=True`)](#e-direct-tool-returns-return_directtrue)
6. [Tool Execution Flow](#6-tool-execution-flow)
7. [Error Handling via Middleware](#7-error-handling-via-middleware)
8. [Quick Revision Reference](#8-quick-revision-reference)

---

## 1. Creating Tools (`@tool`)

The simplest way to create a tool is with the `@tool` decorator.

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for.
        limit: Maximum number of results to return (default 10).
    """
    return f"Found {limit} records for query '{query}'"
```

### ⚠️ Essential Rules:
- **Type hints are required**: LangChain uses Python type annotations to generate the tool's JSON schema for the LLM.
- **Docstrings provide instructions**: The docstring explains the purpose of the tool and its parameters to the model.
- **Naming convention**: Prefer `snake_case` (e.g., `web_search` instead of `Web Search`) to avoid provider compatibility errors.

---

## 2. Input Schema & Validation (Pydantic / JSON Schema)

For complex inputs or strict validation, use **Pydantic models**:

```python
from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool

class WeatherInput(BaseModel):
    """Input schema for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Whether to include 5-day weather forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast for a location."""
    temp = 25 if units == "celsius" else 77
    result = f"Current weather in {location}: {temp}°{units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

### 🚫 Reserved Parameter Names

Do not name your tool parameters `config` or `runtime`:

| Reserved Name | Purpose |
| :--- | :--- |
| `config` | Reserved for internal `RunnableConfig` |
| `runtime` | Reserved for `ToolRuntime` context injection |

---

## 3. Custom Name & Description

You can explicitly override the tool name and description when passing it to the `@tool` decorator:

```python
@tool("calculator", description="Performs arithmetic calculations. Use this for math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

print(calc.name)         # calculator
print(calc.description)  # Performs arithmetic calculations...
```

---

## 4. Tool Runtime & Context Access (`ToolRuntime`)

The `ToolRuntime` parameter gives tools access to runtime memory, state, context, and stream channels.

> 💡 **Note**: The `runtime` parameter is automatically injected by LangChain/LangGraph and is **completely hidden from the model's schema**.

| Runtime Component | Access Property | Purpose |
| :--- | :--- | :--- |
| **Short-term Memory** | `runtime.state` | Mutable state for current conversation (messages, custom state) |
| **Context** | `runtime.context` | Immutable per-run config (user_id, session auth) |
| **Long-term Store** | `runtime.store` | Persistent cross-session database/store |
| **Stream Writer** | `runtime.stream_writer` | Emit live progress updates during long-running tool execution |
| **Execution Info** | `runtime.execution_info` | Run ID, thread ID, retry attempts |
| **Server Info** | `runtime.server_info` | LangGraph Server assistant ID, graph ID, user identity |
| **Tool Call ID** | `runtime.tool_call_id` | Unique ID of the current tool invocation |

### Example: Accessing Conversation History & Preferences

```python
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage

@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message sent by the user."""
    messages = runtime.state["messages"]
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return "No user message found."
```

### Example: Streaming Progress from Tools

```python
@tool
def process_data(dataset_id: str, runtime: ToolRuntime) -> str:
    """Process large datasets with real-time progress updates."""
    writer = runtime.stream_writer
    writer("Step 1: Downloading dataset...")
    writer("Step 2: Transforming records...")
    writer("Step 3: Indexing database...")
    return f"Dataset {dataset_id} processed successfully."
```

---

## 5. Tool Return Values & Types

### A. Return a String
Plain text output converted directly into a `ToolMessage`:

```python
@tool
def get_status(service_name: str) -> str:
    """Check if a service is operational."""
    return f"Service {service_name} is operational."
```

### B. Return an Object
Returns structured dictionaries/JSON that the model can inspect field-by-field:

```python
@tool
def get_user_profile(user_id: str) -> dict:
    """Fetch structured profile information."""
    return {
        "id": user_id,
        "name": "Alice",
        "tier": "enterprise",
        "quota_remaining": 1500
    }
```

### C. Return Multimodal Content
Return structured content blocks with images, audio, or files:

```python
@tool
def capture_chart(symbol: str) -> list[dict]:
    """Generate a financial chart image."""
    return [
        {"type": "text", "text": f"Stock chart for {symbol}:"},
        {"type": "image", "url": f"https://charts.example.com/{symbol}.png"}
    ]
```

### D. Return a `Command` (State Mutation)
Use `Command` when a tool needs to mutate the agent/graph state:

```python
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain.tools import tool, ToolRuntime

@tool
def set_preferred_language(lang: str, runtime: ToolRuntime) -> Command:
    """Update the user's preferred language in state."""
    return Command(
        update={
            "preferred_lang": lang,
            # Must include ToolMessage matching tool_call_id to keep history valid
            "messages": [
                ToolMessage(
                    content=f"Language set to {lang}.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )
```

### E. Direct Tool Returns (`return_direct=True`)
Short-circuits the agent loop: returns the tool output directly to the user **without an extra model call**:

```python
@tool(return_direct=True)
def get_order_tracking(order_id: str) -> str:
    """Look up tracking status and return directly to user."""
    return f"Order #{order_id} is in transit and arriving tomorrow."
```

---

## 6. Tool Execution Flow

```text
       ┌──────────────┐
       │     User     │
       └──────┬───────┘
              │ 1. "What's the weather in Boston?"
              ▼
       ┌──────────────┐
       │   AI Model   │ ◄── Decides tool & extracts arguments
       └──────┬───────┘
              │ 2. AIMessage(tool_calls=[get_weather(location='Boston')])
              ▼
       ┌──────────────┐
       │ Tool Engine  │ ◄── Executes get_weather('Boston')
       └──────┬───────┘
              │ 3. ToolMessage(content="72°F and sunny", tool_call_id="call_1")
              ▼
       ┌──────────────┐
       │   AI Model   │ ◄── Formulates final response with tool result
       └──────┬───────┘
              │ 4. "The weather in Boston is 72°F and sunny."
              ▼
       ┌──────────────┐
       │     User     │
       └──────────────┘
```

---

## 7. Error Handling via Middleware

Wrap tool executions in middleware to convert exceptions into graceful `ToolMessage` payloads instead of crashing:

```python
from collections.abc import Callable
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage]
) -> ToolMessage:
    """Catch tool exceptions and return error messages to the model."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error encountered: {e}. Please check your arguments.",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[search_database],
    middleware=[handle_tool_errors]
)
```

---

## 8. Quick Revision Reference

| Concept | Key Summary |
| :--- | :--- |
| **`@tool`** | Decorator to convert a Python function into a LangChain tool |
| **Type Hints** | Required; automatically generates JSON input schema for LLM |
| **Docstrings** | Explains the tool's purpose and usage to the LLM |
| **`args_schema`** | Pydantic model for complex schemas and validation |
| **`ToolRuntime`** | Injected runtime object for state, context, store, and streaming |
| **`ToolMessage`** | Message containing tool execution result (`tool_call_id` required) |
| **`Command`** | State mutation return type from tools |
| **`return_direct=True`**| Short-circuits loop to return tool output directly to user |
| **`wrap_tool_call`** | Middleware hook to intercept tool calls and handle errors |

---

### 🔑 Core Takeaway

> **Tools give LLMs hands and eyes to interact with external environments, execute computation, fetch real-time data, and mutate application state.**
