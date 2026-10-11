# Human-in-the-Loop (HITL) 👤🔄

> Guide to Human-in-the-Loop (HITL) workflows, breakpoints, approvals, state editing, and human intervention in modern LangChain & LangGraph agents.

---

## 1. Overview

When an agent proposes an action that carries risks or requires confirmation—such as modifying files, executing SQL queries, sending emails, or transferring money—the HITL middleware can pause execution and wait for human input.

### How it works:
1. **Policy Inspection**: The middleware checks each proposed tool call against a configurable policy (`interrupt_on`).
2. **Interrupt & Persistence**: If review is needed, the middleware raises an `interrupt`, safely pausing graph execution. State is automatically saved via a checkpointer (e.g. `InMemorySaver` or `PostgresSaver`).
3. **Human Decision**: A reviewer supplies a decision (`approve`, `edit`, `reject`, or `respond`).
4. **Resumed Execution**: The graph resumes with the approved, edited, or synthesized result.

---

## 2. Interrupt Decision Types

The middleware supports four built-in response actions:

| Decision Type | Description | Example Use Case |
| :--- | :--- | :--- |
| ✅ **`approve`** | Execute the tool with the original arguments proposed by the agent. | Send an email draft exactly as written. |
| ✏️ **`edit`** | Modify the tool arguments before execution. | Fix recipient email address or change parameters. |
| ❌ **`reject`** | Skip tool execution completely and return rejection feedback to the agent. | Deny file deletion and explain why. |
| 💬 **`respond`** | Return human's message directly as a synthetic tool result (skips execution). | Answer an `"ask_user"` prompt directly. |

> [!TIP]
> When **editing** tool arguments, make changes conservatively. Significant modifications to the original arguments may cause the model to re-evaluate its approach and potentially execute the tool multiple times or take unexpected actions.

> [!IMPORTANT]
> - Use **`reject`** when denying an action.
> - Use **`respond`** only when the human is acting as the tool (such as answering an `ask_user` tool). Do **not** use `respond` to deny side-effecting tools, because its message is treated as a successful tool result.

---

## 3. Configuring Interrupts

Configure `HumanInTheLoopMiddleware` with an `interrupt_on` dictionary mapping tool names to approval policies:

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
llm = init_chat_model("groq:qwen/qwen3.8-27b")

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"Wrote to {path}"

@tool
def execute_sql(query: str) -> str:
    """Execute SQL query."""
    return f"Executed: {query}"

@tool
def read_data(item_id: str) -> str:
    """Safe read operation."""
    return f"Data for {item_id}"

agent = create_agent(
    model=llm,
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # All decisions (approve, edit, reject, respond) allowed
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # No editing allowed
                "read_data": False,  # Safe operation: auto-approve without interruption
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    # Checkpointer is mandatory for HITL to persist state across pauses
    checkpointer=InMemorySaver(),
)
```

> [!NOTE]
> You must configure a checkpointer to persist the graph state across interrupts. In production, use persistent storage such as `AsyncPostgresSaver` or `MongoDBSaver`. For local testing, use `InMemorySaver`.
> Always pass a `config` containing a `thread_id` to associate execution with a specific conversation thread.

### Configuration Reference

- **`interrupt_on`** (`dict`): Mapping of tool names to approval configs.
  - `True`: Interrupt with default config (all decisions allowed).
  - `False`: Auto-approve without pausing.
  - `dict` (`InterruptOnConfig`): Fine-grained configuration.
- **`description_prefix`** (`str`, default: `"Tool execution requires approval"`): Prefix for action descriptions shown to reviewers.
- **`InterruptOnConfig` fields**:
  - `allowed_decisions` (`list[str]`): List of allowed decisions: `["approve", "edit", "reject", "respond"]`.
  - `description` (`str | callable`): Static string or custom function returning description.
  - `when` (`callable`): Predicate function receiving `ToolCallRequest` returning `True` (interrupt) or `False` (auto-approve).

---

## 4. Conditional Interrupts (`when`)

Use the `when` predicate to pause only specific high-risk invocations (e.g., based on parameters) while letting safe ones run automatically:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest

def writes_outside_workspace(request: ToolCallRequest) -> bool:
    """Pause writes targeting paths outside the /workspace/ directory."""
    path = request.tool_call["args"].get("path", "")
    return not path.startswith("/workspace/")

def is_write_query(request: ToolCallRequest) -> bool:
    """Pause SQL queries that mutate data (non-SELECT queries)."""
    query = request.tool_call["args"].get("query", "")
    return not query.lstrip().upper().startswith("SELECT")

middleware = [
    HumanInTheLoopMiddleware(
        interrupt_on={
            "write_file": {
                "allowed_decisions": ["approve", "edit", "reject"],
                "when": writes_outside_workspace,
            },
            "execute_sql": {
                "allowed_decisions": ["approve", "reject"],
                "when": is_write_query,
            },
        }
    )
]
```

---

## 5. Responding to Interrupts

When an interrupt is triggered, the agent invocation pauses. You can inspect the interrupt and resume with `Command(resume=...)`.

### Step 1: Initial Invocation (Triggers Interrupt)

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "thread-audit-001"}}

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Delete old records from the database"}
        ]
    },
    config=config,
    version="v2",
)

# Inspect pending interrupt
print("Pending Interrupts:", result.interrupts)
```

### Step 2: Resuming with Human Decisions

#### Option 1: ✅ `approve` (Run as proposed)
```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {"type": "approve"}
            ]
        }
    ),
    config=config,
    version="v2",
)
```

#### Option 2: ✏️ `edit` (Modify arguments before running)
```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "write_file",
                        "args": {"path": "/workspace/safe_output.txt", "content": "Sanitized data"}
                    }
                }
            ]
        }
    ),
    config=config,
    version="v2",
)
```

#### Option 3: ❌ `reject` (Cancel tool call and give feedback)
```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    "message": "User rejected this SQL delete operation. Do not delete any records."
                }
            ]
        }
    ),
    config=config,
    version="v2",
)
```

#### Option 4: 💬 `respond` (Synthetic response for interactive prompts)
```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "respond",
                    "message": "The user selected option B."
                }
            ]
        }
    ),
    config=config,
    version="v2",
)
```

---

## 6. Multiple Decisions Batching

When multiple tool calls are triggered in parallel, supply decisions in the exact order of actions:

```python
resume_payload = {
    "decisions": [
        {"type": "approve"},
        {
            "type": "edit",
            "edited_action": {
                "name": "write_file",
                "args": {"path": "/workspace/data.json", "content": "{}"}
            }
        },
        {
            "type": "reject",
            "message": "Skipping secondary operation due to permission limits."
        }
    ]
}

agent.invoke(Command(resume=resume_payload), config=config, version="v2")
```

---

## 7. Streaming with Human-in-the-Loop

Use `stream_events()` to stream tokens live and detect when an interrupt occurs:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "stream-session-1"}}

# 1. Stream agent progress and LLM tokens until interrupt
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Delete records from database"}]},
    config=config,
    version="v3",
)

for message in stream.messages:
    for token in message.text:
        print(token, end="", flush=True)

# 2. Check for interruption
if stream.interrupted:
    print(f"\n\n[PAUSED] Action requires human decision: {stream.interrupts}")

# 3. Resume streaming after decision
stream = agent.stream_events(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    version="v3",
)

for message in stream.messages:
    for token in message.text:
        print(token, end="", flush=True)
```

---

## 8. Execution Lifecycle

`HumanInTheLoopMiddleware` uses an `after_model` hook:

1. **Model Invocation**: LLM generates a response that includes one or more tool calls.
2. **Evaluation**: Middleware checks the tool calls against `interrupt_on` and `when` rules.
3. **Interrupt**: If matched, builds a `HITLRequest` and halts with `interrupt()`.
4. **Decision Handling**: Upon receiving `resume`, executes approved/edited calls, synthesizes `ToolMessage` for rejected calls, or injects answers for `respond` calls.
5. **Resume**: Control passes back to the model with tool execution results.

---

## Official Documentation
- [LangChain Human-in-the-loop Documentation](https://docs.langchain.com/oss/python/langchain/middleware/built-in#human-in-the-loop)
- [LangGraph Interrupts Guide](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Persistence Reference](https://docs.langchain.com/oss/python/langgraph/persistence)
