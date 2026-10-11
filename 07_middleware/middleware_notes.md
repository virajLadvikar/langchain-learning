# LangChain Middleware — Revision & Practice Notes ⚙️🔄

> Short notes based on the LangChain middleware overview, prebuilt middleware, and custom middleware documentation. Contains core concepts, reference tables, and complete runnable examples.

---

## 1. What is Middleware? 💡

Middleware lets you intercept, control, or customize agent execution at specific lifecycle points.

### Main Uses 🎯
- **Observability**: Log, trace, and debug agent requests and tool invocations.
- **Dynamic Prompts & Tool Selection**: Modify prompts or choose tools/models dynamically based on runtime context.
- **Reliability & Resilience**: Retry failed model or tool calls; use fallback models.
- **Memory Optimization**: Summarize long conversation histories automatically.
- **Governance & Safety**: Add human approval (HITL), rate limits, and PII detection/redaction.

Middleware is passed directly to `create_agent(..., middleware=[...])`. It executes inside the LangGraph runtime returned by `create_agent`; it is not a separate external server or layer.

---

## 2. Complete Quickstart Example 🚀

Here is a fully runnable end-to-end example creating an agent with retry and call-limit middleware:

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ModelCallLimitMiddleware,
)
from langchain.tools import tool

# 1. Load environment variables (API keys)
load_dotenv()

# 2. Initialize the chat model
llm = init_chat_model("groq:qwen/qwen3.8-27b")
# Or: llm = init_chat_model("google_genai:gemini-2.5-flash")

# 3. Define tools
@tool("get_weather", description="Get current weather for a city")
def get_weather(city: str) -> str:
    """Fetch current weather."""
    return f"The weather in {city} is sunny and 24°C."

# 4. Create the agent with middleware
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful travel assistant. Always use the weather tool.",
    middleware=[
        ModelRetryMiddleware(max_retries=3, initial_delay=1.0, backoff_factor=2.0),
        ModelCallLimitMiddleware(run_limit=5),
    ],
)

# 5. Invoke the agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
})

# 6. Print the result
print(response["messages"][-1].content)
```

---

## 3. The Agent Loop & Middleware Execution

The standard agent loop consists of:

```text
       ┌──────────────┐
       │ User Input   │
       └──────┬───────┘
              │  [before_agent]
              ▼
     ┌──────────────────┐
     │  before_model    │
     └────────┬─────────┘
              ▼
    [wrap_model_call] ──► Chat Model Invocations
              ▼
     ┌──────────────────┐
     │   after_model    │
     └────────┬─────────┘
              │ Tool call requested?
       ┌──────┴──────┐
      YES            NO ──► [after_agent] ──► Final Output
       │
       ▼
  [wrap_tool_call] ──► Execute Tool ──► Tool Result
       │
       └────── Loops back to before_model
```

Middleware hooks let you inspect or alter state before, during, or after model and tool executions.

---

## 4. Built-in Middleware Reference

| Middleware | Purpose | Key Parameters |
| --- | --- | --- |
| `ToolRetryMiddleware` | Retries failed tool calls | `max_retries`, `initial_delay`, `backoff_factor` |
| `ModelRetryMiddleware` | Retries failed model API calls | `max_retries`, `initial_delay`, `backoff_factor` |
| `ModelFallbackMiddleware` | Tries secondary models if the primary model fails | `models=[fallback_model]` |
| `SummarizationMiddleware` | Summarizes older history when context limit is reached | `model`, `trigger=("tokens", 4000)`, `keep=("messages", 20)` |
| `HumanInTheLoopMiddleware` | Pauses execution for human approval/editing | `interrupt_on={"tool_name": True}` (Requires checkpointer) |
| `ModelCallLimitMiddleware` | Caps maximum LLM calls to prevent infinite loops | `run_limit=10` |
| `ToolCallLimitMiddleware` | Caps maximum tool invocations per step/run | `run_limit=10` |
| `ToolErrorMiddleware` | Catches tool exceptions and formats them as model messages | Exception mapping |
| `PIIMiddleware` | Detects or redacts sensitive personal data | Pattern matchers and redaction rules |

---

## 5. Complete Runnable Examples for Built-in Middleware

### A. Fallback Models (`ModelFallbackMiddleware`)

Automatically route to a backup model if the primary model fails or hits rate limits:

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

load_dotenv()

# Primary and fallback models
primary_model = init_chat_model("groq:qwen/qwen3.8-27b")
fallback_model = init_chat_model("google_genai:gemini-2.5-flash")

agent = create_agent(
    model=primary_model,
    tools=[],
    system_prompt="You are a reliable AI assistant.",
    middleware=[
        ModelFallbackMiddleware(fallback_model)
    ],
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}]
})

print(response["messages"][-1].content)
```

---

### B. Conversation Summarization (`SummarizationMiddleware`)

Automatically compresses long conversations while preserving recent context:

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = init_chat_model("groq:qwen/qwen3.8-27b")

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 2000),  # Trigger when tokens reach 2000
            keep=("messages", 10),     # Keep the 10 most recent messages intact
        )
    ],
)

config = {"configurable": {"thread_id": "session-123"}}

# Multi-turn interaction
agent.invoke({"messages": [{"role": "user", "content": "Hello, my project is Project Alpha."}]}, config)
response = agent.invoke({"messages": [{"role": "user", "content": "What project did I mention?"}]}, config)

print(response["messages"][-1].content)
```

---

### C. Human-in-the-Loop Approval (`HumanInTheLoopMiddleware`)

Pauses tool execution to require confirmation before taking risky actions (e.g., sending emails, deleting records, transferring funds).

> 📖 **Full Guide**: See [hitl_middleware_notes.md](hitl_middleware_notes.md) for conditional `when` predicates, all 4 decision types, and streaming interrupts.

#### The 4 Decision Types
| Decision Type | Description | Example |
| :--- | :--- | :--- |
| ✅ **`approve`** | Run tool call as proposed | Send email draft as-is |
| ✏️ **`edit`** | Modify arguments before running | Change recipient or amount |
| ❌ **`reject`** | Cancel tool and return feedback message | Deny deletion with reason |
| 💬 **`respond`** | Direct synthetic answer for interactive prompt | Human answers an `ask_user` tool |

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

llm = init_chat_model("groq:qwen/qwen3.8-27b")

@tool("read_balance", description="Check account balance")
def read_balance(account_id: str) -> str:
    """Read user balance."""
    return f"Account {account_id} balance: $5,000"

@tool("transfer_funds", description="Transfer money between accounts")
def transfer_funds(account_id: str, amount: float) -> str:
    """Perform funds transfer."""
    return f"Successfully transferred ${amount} from account {account_id}."

checkpointer = InMemorySaver()

# Interrupt specifically before transfer_funds executes
agent = create_agent(
    model=llm,
    tools=[read_balance, transfer_funds],
    checkpointer=checkpointer,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "transfer_funds": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_balance": False,  # Safe read, auto-approve
            }
        )
    ],
)

config = {"configurable": {"thread_id": "tx-456"}}

# Step 1: Agent will pause before executing transfer_funds
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Transfer $500 from account ACC123"}]},
    config=config,
    version="v2",
)

# Inspect interrupts
if result.interrupts:
    print("Pending Interrupt:", result.interrupts)

# Step 2: Resume with human approval decision
resumed = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    version="v2",
)

print(resumed["messages"][-1].content)
```

---

## 6. Complete Runnable Examples for Custom Middleware

### A. Node-Style Hooks (`@before_model` & `@after_model`)

Node-style hooks run at lifecycle points and can inspect state or return state updates:

```python
import os
from typing import Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain.tools import tool
from langgraph.runtime import Runtime

load_dotenv()
llm = init_chat_model("groq:qwen/qwen3.8-27b")

@tool("calculator", description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

# 1. Inspect state before each model call
@before_model
def audit_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    msg_count = len(state["messages"])
    print(f"\n[Audit Before Model] Current message history depth: {msg_count}")
    return None

# 2. Inspect response after each model call
@after_model
def audit_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_msg = state["messages"][-1]
    print(f"[Audit After Model] Model emitted type: {type(last_msg).__name__}")
    return None

agent = create_agent(
    model=llm,
    tools=[add],
    middleware=[audit_before_model, audit_after_model],
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 25 + 75?"}]
})

print("\nFinal Answer:", response["messages"][-1].content)
```

---

### B. Wrap-Style Hooks (`@wrap_model_call` & `@wrap_tool_call`)

Wrap-style hooks surround execution like Python decorators or HTTP middleware, allowing performance metrics, custom error recovery, or retries:

```python
import time
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain.tools import tool

load_dotenv()
llm = init_chat_model("groq:qwen/qwen3.8-27b")

@tool("get_stock_price", description="Fetch stock price")
def get_stock_price(ticker: str) -> str:
    time.sleep(0.1)  # Simulate network latency
    return f"{ticker}: $185.50"

# Timing wrapper around model execution
@wrap_model_call
def time_model_call(handler, request):
    start = time.perf_counter()
    print("[Wrap Model] Dispatching model request...")
    result = handler(request)
    elapsed = time.perf_counter() - start
    print(f"[Wrap Model] Call completed in {elapsed:.3f}s")
    return result

# Logging wrapper around tool execution
@wrap_tool_call
def log_tool_execution(handler, request):
    print(f"[Wrap Tool] Executing tool: '{request.tool_name}' with args: {request.tool_args}")
    result = handler(request)
    print(f"[Wrap Tool] Tool returned: {result}")
    return result

agent = create_agent(
    model=llm,
    tools=[get_stock_price],
    middleware=[time_model_call, log_tool_execution],
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the price of AAPL?"}]
})

print("\nFinal Response:\n", response["messages"][-1].content)
```

---

### C. Dynamic Prompts from Runtime Context (`@dynamic_prompt`)

Dynamically inject personalized or domain-specific instructions into the system prompt:

```python
from typing import TypedDict
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

load_dotenv()
llm = init_chat_model("groq:qwen/qwen3.8-27b")

# Define schema for invocation context
class UserContext(TypedDict):
    user_name: str
    preferred_tone: str

# Dynamic prompt hook reads context per invocation
@dynamic_prompt
def custom_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.get("user_name", "Valued Customer")
    tone = request.runtime.context.get("preferred_tone", "professional")
    return (
        f"You are a friendly personal assistant. "
        f"The user's name is {user_name}. Always respond in a {tone} tone."
    )

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[custom_system_prompt],
    context_schema=UserContext,
)

# Run with customized context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Give me a one-line motivation for Monday."}]},
    context={"user_name": "Alice", "preferred_tone": "enthusiastic and cheerful"}
)

print(result["messages"][-1].content)
```

---

### D. Class-Based Middleware (`AgentMiddleware`)

When bundling multiple hooks, maintaining state, or distributing reusable middleware packages, subclass `AgentMiddleware`:

```python
import os
from typing import Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langgraph.runtime import Runtime

load_dotenv()
llm = init_chat_model("groq:qwen/qwen3.8-27b")

@tool("search_docs", description="Search internal docs")
def search_docs(query: str) -> str:
    return f"Documentation extract matching query '{query}'"

class SecurityAuditMiddleware(AgentMiddleware):
    def __init__(self, service_tag: str):
        self.service_tag = service_tag

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[{self.service_tag}] Agent starting session...")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[{self.service_tag}] Model call pending...")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[{self.service_tag}] Agent completed successfully.")
        return None

agent = create_agent(
    model=llm,
    tools=[search_docs],
    middleware=[SecurityAuditMiddleware(service_tag="SECURITY_AGENT_V1")],
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Find docs on API authentication."}]
})

print(response["messages"][-1].content)
```

---

## 7. Middleware Execution Order 🔄

When registering multiple middleware in `create_agent(..., middleware=[M1, M2])`:

1. **`before_*` hooks**: Run in **declaration order** (`M1` -> `M2`).
2. **`wrap_*` hooks**: Nest from the outside in (first middleware wraps later ones: `M1(M2(target))`).
3. **`after_*` hooks**: Run in **reverse order** (`M2` -> `M1`).

---

## 8. Quick Practice Checklist 📋

- [ ] Run an agent with `ModelRetryMiddleware` and `ModelCallLimitMiddleware`.
- [ ] Implement `ModelFallbackMiddleware` switching from a primary to a secondary model.
- [ ] Configure `SummarizationMiddleware` with an `InMemorySaver` checkpointer.
- [ ] Implement `HumanInTheLoopMiddleware` pausing on a sensitive action.
- [ ] Write a custom `@before_model` logging hook.
- [ ] Write a custom `@wrap_model_call` calculating execution time.
- [ ] Build a `@dynamic_prompt` reading context from `runtime.context`.
- [ ] Implement a class-based `AgentMiddleware`.

---

## 9. Interview Revision — One-Line Answers 🎯

- **What is middleware?** A mechanism to intercept, observe, and modify agent execution around model and tool steps.
- **Why use retry middleware?** To handle transient network or rate-limit failures without failing the entire run.
- **Retry vs Fallback?** Retry tries the same model again with backoff; Fallback switches to an alternative model provider.
- **Node-style vs Wrap-style?** Node-style hooks execute sequentially and can update state; Wrap-style hooks wrap execution to control or time the underlying handler.
- **Why does HITL middleware need a checkpointer?** Because execution pauses and the thread state must be saved to disk/memory until resumed.

---

## 10. Official Documentation 📚
- [LangChain Middleware Overview](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [Prebuilt Middleware Reference](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [Custom Middleware Guide](https://docs.langchain.com/oss/python/langchain/middleware/custom)
