# LangChain Short-Term Memory 💾🧠

> Comprehensive revision notes, architectural diagrams, runnable examples, and memory optimization strategies for managing **short-term conversation history** and **state persistence** in modern LangChain & LangGraph agents.

---

## 📑 Contents

1. [Overview & Core Value](#1-overview--core-value-)
2. [How Short-Term Memory Works](#2-how-short-term-memory-works-)
   - [Memory Architecture Flow](#memory-architecture-flow)
   - [Key Concepts & Terminology](#key-concepts--terminology)
3. [Basic Example: Stateful Conversation](#3-basic-example-stateful-conversation-)
   - [The Role of `thread_id`](#the-role-of-thread_id)
4. [In-Memory vs. Database Persistence](#4-in-memory-vs-database-persistence-)
   - [Development: `InMemorySaver`](#development-inmemorysaver)
   - [Production: `PostgresSaver`](#production-postgressaver)
5. [Managing Long Conversation Histories](#5-managing-long-conversation-histories-)
   - [5.1 Trimming Messages](#51-trimming-messages-)
   - [5.2 Deleting Messages (`RemoveMessage`)](#52-deleting-messages-removemessage-)
   - [5.3 Automatic Summarization (`SummarizationMiddleware`)](#53-automatic-summarization-summarizationmiddleware-)
6. [Custom Agent State](#6-custom-agent-state-)
7. [Accessing & Modifying Memory from Tools](#7-accessing--modifying-memory-from-tools-)
8. [Dynamic Prompts from State & Context](#8-dynamic-prompts-from-state--context-)
9. [Lifecycle Middleware Hooks for Memory](#9-lifecycle-middleware-hooks-for-memory-)
10. [Short-Term vs. Long-Term Memory](#10-short-term-vs-long-term-memory-)
11. [Interview Revision Notes & Checklist](#11-interview-revision-notes--checklist-)
12. [Official References](#12-official-references-)

---

## 1. Overview & Core Value 💡

**Short-term memory** enables an AI agent to retain conversational context across consecutive turns within a single thread or session.

- A **Thread** isolates related interactions into one coherent session, similar to a single email thread or chat room.
- Without memory, language models are completely stateless: every invocation is evaluated in total isolation.
- Unchecked conversation growth can exceed the model's context window, increase latency, trigger token bloat, and degrade model reasoning ("lost-in-the-middle" problem).

> [!IMPORTANT]
> **Short-Term vs. Long-Term Memory**:
> - **Short-term memory** is scoped strictly to the **current thread** (session context).
> - **Long-term memory** persists user facts, preferences, and knowledge across **different sessions** and days/weeks.

---

## 2. How Short-Term Memory Works ⚙️

In modern LangChain / LangGraph agents:
1. Short-term memory is persisted as part of the agent's **State** (under the `messages` key).
2. A **Checkpointer** saves and restores state snapshots after every agent step.
3. A unique `thread_id` inside `config={"configurable": {"thread_id": "..."}}` identifies which conversation thread to resume.
4. When a user sends a message, the checkpointer loads the existing thread messages, appends the new input, runs model reasoning, and commits the updated state.

### Memory Architecture Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Agent as Agent Runtime
    participant Saver as Checkpointer (State Saver)
    participant LLM as Chat Model

    User->>Agent: invoke("Hi, I'm Bob", config={thread_id: "conv-1"})
    Agent->>Saver: Load state for "conv-1" (empty initial state)
    Saver-->>Agent: Returns initial state
    Agent->>LLM: Prompt + [HumanMessage("Hi, I'm Bob")]
    LLM-->>Agent: AIMessage("Hello Bob! How can I help?")
    Agent->>Saver: Commit snapshot for "conv-1" (2 messages saved)
    Agent-->>User: "Hello Bob! How can I help?"

    Note over User,Saver: Turn 2 (Resuming the exact same thread)

    User->>Agent: invoke("What's my name?", config={thread_id: "conv-1"})
    Agent->>Saver: Load state for "conv-1"
    Saver-->>Agent: Restores [HumanMessage, AIMessage]
    Agent->>LLM: Prompt + History + [HumanMessage("What's my name?")]
    LLM-->>Agent: AIMessage("Your name is Bob.")
    Agent->>Saver: Commit updated snapshot (4 messages saved)
    Agent-->>User: "Your name is Bob."
```

### Key Concepts & Terminology

| Term | Responsibility | Example |
| :--- | :--- | :--- |
| **Short-Term Memory** | Retaining context within a single thread/conversation | Remembers user's name during chat |
| **Agent State** | Mutable dictionary maintained across execution steps | `{"messages": [...], "user_id": "123"}` |
| **Checkpointer** | Persistent engine that saves and restores state snapshots | `InMemorySaver`, `PostgresSaver` |
| **Thread ID** | Unique session key scoping execution state | `"configurable": {"thread_id": "sess-42"}` |
| **`InMemorySaver`** | Ephemeral in-memory checkpointer for development | Useful for unit tests & interactive notebooks |
| **`PostgresSaver`** | Production database checkpointer for distributed apps | Survives pod/server restarts |

---

## 3. Basic Example: Stateful Conversation 🚀

This runnable example uses `InMemorySaver` to retain conversational history:

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# 1. Initialize chat model
llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)

# 2. Create stateful agent with checkpointer
agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=InMemorySaver(),
)

# 3. Configure unique thread ID
config = {"configurable": {"thread_id": "session-bob-01"}}

# Turn 1: Introduce name
turn_1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob and I am a QA Engineer."}]},
    config=config,
)
print("Turn 1 Response:\n", turn_1["messages"][-1].content)

# Turn 2: Query remembered name
turn_2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name and job title?"}]},
    config=config,
)
print("\nTurn 2 Response:\n", turn_2["messages"][-1].content)
```

### The Role of `thread_id`

`thread_id` isolates conversation contexts completely:

```python
# Session A (Bob's conversation)
config_a = {"configurable": {"thread_id": "thread-bob"}}
agent.invoke({"messages": [{"role": "user", "content": "My name is Bob."}]}, config=config_a)

# Session B (Alice's conversation)
config_b = {"configurable": {"thread_id": "thread-alice"}}
agent.invoke({"messages": [{"role": "user", "content": "My name is Alice."}]}, config=config_b)

# Responding to Session A returns Bob; responding to Session B returns Alice!
```

---

## 4. In-Memory vs. Database Persistence 🗄️

### Development: `InMemorySaver`
```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
```
- ⚡ Ultra-fast, zero dependencies.
- ⚠️ State lives purely in RAM; destroyed upon application restart.

### Production: `PostgresSaver`
For persistent, multi-instance web backends:

```bash
uv add langgraph-checkpoint-postgres "psycopg[binary]"
```

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/langgraph_db?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # Initializes tables on first boot
    
    agent = create_agent(
        model=llm,
        tools=[],
        checkpointer=checkpointer,
    )
```

> [!CAUTION]
> Never commit database passwords or production connection strings to version control. Load them via environment variables or secret management tools.

---

## 5. Managing Long Conversation Histories ✂️

When conversations grow over many turns, older messages inflate prompt token costs and can exceed context limits. LangChain provides three primary management strategies:

### 5.1 Trimming Messages ✂️
Trims messages before passing them to the model while keeping the most recent exchanges:

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages=state["messages"],
    max_tokens=2000,
    strategy="last",
    token_counter=llm,
    start_on="human",
    include_system=True,
)
```

### 5.2 Deleting Messages (`RemoveMessage`) 🗑️
Explicitly purge messages from the checkpointer state:

```python
from langchain_core.messages import RemoveMessage

def purge_old_messages(state):
    messages = state["messages"]
    if len(messages) > 10:
        # Remove the earliest 4 messages
        return {
            "messages": [RemoveMessage(id=msg.id) for msg in messages[:4]]
        }
    return None
```

### 5.3 Automatic Summarization (`SummarizationMiddleware`) 📝
Compress older turns into an evolving running summary:

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="google_genai:gemini-2.5-flash",
            trigger=("tokens", 3000),  # Trigger when token count exceeds 3000
            keep=("messages", 6),       # Retain 6 most recent messages verbatim
        )
    ],
    checkpointer=InMemorySaver(),
)
```

---

## 6. Custom Agent State 🧩

By default, agent state contains the `messages` list. You can extend `AgentState` with custom metadata (e.g., user profiles, permissions, theme settings):

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class CustomAgentState(AgentState):
    user_id: str
    user_role: str
    preferences: dict

agent = create_agent(
    model=llm,
    tools=[],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What is my role?"}],
        "user_id": "usr-9901",
        "user_role": "Lead QA Engineer",
        "preferences": {"theme": "dark"},
    },
    config={"configurable": {"thread_id": "thread-admin"}},
)
```

---

## 7. Accessing & Modifying Memory from Tools 🔧

A tool can inspect or update state using the `ToolRuntime` injection parameter (hidden from the model schema):

```python
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage

@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """Read user metadata stored in the current session memory."""
    state = runtime.state
    user_id = state.get("user_id", "anonymous")
    return f"Active session user: {user_id}"

@tool
def update_user_preference(theme: str, runtime: ToolRuntime) -> str:
    """Update user preferences directly in agent state."""
    # State update can be emitted via Command
    return f"Theme updated to: {theme}"
```

---

## 8. Dynamic Prompts from State & Context 🎭

Use `@dynamic_prompt` to customize system instructions using runtime session data:

```python
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class AppContext(TypedDict):
    customer_tier: str
    user_name: str

@dynamic_prompt
def personalize_prompt(request: ModelRequest) -> str:
    context = request.runtime.context
    tier = context.get("customer_tier", "Standard")
    name = context.get("user_name", "Valued Customer")
    return f"You are a premium concierge assisting {name} ({tier} Member). Be courteous and prompt."

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[personalize_prompt],
    context_schema=AppContext,
)
```

---

## 9. Lifecycle Middleware Hooks for Memory 🪝

- **`@before_model`**: Runs immediately before the LLM is called. Ideal for dynamically trimming long histories or injecting context blocks.
- **`@after_model`**: Runs immediately after the model response is received. Ideal for monitoring token counts, caching outputs, or stripping sensitive tokens.

---

## 10. Short-Term vs. Long-Term Memory ⚖️

| Feature | Short-Term Memory 💾 | Long-Term Memory 🏛️ |
| :--- | :--- | :--- |
| **Scope** | Single conversation thread / session | Across multiple sessions, threads, and days |
| **Primary Purpose** | Maintains context of the ongoing dialogue | Recalls user preferences, historical facts, and profiles |
| **Lifecycle** | Cleared when thread ends / archived | Stored permanently in vector DB / user store |
| **Implementation** | `checkpointer` (`InMemorySaver`, `PostgresSaver`) | `Store` interface (`InMemoryStore`, vector stores) |
| **Example** | *"As I mentioned two messages ago..."* | *"Welcome back Alice! Using your preferred dark theme."* |

---

## 11. Interview Revision Notes & Checklist 📋

- [ ] **Thread-Level Isolation**: Short-term memory is scoped strictly to a `thread_id`.
- [ ] **Checkpointer Requirement**: Agents must configure a checkpointer to retain state between `.invoke()` calls.
- [ ] **Stateless Models**: Language models never store state internally; memory is re-injected as formatted message history on every turn.
- [ ] **Context Window Bloat**: Mitigate token bloat using message trimming, deletion (`RemoveMessage`), or `SummarizationMiddleware`.
- [ ] **Tool State Access**: Tools read state cleanly using `runtime: ToolRuntime`.
- [ ] **Persistence Choices**: Use `InMemorySaver` for local learning and tests; use `PostgresSaver` for resilient production deployments.

---

## 12. Official References 📚

- [LangChain Short-Term Memory Guide](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangChain Long-Term Memory Guide](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [LangGraph Persistence and Checkpointers](https://docs.langchain.com/oss/python/langgraph/persistence)
