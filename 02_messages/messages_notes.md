# LangChain Messages 💬

LangChain Messages are the fundamental units of context exchanged between users, models, and tools.

A message contains:
- 🎭 **Role**: Who sent it (`system`, `human`, `ai`, `tool`)
- 📦 **Content**: Text, images, audio, files, etc.
- 🏷️ **Metadata**: Message IDs, token usage, provider response metadata, etc.

---

## 📑 Contents

1. [Basic Usage](#1-basic-usage)
2. [Message Types](#2-message-types)
3. [Message Content Formats](#3-message-content-formats)
4. [AIMessage Attributes](#4-aimessage-attributes)
5. [ToolMessage](#5-toolmessage)
6. [Multimodal Messages](#6-multimodal-messages)
7. [Content Blocks](#7-content-blocks)
8. [Token Usage](#8-token-usage)
9. [Serialization & Deserialization](#9-serialization--deserialization)
10. [Quick Revision](#10-quick-revision)

---

## 1. Basic Usage

Import the standard message classes directly from `langchain_core.messages`:

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("google_genai:gemini-2.5-flash")

# Create message list and pass to model
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is RAG?")
]

response = model.invoke(messages)
print(response.content)
```

LangChain provides a unified message interface across all model providers.

---

## 2. Message Types

| Message Class | Role | Purpose |
| :--- | :--- | :--- |
| **`SystemMessage`** | `system` | Controls model behavior, persona, and instructions |
| **`HumanMessage`** | `user` | Represents user input / prompts |
| **`AIMessage`** | `assistant` | Represents model responses |
| **`ToolMessage`** | `tool` | Contains execution results returned by tools |

### Examples:

#### A. `SystemMessage`
Defines the model's persona, role, and instructions:
```python
system = SystemMessage("You are a helpful Python assistant.")
```

#### B. `HumanMessage`
Represents user input:
```python
human = HumanMessage("Explain REST API in simple terms.")
```

#### C. `AIMessage`
Represents the model's response:
```python
response = model.invoke("Explain AI")
print(type(response))  # <class 'langchain_core.messages.ai.AIMessage'>
```

#### D. `ToolMessage`
Contains the result returned by a tool after a model tool call:
```python
from langchain_core.messages import ToolMessage

tool_message = ToolMessage(
    content="Sunny, 72°F",
    tool_call_id="call_123"
)
```

---

## 3. Message Content Formats

Message content can be represented in multiple formats:

### 1. Simple String
```python
HumanMessage("What is machine learning?")
```

### 2. Message Objects List
```python
messages = [
    SystemMessage("You are a teacher."),
    HumanMessage("Explain RAG.")
]
response = model.invoke(messages)
```

### 3. Dictionary Format (OpenAI Schema)
```python
messages = [
    {"role": "system", "content": "You are a teacher."},
    {"role": "user", "content": "Explain RAG."}
]
response = model.invoke(messages)
```

---

## 4. AIMessage Attributes

`AIMessage` represents the output returned by a model and contains rich metadata:

```python
response = model.invoke("What is LangChain?")
print(response.content)
```

### 🔍 Key Attributes:

| Attribute | Description | Example Output |
| :--- | :--- | :--- |
| `response.content` | Raw text response | `"LangChain is a framework..."` |
| `response.content_blocks` | Standardized multimodal / reasoning blocks | `[{'type': 'text', 'text': '...'}]` |
| `response.tool_calls` | List of tool calls requested by the model | `[{'name': 'search', 'args': {...}, 'id': '...'}]` |
| `response.usage_metadata` | Input, output & total token counts | `{'input_tokens': 12, 'output_tokens': 45}` |
| `response.response_metadata` | Provider response information & raw headers | `{'finish_reason': 'stop', 'model_name': '...'}` |
| `response.id` | Unique ID of the response | `'run-0192a3b4-...'` |

---

## 5. ToolMessage 🔨

When using tools, the message flow follows an iterative round-trip:

```text
HumanMessage ──► AIMessage (Tool Call) ──► Tool Execution ──► ToolMessage ──► Final AIMessage
```

### Code Example:

```python
from langchain_core.messages import AIMessage, ToolMessage

ai_message = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "Mumbai"},
            "id": "call_123"
        }
    ]
)

# The tool_call_id MUST match the ID from the corresponding AIMessage tool call
tool_message = ToolMessage(
    content="Sunny, 30°C",
    tool_call_id="call_123"
)
```

---

## 6. Multimodal Messages 🖼️

Messages can contain different types of content:
- 📝 Text
- 🖼️ Images
- 🔊 Audio
- 🎥 Video
- 📄 Files / PDFs

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image", "url": "https://example.com/image.jpg"}
    ]
}
```

> **Note**: Supported formats depend on the provider and model (e.g. Gemini 2.5, GPT-4o, Claude 3.7).

---

## 7. Content Blocks

LangChain standardizes responses into **content blocks** across all providers for consistency:

```python
response.content_blocks
```

### Common Block Types:

| Block Type | Purpose |
| :--- | :--- |
| `text` | Text content |
| `reasoning` | Model internal thinking/reasoning chain |
| `image` | Image data |
| `audio` | Audio data |
| `video` | Video data |
| `file` | Files / PDFs |
| `tool_call` | Tool invocation request |
| `server_tool_call` | Server-side provider tool execution |

---

## 8. Token Usage 📊

`AIMessage` provides token usage via `usage_metadata`:

```python
response = model.invoke("Explain RAG")
print(response.usage_metadata)
```

### Example Output:
```json
{
  "input_tokens": 8,
  "output_tokens": 304,
  "total_tokens": 312
}
```

### 💡 Useful for:
- Cost tracking & optimization
- Token usage monitoring
- Latency and throughput analysis

---

## 9. Serialization & Deserialization 💾

Messages can be converted into plain Python dictionaries for database storage and restored later:

```python
from langchain_core.load import dumpd, load

# 1. Serialize message to dictionary
serialized = dumpd(response)

# 2. Restore message back to LangChain object
restored = load(serialized)
```

> ⚠️ **Security Warning**: Do not call `load()` on untrusted data because deserialization can instantiate Python objects and trigger unintended side effects.

---

## 10. Quick Revision

| Concept | Remember |
| :--- | :--- |
| **`SystemMessage`** | Sets model behavior & instructions |
| **`HumanMessage`** | User input |
| **`AIMessage`** | Model output |
| **`ToolMessage`** | Tool execution result (`tool_call_id` required) |
| **`content`** | Raw message text/data |
| **`content_blocks`** | Standardized multimodal/structural blocks |
| **`tool_calls`** | Model-generated tool call requests |
| **`usage_metadata`** | Token usage metrics |
| **`response_metadata`** | Provider response details |
| **`dumpd()`** | Serialize message to dictionary |
| **`load()`** | Deserialize dictionary back to message |

---

### 🔑 Core Message Flow

```text
User 
 │ (HumanMessage)
 ▼
AI Model 
 │ (AIMessage with tool_calls)
 ▼
Tool Execution ───► ToolMessage ───► AI Model ───► Final AIMessage ───► User
```

### 📌 Key Takeaway
> **Messages are the standardized communication language between users, AI models, and tools across all providers in LangChain.**
