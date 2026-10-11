# LangChain Models 🧠

Practical notes and examples for working with **LangChain Chat Models**.

LangChain models are the reasoning engines behind standalone applications and agents. They support **generation, tool calling, structured output, multimodality, and reasoning**.

---

## 📑 Contents

1. [Model Initialization](#1-model-initialization)
2. [Providers](#2-providers)
3. [Model Parameters](#3-model-parameters)
4. [Invocation Methods](#4-invocation-methods)
5. [Batch Processing](#5-batch-processing)
6. [Tool Calling](#6-tool-calling)
7. [Structured Output](#7-structured-output)
8. [Advanced Features](#8-advanced-features)
   - [Model Profiles](#model-profiles)
   - [Multimodal](#multimodal)
   - [Reasoning](#reasoning)
   - [Prompt Caching](#prompt-caching)
   - [Server-Side Tool Use](#server-side-tool-use)
   - [Token Usage](#token-usage)
   - [Exception Handling](#exception-handling)
   - [Rate Limiting](#rate-limiting)
   - [Dynamic Model Selection](#dynamic-model-selection)
9. [Quick Revision](#9-quick-revision)

---

## 1. Model Initialization

`init_chat_model()` provides a common interface for initializing chat models from different providers.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "google_genai:<model-name>"
)
```

Examples:

```python
init_chat_model("openai:<model-name>")
init_chat_model("google_genai:<model-name>")
init_chat_model("anthropic:<model-name>")
init_chat_model("groq:<model-name>")
```

---

## 2. Providers

LangChain supports multiple model providers through dedicated integration packages.

| Provider | Package |
|---|---|
| Google Gemini | `langchain-google-genai` |
| OpenAI | `langchain-openai` |
| Anthropic | `langchain-anthropic` |
| Groq | `langchain-groq` |
| Hugging Face | `langchain-huggingface` |
| Ollama | `langchain-ollama` |
| OpenRouter | `langchain-openrouter` |
| AWS Bedrock | `langchain-aws` |

---

## 3. Model Parameters

Common parameters:

```python
model = init_chat_model(
    "google_genai:<model-name>",
    temperature=0.7,
    max_tokens=1024,
    timeout=60,
    max_retries=6
)
```

| Parameter | Purpose |
|---|---|
| `temperature` | Controls output randomness |
| `max_tokens` | Limits output length |
| `timeout` | Maximum request time |
| `max_retries` | Retry attempts for temporary failures |

LangChain automatically retries certain network errors, rate limits (`429`), and server errors (`5xx`). Errors such as `401` and `404` are not retried.

---

## 4. Invocation Methods

### `invoke()`

Returns the complete response.

```python
response = model.invoke("What is RAG?")
print(response.content)
```

### `stream()`

Returns the response progressively.

```python
for chunk in model.stream("Explain RAG"):
    print(chunk.text, end="", flush=True)
```

**In short:**

```text
invoke() → Complete response
stream() → Response chunks
```

---

## 5. Batch Processing

### `batch()`

Processes multiple independent requests in parallel.

```python
queries = [
    "What is RAG?",
    "What is LangChain?",
    "What is an AI agent?"
]

responses = model.batch(queries)

for response in responses:
    print(response.content)
```

Results are returned in **input order**.

### `batch_as_completed()`

Returns results as soon as they finish.

```python
for index, response in model.batch_as_completed(queries):
    print(index, response.content)
```

Results may arrive **out of order**. The index identifies the original input.

### `max_concurrency`

Controls the maximum number of parallel requests.

```python
responses = model.batch(
    queries,
    config={"max_concurrency": 3}
)
```

---

## 6. Tool Calling 🔨

Tool calling allows models to request external functions such as APIs, database queries, or other tools.

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"It's sunny in {location}."

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke(
    "What's the weather in Boston?"
)

print(response.tool_calls)
```

### Tool Calling Flow

```text
User → Model → Tool Call → Tool Execution → Result → Model → Response
```

---

## 7. Structured Output 📐

Structured output ensures that model responses follow a defined schema.

### Pydantic

```python
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int
    director: str
    rating: float

structured_model = model.with_structured_output(Movie)

response = structured_model.invoke(
    "Give details about the movie Inception"
)

print(response)
```

Pydantic is useful when you need **validated, predictable, machine-readable output**.

---

# 8. Advanced Features

## Model Profiles

A model profile provides information about the model's capabilities.

```python
model.profile
```

It can contain information such as:

- Maximum input tokens
- Tool calling support
- Structured output support
- Image input
- Reasoning support



---

## Multimodal

Some models can process content beyond text:

- 🖼️ Images
- 🔊 Audio
- 🎥 Video

Multimodal content is represented using **content blocks**.

```python
response = model.invoke("Describe this image")

print(response.content_blocks)
```

---

## Reasoning

Some models support multi-step reasoning and configurable reasoning effort.

```python
response = model.invoke(
    "Solve this complex problem",
    reasoning_effort="high"
)
```

Supported reasoning levels depend on the model and provider.

---

## Prompt Caching

Prompt caching can reduce **latency and cost** when repeatedly processing the same content.

Caching can be handled by:

- Provider-level caching
- Explicit provider controls
- LangChain middleware



---

## Server-Side Tool Use

Some providers can execute tools such as web search or code interpreters on the provider side.

```python
tool = {"type": "web_search"}

model_with_tools = model.bind_tools([tool])

response = model_with_tools.invoke(
    "What is the latest news?"
)
```



---

## Token Usage

Model responses can provide token usage information such as:

- Input tokens
- Output tokens
- Total tokens
- Cached tokens

Useful for **cost and performance monitoring**.

---

## Exception Handling ⚠️

Handle common model errors such as authentication failures, rate limits, timeouts, and network errors.

```python
try:
    response = model.invoke("What is RAG?")
    print(response.content)

except Exception as e:
    print(f"Error: {e}")
```

---

## Rate Limiting 🚦

Rate limiting controls how frequently requests are sent to a provider.

It helps prevent excessive API usage and `429 Too Many Requests` errors.

---

## Dynamic Model Selection 🔄

Models can be selected or configured dynamically at runtime.

Useful when switching models based on:

- Task complexity
- Cost
- Provider availability
- Model capabilities

---

# 9. Quick Revision

| Method / Feature | Purpose |
|---|---|
| `init_chat_model()` | Initialize a model |
| `invoke()` | Get complete response |
| `stream()` | Stream response |
| `batch()` | Process multiple requests |
| `batch_as_completed()` | Get results as they finish |
| `max_concurrency` | Limit parallel requests |
| `bind_tools()` | Enable tool calling |
| `with_structured_output()` | Generate structured responses |
| `profile` | Check model capabilities |
| Multimodal | Process non-text content |
| Reasoning | Configure model reasoning |
| Prompt Caching | Reduce repeated processing cost/latency |
| Token Usage | Monitor token consumption |
| Rate Limiting | Control request frequency |
| Dynamic Selection | Switch models at runtime |

---

## 🔑 Core Concept

```text
                    LangChain Model
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
     Invoke            Stream             Batch
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ↓
                    Model Features
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
   Tool Calling    Structured Output    Multimodal
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ↓
                   Advanced Features
                          │
       Reasoning • Caching • Profiles
       Token Usage • Rate Limiting
       Dynamic Model Selection
```