# LangChain Structured Output 📐

**Structured Output** forces language models and agents to return data conforming strictly to a predefined schema (e.g. Pydantic models, Dataclasses, TypedDicts, or JSON Schemas) instead of unstructured natural language text.

The validated result is captured, typed, and returned in the `'structured_response'` key of the execution result.

---

## 📑 Contents

1. [Overview & Core Value](#1-overview--core-value)
2. [Standalone Model vs Agent Structured Output](#2-standalone-model-vs-agent-structured-output)
3. [Strategies: `ProviderStrategy` vs `ToolStrategy`](#3-strategies-providerstrategy-vs-toolstrategy)
   - [Provider Strategy (Native Schema Enforcement)](#a-providerstrategy-native-provider-support)
   - [Tool Strategy (Function Calling Schema Enforcement)](#b-toolstrategy-tool-calling-fallback)
4. [Supported Schema Types](#4-supported-schema-types)
   - [Pydantic Models](#1-pydantic-basemodel)
   - [Dataclasses](#2-dataclasses)
   - [TypedDict](#3-typeddict)
   - [JSON Schema Dictionaries](#4-json-schema-dictionaries)
   - [Union Types (Multiple Possible Schemas)](#5-union-types)
5. [Customizing Tool Messages (`tool_message_content`)](#5-customizing-tool-messages-tool_message_content)
6. [Intelligent Error Handling & Auto-Retries](#6-intelligent-error-handling--auto-retries)
   - [Schema Validation Errors](#a-schema-validation-errors)
   - [Multiple Structured Outputs Error](#b-multiple-structured-outputs-error)
   - [`handle_errors` Configuration Options](#c-handle_errors-configuration-options)
7. [Quick Revision Reference](#7-quick-revision-reference)

---

## 1. Overview & Core Value

Without structured output, developers must write fragile regex or manual JSON parsing logic. With structured output:
- ✅ Outputs are **type-safe** and validated immediately.
- ✅ Missing or malformed fields trigger **automatic model re-prompts**.
- ✅ Structured data can be passed directly to downstream databases or frontend APIs.

```text
Prompt ──► Model / Agent ──► Schema Validation ──► result["structured_response"]
                                    │
                                (If Invalid)
                                    ▼
                         Auto Re-prompt with Error
```

---

## 2. Standalone Model vs Agent Structured Output

### A. Standalone Model (`with_structured_output`)
For single prompt-response extraction tasks without an agent loop:

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class Contact(BaseModel):
    name: str
    email: str

model = init_chat_model("google_genai:gemini-2.5-flash")
structured_model = model.with_structured_output(Contact)

response = structured_model.invoke("My name is Alice and email is alice@example.com")
print(response)  # Contact(name='Alice', email='alice@example.com')
```

### B. Agent (`create_agent` with `response_format`)
For multi-turn agents that can use tools and return structured conclusions:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[search_tool],
    response_format=Contact
)

result = agent.invoke({"messages": [{"role": "user", "content": "Extract contact for Alice..."}]})
print(result["structured_response"])
```

---

## 3. Strategies: `ProviderStrategy` vs `ToolStrategy`

LangChain supports two execution strategies under the hood:

| Strategy | Mechanism | Best Used When |
| :--- | :--- | :--- |
| **`ProviderStrategy`** | Provider-native JSON / grammar enforcement (OpenAI, Gemini, Claude, Groq) | Model supports native structured output |
| **`ToolStrategy`** | Generates a synthetic tool call and validates arguments | Model lacks native mode but supports tool calling |

> 💡 **Auto-Selection**: When you pass a schema directly (e.g. `response_format=ContactInfo`), LangChain automatically picks `ProviderStrategy` if supported, otherwise falling back to `ToolStrategy`.

### A. `ProviderStrategy` (Native Provider Support)
```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ProviderStrategy(schema=Contact, strict=True)
)
```

### B. `ToolStrategy` (Tool Calling Fallback)
```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="groq:qwen/qwen3.8-27b",
    response_format=ToolStrategy(schema=Contact, handle_errors=True)
)
```

---

## 4. Supported Schema Types

### 1. Pydantic `BaseModel`
Returns a validated Pydantic model instance:
```python
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the reviewed product")
    rating: int = Field(description="Score between 1 and 5", ge=1, le=5)
    summary: str = Field(description="Brief 1-sentence summary")
```

### 2. Dataclasses
Returns a standard Python dictionary:
```python
from dataclasses import dataclass

@dataclass
class UserProfile:
    username: str
    is_active: bool
```

### 3. TypedDict
Returns a type-annotated dictionary:
```python
from typing_extensions import TypedDict

class MovieDetails(TypedDict):
    title: str
    year: int
    genre: str
```

### 4. JSON Schema Dictionaries
Must include top-level `title`, `type`, and `properties`:
```python
contact_schema = {
    "title": "Contact",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"}
    },
    "required": ["name", "email"]
}
# Wrap explicitly in strategy
response_format = ToolStrategy(contact_schema)
```

### 5. `Union` Types
Allows the agent to choose between multiple output schemas dynamically:
```python
from typing import Union

class SupportTicket(BaseModel):
    ticket_id: str
    issue: str

class GeneralInquiry(BaseModel):
    question: str

# Model selects either SupportTicket or GeneralInquiry based on user query
response_format = ToolStrategy(Union[SupportTicket, GeneralInquiry])
```

---

## 5. Customizing Tool Messages (`tool_message_content`)

When using `ToolStrategy`, customize the acknowledgement message recorded in conversation history:

```python
agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    response_format=ToolStrategy(
        schema=ProductReview,
        tool_message_content="Review analyzed and stored successfully."
    )
)
```

---

## 6. Intelligent Error Handling & Auto-Retries

When models make formatting or validation mistakes with `ToolStrategy`, LangChain automatically intercepts the error and gives the model feedback to self-correct.

### A. Schema Validation Errors
If the model produces a value outside constraints (e.g. `rating=10` when `le=5`), LangChain injects a corrective `ToolMessage`:
```text
Error: Failed to parse structured output for tool 'ProductReview': 
1 validation error: Input should be less than or equal to 5.
Please fix your mistakes.
```

### B. Multiple Structured Outputs Error
If the model tries calling two mutually exclusive schemas at once:
```text
Error: Model incorrectly returned multiple structured responses when only one is expected.
Please fix your mistakes.
```

### C. `handle_errors` Configuration Options

| Option | Syntax | Behavior |
| :--- | :--- | :--- |
| **Default (`True`)** | `handle_errors=True` | Catches all validation errors and prompts model to retry |
| **Custom String** | `handle_errors="Provide 1-5 rating only."` | Returns a fixed string error prompt to the model |
| **Specific Exception** | `handle_errors=ValueError` | Only retries on `ValueError`, propagates others |
| **Custom Function** | `handle_errors=custom_fn` | Dynamic function returning tailored corrective instructions |
| **Disabled (`False`)** | `handle_errors=False` | Disables retry loop and raises exceptions immediately |

---

## 7. Quick Revision Reference

| Feature | Key Summary |
| :--- | :--- |
| **`response_format`** | Parameter in `create_agent` to enforce structured output |
| **`structured_response`** | Result key containing the final parsed and validated output |
| **`ProviderStrategy`** | Native model provider structured output (strictest & fastest) |
| **`ToolStrategy`** | Tool-calling based structured extraction with automated retry loop |
| **Pydantic Models** | Most recommended schema format for validation and type-hinting |
| **`Union[SchemaA, SchemaB]`** | Enables dynamic schema selection based on conversation context |
| **`handle_errors`** | Configures auto-retry behavior on validation failures |

---

### 🔑 Core Takeaway

> **Structured Output turns unpredictable natural language generation into reliable, schema-validated data objects that can be directly consumed by production software and databases.**
