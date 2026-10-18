# Agentic AI Chatbot

An end-to-end Agentic AI Chatbot built with **Python, LangChain, LangGraph, and OpenAI**. This project starts with a graph-based conversational workflow and is designed to evolve into a production-oriented AI application with tool calling, Retrieval-Augmented Generation (RAG), database persistence, human approval, monitoring, and cloud deployment.

## Overview

Traditional chatbots generally receive a question and return an answer. An agentic chatbot can go further by using an orchestrated workflow to decide what action to take, call tools when necessary, retrieve information from documents, and manage multi-step tasks.

This project uses LangGraph to define the chatbot workflow through states, nodes, and edges. OpenAI provides the language model that generates responses.

The project will be developed incrementally, with each component implemented and tested before moving to the next stage.

## Current Implementation

The initial version includes:

- **OpenAI LLM integration:** Uses an OpenAI chat model through LangChain.
- **LangGraph workflow:** Defines and compiles a graph-based execution flow.
- **State management:** Uses a typed state structure to pass messages through the graph.
- **Chatbot node:** Sends the user's message to the language model and returns its response.
- **User input:** Accepts a query through the terminal.
- **Response generation:** Displays the model's answer in the terminal.

### Current Workflow

```text
User Input
    |
    v
HumanMessage
    |
    v
LangGraph START
    |
    v
Chatbot Node
    |
    v
OpenAI LLM
    |
    v
Chatbot Response
    |
    v
LangGraph END
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenAI API | Language model and response generation |
| LangChain | LLM integration and message abstractions |
| LangGraph | Workflow orchestration and graph execution |
| python-dotenv | Loading environment variables |
| Git and GitHub | Version control and source code management |

Additional technologies will be introduced as the project expands.

## Project Structure

```text
Agentic_Chatbot/
│
├── ChatBot_workflow.py   # Chatbot node and LangGraph workflow
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes secrets and local files
├── .env                  # API keys; not committed to Git
└── README.md             # Project documentation
```

## Prerequisites

Before running the application, make sure you have:

- Python 3.11 or a compatible supported Python version.
- Git installed on your system.
- An OpenAI API key with API access and available billing or credits.
- VS Code or another Python development environment.

A ChatGPT subscription does not automatically include OpenAI API credits.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AyushKondalkar/Agentic_AI_ChatBot.git
cd Agentic_AI_ChatBot
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your actual API key.

**Security:** Never commit your `.env` file, publish your API key, or hardcode credentials in Python source files.

### 5. Run the chatbot

```powershell
python ChatBot_workflow.py
```

Enter a question when prompted:

```text
You: What is Retrieval-Augmented Generation?

Chatbot: Retrieval-Augmented Generation (RAG) is an approach
that retrieves relevant external information and uses it to
help a language model generate a grounded response.
```

The response above is an illustrative example; actual output depends on the model.

## How the Code Works

### 1. LLM initialization

The application initializes an OpenAI chat model using `ChatOpenAI`. The model receives messages and generates a response.

### 2. State definition

The `ChatbotState` type defines the data passed between nodes. In the current implementation, it stores a list of messages.

### 3. Node creation

The chatbot node calls the language model using the messages in the current state and returns the generated response.

### 4. Graph construction

LangGraph's `StateGraph` is initialized with the state definition. The chatbot node is registered, and edges connect the start of the graph to the chatbot node and then to the end.

### 5. Graph compilation

The graph is compiled into an executable workflow.

### 6. Execution

The user's input is converted into a `HumanMessage` and passed to the compiled graph through `invoke()`. The final assistant response is printed in the terminal.

## Planned Features and Roadmap

The following capabilities are planned extensions, not features already implemented in the initial version.

### Phase 1: Core chatbot
- [x] OpenAI LLM integration
- [x] LangGraph state definition
- [x] Chatbot node and graph edges
- [x] Terminal-based user input and response
- [ ] Multi-turn conversation handling

### Phase 2: Agentic workflow and tools
- [ ] Create custom tools
- [ ] Integrate tool calling
- [ ] Add conditional routing
- [ ] Implement tool execution and response handling
- [ ] Add error handling and retry logic

### Phase 3: Database and persistence
- [ ] Store conversation history
- [ ] Integrate SQLite for local development
- [ ] Add PostgreSQL for persistent application data
- [ ] Implement conversation and user identifiers
- [ ] Configure LangGraph checkpoint persistence

### Phase 4: Retrieval-Augmented Generation
- [ ] Load PDF and text documents
- [ ] Split documents into chunks
- [ ] Generate embeddings
- [ ] Store embeddings in a vector database
- [ ] Retrieve relevant document chunks
- [ ] Generate grounded answers with source references

### Phase 5: Human-in-the-Loop
- [ ] Add approval checkpoints
- [ ] Pause execution before selected sensitive actions
- [ ] Resume workflows after human approval
- [ ] Handle approval, rejection, and interruption states

### Phase 6: Observability and evaluation
- [ ] Integrate LangSmith tracing
- [ ] Inspect graph execution and tool calls
- [ ] Track errors and latency
- [ ] Evaluate response quality
- [ ] Test retrieval relevance and groundedness

### Phase 7: API and deployment
- [ ] Build a FastAPI application
- [ ] Add request validation and error handling
- [ ] Containerize the application with Docker
- [ ] Deploy an initial version to Render
- [ ] Explore AWS deployment and managed services
- [ ] Configure environment variables and production logging

## Security Considerations

Security will be important as the chatbot becomes more capable.

- Keep API keys in environment variables or a managed secrets service.
- Exclude `.env`, virtual environments, and generated files from Git.
- Validate user input at the API boundary.
- Restrict tool permissions to the minimum required.
- Require human approval for sensitive or irreversible actions.
- Avoid logging secrets or unnecessary personal information.
- Apply authentication, authorization, and rate limits before exposing the API publicly.

## Error Handling and Troubleshooting

**OpenAI authentication errors**

Check that `OPENAI_API_KEY` is set correctly and that the API account has access to the selected model.

**Module not found**

Activate the intended virtual environment and install dependencies from `requirements.txt`.

**Model unavailable**

Verify the selected model name and your API account's model access.

**Git accidentally stages `.env`**

Add `.env` to `.gitignore`. If the API key was exposed, revoke it and generate a replacement. Ignoring a file does not remove it from previous commits.

## Learning Objectives

This project provides practical experience with:

- LLM integration using OpenAI and LangChain.
- Graph-based AI workflow orchestration using LangGraph.
- State, nodes, edges, and conditional execution.
- Tool calling and agent design.
- RAG architecture and vector search.
- Conversation persistence and database integration.
- Human approval workflows.
- LLM tracing and evaluation with LangSmith.
- API development and cloud deployment.

## Future Vision

The long-term goal is to transform the initial terminal chatbot into a modular, observable, and deployable Agentic AI application. The completed system is intended to combine LLM reasoning, tools, retrieval, persistent state, approval workflows, testing, and deployment practices.

The roadmap will be updated as each capability is implemented and verified.

## Author

**Ayush Kondalkar**

GitHub: [AyushKondalkar](https://github.com/AyushKondalkar)

Project Repository: [Agentic_AI_ChatBot](https://github.com/AyushKondalkar/Agentic_AI_ChatBot)
