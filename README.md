# Simple AI Assistant

A command-line AI assistant for the fictional company **Nimbus Technologies**. It chats in the terminal, can call tools, and answers company questions with **RAG (Retrieval-Augmented Generation)** over a small set of local text documents.

It runs on any OpenAI-compatible API. It was built with a local model served by [Ollama](https://ollama.com/) (`qwen2.5:3b` for chat, `nomic-embed-text` for embeddings).

## Features

- **Streaming chat**: responses print token by token in the terminal.
- **Tool calling**: the model can call these tools, and can chain them over several steps (up to 5 per question):
  - `get_current_time`: the current date and time
  - `get_city_weather`: live weather from [wttr.in](https://wttr.in)
  - `get_user_city`: the user's city
  - `search_knowledge_base`: semantic search over company documents
- **RAG over a knowledge base**: the `.txt` files in `knowledge_base/` are split into chunks, embedded, and saved to a JSON vector store. Questions are matched to chunks using cosine similarity.
- **Grounded answers with citations**: company questions are answered only from the retrieved text, with the source cited, e.g. `(leave_policy.txt #3)`.
- **Debug logging**: set `DEBUG=True` to see which tools were called and the search scores.

## Project structure

```
simple_ai_assistant.py   # Main chat loop: streaming + multi-step tool calling
rag.py                   # Chunking, embeddings, vector store and similarity search
code_snippets/tools.py   # Tool definitions and the tool dispatcher
Utilities/utilities.py   # Debug logger
knowledge_base/          # Company documents (.txt) used for RAG
code_snippets/           # Small experiments (streaming, embeddings, search)
```

## Setup

1. **Install and start Ollama**, then pull the models:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   pip install openai python-dotenv numpy requests
   ```

3. **Configure environment variables:** copy `.env.example` to `.env` and adjust if needed:
   ```bash
   cp .env.example .env
   ```

   | Variable | Description |
   |---|---|
   | `BASE_URL` | OpenAI-compatible API URL (Ollama: `http://localhost:11434/v1`) |
   | `API_KEY` | API key (any value works for Ollama) |
   | `MODEL` | Chat model name |
   | `EMBEDDING_MODEL` | Embedding model name |
   | `KB_FOLDER` | Folder containing knowledge base `.txt` files |
   | `VECTOR_STORE_FILE` | Path where embeddings are cached |
   | `DEBUG` | `True` to show tool and search logs |

## Usage

```bash
python simple_ai_assistant.py
```

On the first run the knowledge base is embedded and saved to `VECTOR_STORE_FILE`. Later runs load it from there. Delete that file to rebuild it after you change the documents.

Example questions:

```
You : How many casual leaves do I get?
You : What's the weather in my city?
You : What time is it?
```

Type `exit`, `quit` or `bye` to leave.
