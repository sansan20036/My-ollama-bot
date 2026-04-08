# My Ollama Bot

A local-first full-stack RAG system for private document intelligence.

---

## Overview

**My Ollama Bot** is a full-stack AI application that combines a modern React frontend with a local LLM backend to support:

- document upload
- retrieval-augmented generation (RAG)
- streaming chat responses
- multi-session conversation management
- local/private deployment

This project is designed for scenarios where **data privacy**, **internal document handling**, and **local AI inference** matter.  
Instead of sending files to external cloud-based AI services, the system is intended to run in a local environment with Ollama and a vector database.

---

## Project Goals

This project was built to explore how a local large language model can be integrated into a practical internal knowledge assistant.

The core objectives are:

- build a complete **frontend + backend + AI** workflow
- support **private document question answering**
- provide a usable **chat-based interface**
- connect **file upload**, **retrieval**, and **response generation**
- improve usability through **streaming output** and **session management**

---

## Key Features

- **Local-first AI assistant** powered by Ollama
- **RAG workflow** for document-based question answering
- **File upload and management**
- **Document viewer / preview support**
- **Streaming chat responses**
- **Multi-session chat interface**
- **Dynamic model loading**
- **Backend status detection**
- **One-click system reset**
- **Interactive full-stack UI**

---

## System Architecture

### Frontend
- React
- Vite
- Axios
- Framer Motion
- React Markdown
- React Syntax Highlighter
- Recharts
- KaTeX

### Backend
- FastAPI
- Ollama
- LangChain
- ChromaDB

### Core Workflow

1. User uploads one or more files
2. Backend processes the uploaded documents
3. Text content is split and stored in a vector database
4. User asks a question in the chat interface
5. Relevant document chunks are retrieved
6. Ollama generates an answer based on retrieved context
7. Frontend displays the response in streaming mode

---

## Tech Stack

### Frontend
- React 18
- Vite
- JavaScript
- Axios
- Framer Motion
- React Markdown
- React Syntax Highlighter
- Recharts
- KaTeX
- dnd-kit

### Backend / AI
- Python
- FastAPI
- Ollama
- LangChain
- ChromaDB

---

## Current Functionality

This project currently includes the following capabilities:

- Chat interface with conversation history
- Session creation / switching / deletion
- File upload workflow
- Uploaded file list display
- File deletion support
- File preview / viewing
- Streaming AI responses
- Dynamic available-model loading
- Backend reboot detection
- Local state persistence
- System reset function

---

## Why This Project Matters

This is not just a simple chat UI demo.

It reflects multiple practical software engineering concerns:

- **full-stack integration** between frontend and backend
- **local LLM orchestration**
- **retrieval-based AI system design**
- **stateful chat session management**
- **file workflow design**
- **real-world usability improvements**

The project demonstrates the ability to build a system that is closer to an internal AI tool than a toy prototype.

---

## Use Cases

This system can be adapted for use in scenarios such as:

- internal knowledge assistants
- private document Q&A
- technical documentation search
- research note retrieval
- local enterprise AI tools
- domain-specific assistant systems

---

## Screenshots

> Add screenshots here after uploading project images.

Recommended screenshots:

1. Main chat interface
2. File upload panel
3. Uploaded file list
4. File viewer / preview page
5. Streaming response example
6. Model selection interface

Example markdown:

```md
## Screenshots

### Main Interface
![Main Interface](./screenshots/main-interface.png)

### File Upload
![File Upload](./screenshots/file-upload.png)

### File Viewer
![File Viewer](./screenshots/file-viewer.png)
