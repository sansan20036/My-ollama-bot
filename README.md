# My Ollama Bot

A local-first RAG chatbot system built for document-based question answering and internal knowledge retrieval.

## Project Overview

My Ollama Bot is a full-stack AI application that combines a React frontend with a local LLM backend to support file upload, document retrieval, and streaming chat responses.

The system is designed for private, on-premise usage, making it suitable for scenarios where data security and internal document handling are important.  
Instead of sending sensitive files to external cloud AI services, this project keeps inference and retrieval workflows on a local environment.

## Key Features

- Local-first AI assistant powered by Ollama
- RAG (Retrieval-Augmented Generation) workflow for document question answering
- File upload and document management
- File viewing interface for uploaded content
- Streaming chat response experience
- Multi-session chat management
- Backend status detection and reset mechanism
- Model selection support
- Modern interactive frontend built with React + Vite

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
- ChromaDB
- LangChain

### Core Flow
1. User uploads files
2. Backend processes and indexes document content
3. User asks a question
4. System retrieves relevant chunks from the vector database
5. Ollama generates the final answer
6. Frontend displays the response in streaming mode

## Tech Stack

### Frontend
- React 18
- Vite
- JavaScript
- Axios
- Framer Motion
- React Markdown
- Recharts
- KaTeX

### AI / Backend
- Python
- FastAPI
- Ollama
- LangChain
- ChromaDB

## Current Functionality

- Chat interface with session switching
- Upload file workflow
- Uploaded file list display
- File deletion support
- File preview / viewer
- Streaming AI response
- Dynamic model loading
- Backend reboot detection
- One-click system reset

## Project Motivation

This project was built to explore how local LLMs can be integrated into a practical document QA system.

The core idea is to build an AI assistant that is not just a chat UI, but a usable internal tool that supports:

- private deployment
- knowledge retrieval
- full-stack system design
- real interaction between frontend, backend, vector search, and local models

## Why This Project Matters

This is not just a simple chatbot demo.  
It reflects several practical engineering concerns:

- building a complete frontend-backend integration
- managing chat sessions and file workflows
- supporting local model inference
- designing a RAG pipeline for real document usage
- improving usability with streaming output and file preview

## Screenshots

> Add screenshots here

Suggested screenshots:
- Main chat interface
- File upload panel
- File list management
- File viewer
- Streaming response example

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/sansan20036/My-ollama-bot.git
cd My-ollama-bot
