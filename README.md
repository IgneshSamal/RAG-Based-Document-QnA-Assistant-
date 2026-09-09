# 📚 RAG-Based PDF Q&A Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** based PDF Question-Answering chatbot that allows users to upload one or multiple PDF documents and interact with them through a conversational interface.

The application processes uploaded documents, converts their content into vector embeddings, stores them in an in-memory vector database, and uses an AI agent with a retrieval tool to answer user questions based on the uploaded documents.

## 🚀 Live Demo

👉 Live Application: https://rag-based-document-qna-assistant.streamlit.app/

---

## ✨ Features

- 📄 Upload one or multiple PDF documents
- 🔍 Automatic document loading and text extraction
- ✂️ Intelligent text chunking using Recursive Character Text Splitter
- 🧠 Google Gemini Embeddings for semantic representation
- 🗄️ In-memory vector database for similarity search
- 🔎 Top-k relevant document retrieval
- 🤖 Agentic question-answering using LangChain
- 🛠️ Custom retrieval tool integrated with the AI agent
- 💬 Interactive conversational chat interface
- 🧠 Short-term conversational memory using LangGraph
- 🔐 Secure API key management using Streamlit Secrets
- ☁️ Deployed publicly using Streamlit Community Cloud

---

# 🏗️ Project Architecture

The application follows a Retrieval-Augmented Generation architecture combined with an agentic workflow.

```text
                         ┌──────────────────────┐
                         │       User           │
                         └──────────┬───────────┘
                                    │
                                    │ Upload PDF(s)
                                    ▼
                         ┌──────────────────────┐
                         │   Streamlit UI       │
                         │   PDF File Uploader  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ PyPDFDirectoryLoader │
                         │  Document Loading    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ RecursiveCharacter   │
                         │ TextSplitter         │
                         │                      │
                         │ Chunk Size: 1000     │
                         │ Overlap: 200         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Google Gemini        │
                         │ Embeddings           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ InMemoryVectorStore  │
                         │                      │
                         │ Vector Database      │
                         └──────────┬───────────┘
                                    │
                                    │ Similarity Search
                                    │ Top 3 Chunks
                                    ▼
                         ┌──────────────────────┐
                         │ retrieve_context()   │
                         │ Custom LangChain     │
                         │ Retrieval Tool       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   LangChain Agent    │
                         │    create_agent      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Groq LLM        │
                         │    gpt-oss-20b       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Generated Answer     │
                         │        ↓             │
                         │   Streamlit Chat     │
                         └──────────────────────┘
