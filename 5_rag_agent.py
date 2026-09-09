
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# SESSION STATE
# ============================================================

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_document(path):

    # Load PDF documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents=docs)

    # ========================================================
    # EMBEDDINGS + VECTOR DATABASE
    # ========================================================

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_store = vector_db

    # ========================================================
    # LLM
    # ========================================================

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    # ========================================================
    # RETRIEVAL TOOL
    # ========================================================

    @tool
    def retrieve_context(query: str):
        """Retrieve documents relevant to a query from the knowledge base."""

        context = ""

        docs = vector_db.similarity_search(
            query=query,
            k=3
        )

        for doc in docs:
            context += doc.page_content + "\n\n"

        return context

    # ========================================================
    # AGENT
    # ========================================================

    system_prompt = """
    You are a helpful assistant that answers questions using
    retrieved context.

    My knowledge base consists of the details from the uploaded
    documents.

    ALWAYS use the `retrieve_context` tool for questions requiring
    external knowledge from the uploaded documents.

    Answer the user's question based only on the retrieved context.
    If the answer cannot be found in the uploaded documents, say
    that the information is not available in the uploaded documents.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    # Store agent in session
    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# ============================================================
# PDF UPLOAD UI
# ============================================================

if not st.session_state.document_uploaded:

    st.title("📚 RAG PDF Q&A Chatbot")

    st.write(
        "Upload one or more PDF documents and ask questions "
        "about their content."
    )

    uploaded = st.file_uploader(
        label="Select PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:

        with st.spinner("Processing documents..."):

            # Relative path
            # Creates doc_files inside the project directory
            path = os.path.join(os.getcwd(), "doc_files")

            # Create folder if it doesn't exist
            os.makedirs(path, exist_ok=True)

            # Save uploaded PDFs
            for file in uploaded:

                file_path = os.path.join(
                    path,
                    file.name
                )

                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            # Process documents
            process_document(path)

            st.success("Documents processed successfully!")

            st.rerun()


# ============================================================
# CHAT UI
# ============================================================

if (
    st.session_state.document_uploaded
    and st.session_state.agent
):

    st.title("💬 Ask Questions")

    # Display previous messages
    for message in st.session_state.messages:

        role = message.get("role")
        content = message.get("content")

        st.chat_message(role).markdown(content)

    # Chat input
    query = st.chat_input(
        "Ask anything related to uploaded documents..."
    )

    if query:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # Display user message
        st.chat_message("user").markdown(query)

        # Invoke agent
        response = st.session_state.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            },
            {
                "configurable": {
                    "thread_id": "1"
                }
            }
        )

        # Get final answer
        answer = response["messages"][-1].content

        # Display answer
        st.chat_message("assistant").markdown(answer)

        # Store answer
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

