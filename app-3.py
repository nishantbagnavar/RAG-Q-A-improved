import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Streamlit UI Setup
st.set_page_config(page_title="Conversational RAG Chatbot", layout="wide")
st.title("📚 Conversational RAG Chatbot")
st.markdown("Upload PDFs and ask questions based on their content.")

# Sidebar API Key Input
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key:", type="password")

# Ensure API key is provided
if not api_key:
    st.warning("⚠️ Please enter your Groq API Key.")
    st.stop()

# Initialize LLM (Groq's Gemma-2)
llm = ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")

# Session ID for chat history
session_id = st.text_input("🆔 Session ID:", value="default_session")

# Session-based storage
if "store" not in st.session_state:
    st.session_state.store = {}

# File Upload Section
uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

# Process Uploaded Files
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = f"./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        # Load PDF
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    # Split Text for Better Indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create Vector Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever()

    # Contextualization Prompt (History-Aware)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Reformulate user queries into standalone questions based on chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # History-Aware Retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Optimized Answering Prompt
    system_prompt = """
    You are a knowledge-based AI assistant. Follow these guidelines:

    1️⃣ **Use ONLY the provided context** – Do not generate answers outside of it.  
    2️⃣ **Be concise  – Provide structured, easy-to-read responses.  
    3️⃣ **If the context lacks an answer, state: 'I don't have enough information to answer that.'**  
    4️⃣ **Maintain clarity & accuracy** – Ensure logical, fact-based replies.  

    Context:  
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Answering Chain (RAG)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Session History Management
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    # Chat-Enabled RAG
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Chat Interface
    user_input = st.text_input("💬 Ask a Question:")

    if user_input:
        session_history = get_session_history(session_id)

        with st.spinner("⏳ Generating response..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"session_id": session_id},
            )

        # Store message history
        session_history.add_user_message(user_input)
        session_history.add_ai_message(response["answer"])

        # Display response
        st.success(response["answer"])

        # Debugging Info (Optional)
        with st.expander("📜 Chat History"):
            st.write(session_history.messages)
else:
    st.info("📥 Upload a PDF to begin chatting!")
