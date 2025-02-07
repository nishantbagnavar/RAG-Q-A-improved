import streamlit as st
import time
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
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

# API Keys
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "2-Q&A_RAG_Chatbot"

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## setting up straemlit app
st.set_page_config(page_title="Conversational Rag")
st.title("Conversational RAG With PDF uploads and Chathistory")
st.write("Uplaod PDFs and chat with their content")

## input the groq api key
api_key=st.sidebar.text_input("Enter your groq api key:",type="password")

## check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model="gemma2-9b-it")

    ## chat interface
    session_id=st.text_input("Session ID",value="default_session")

    ## satefully manages chat history
    if "store" not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)

    ## process uploaded file
    if uploaded_files:
        documents=[]

        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            ##loader
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        ## splitting
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)

        ## vectorstore and embeddings
        vectorstore=FAISS.from_documents(splits,embeddings)

        ## retriever
        retriever=vectorstore.as_retriever()

        ## prompts for history aware retriever
        contextualize_q_system_prompt=(
            """
            Given a chat history and latest user question which might reference 
            context in the chat history,formulate a standalone queistion which
            can be understood without chat history.Do not answer question, just
            reformulate it if needed and otherwise return it as is
            """
            
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")

            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


        ## Answer Question prompt
        system_prompt=(
            """
            You are an assistant for question answering tasks,use the following peices 
            of retrieved context to answer the question. If you dont know the answer,
            that you dont know. use three sentences maximum and keep the answer concise
          
            """
            "\n\n"
            "{context}"
            
        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )


        ## chains
        question_anwer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_anwer_chain)

        ## history
        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


        user_input=st.text_input("Your Question: ")
        if user_input:
            session_history=get_session_history(session_id)

            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={"session_id":session_id}
            )

            st.write(st.session_state.store)
            st.write("Assistant: ",response["answer"])
            st.write("chat_history: ",session_history.messages)

else:
    st.warning("Enter api key")




            

