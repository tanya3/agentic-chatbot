# app/agents/agentic_rag_agent.py

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
from configs.app_config import (LOADER_MAPPING, MODEL_NAME)
from configs.agent_config import (RAG_PROMPT_TEMPLATE)

def handle_uploaded_doc_query(state: dict) -> Dict[str, Any]:
    logging.info("Agentic RAG Agent is running (will read from Session State)...")
    query: Optional[str] = state.get("query")

    processed_info = st.session_state.get("processed_upload_info")

    if not query:
        logging.error("Agentic RAG: No query found in state!")
        return {"answer": "Query not found.", "source": "Agentic RAG (Error)"}

    if not processed_info or not processed_info.get("content") or not processed_info.get("filename"):
        logging.warning("Agentic RAG: No processed and active document found in session.")
        return {"answer": "Please upload a document before asking a question.", "source": "Agentic RAG (Error: No Document)"}

    uploaded_file_data: bytes = processed_info["content"]
    uploaded_file_name: str = processed_info["filename"]
    source_info = f"Active Document ({uploaded_file_name})"

    temp_file_path = None
    try:
        file_suffix = Path(uploaded_file_name).suffix.lower() or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
             temp_file.write(uploaded_file_data)
             temp_file_path = temp_file.name
             logging.info(f"Document from session saved temporarily to: {temp_file_path}")

        loader_class = LOADER_MAPPING.get(file_suffix)
        if not loader_class:
            logging.error(f"Unsupported file type: {file_suffix}")
            return {"answer": f"Sorry, file type '{file_suffix}' is not supported.", "source": source_info + " (Error)"}
        loader = loader_class(temp_file_path)
        documents = loader.load()
        logging.info(f"{len(documents)} document chunks loaded ({uploaded_file_name}).")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Document split into {len(chunks)} chunks.")
        if not chunks:
             logging.warning("No meaningful text chunks could be extracted from the document.")
             return {"answer": "No meaningful content could be extracted from the active document.", "source": source_info}

        embeddings = SentenceTransformerEmbeddings(model_name=MODEL_NAME)

        logging.info("Creating in-memory Chroma vector store...")
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        logging.info("Vector store is ready.")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

        llm = get_llm(temperature=0.2)
        if not llm:
             logging.error("LLM instance could not be initialized!")
             return {"answer": "Answer generation engine (LLM) could not be started.", "source": "Agentic RAG (Error)"}
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        )
        logging.info("RAG chain is being invoked with LLM (in context of active document)...")
        answer = rag_chain.invoke(query)
        logging.info("Response received from LLM.")

        relevant_docs = retriever.get_relevant_documents(query)
        context_for_display = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        return { "answer": answer, "context": context_for_display, "source": source_info }

    except Exception as e:
        logging.error(f"Error during Agentic RAG (active document): {e}", exc_info=True)
        return {"answer": f"Sorry, an error occurred while processing the active document: {type(e).__name__}", "source": source_info + " (Error)"}

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logging.info(f"Temporary file deleted: {temp_file_path}")
            except Exception as e_clean:
                logging.error(f"Error while deleting temporary file: {e_clean}", exc_info=True)
