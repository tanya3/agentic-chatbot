# app/ui/streamlit_app.py

import sys
import hashlib
import streamlit as st
from pathlib import Path
import logging
import time
from langchain_core.messages import HumanMessage, AIMessage

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from app.graph import graph_app
from app.core.llm import get_llm

# Phrases the RAG prompts (configs/agent_config.py: PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE)
# are instructed to emit verbatim when retrieved context doesn't answer the question.
# Used to tell the user retrieval succeeded but was judged irrelevant, rather than
# leaving "5 chunks retrieved" next to "not found" looking contradictory.
_NOT_FOUND_PHRASES = (
    "was not found in the provided documents",
    "was not found in the active document",
)

# RAG context chunks come straight from the (Turkish) source corpus, unlike the
# synthesized answer which agents already translate to English. Translations are
# cached per-session, keyed by content hash, since Streamlit reruns the whole
# script on every interaction and the toggle can be flipped repeatedly.
_CONTEXT_TRANSLATION_CACHE_KEY = "_rag_context_translations"


def _is_not_found_answer(answer: str) -> bool:
    if not answer:
        return False
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in _NOT_FOUND_PHRASES)


def _translate_context_to_english(context: str) -> str:
    cache = st.session_state.setdefault(_CONTEXT_TRANSLATION_CACHE_KEY, {})
    cache_key = hashlib.sha256(context.encode("utf-8")).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    llm = get_llm(temperature=0)
    if llm is None:
        return "⚠️ Translation unavailable (ANTHROPIC_API_KEY not configured)."

    try:
        response = llm.invoke([
            HumanMessage(content=(
                "Translate the following retrieved document excerpt(s) into natural, "
                "fluent English. Return only the translation, with no commentary or "
                "preamble:\n\n" + context
            ))
        ])
        translated = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        logging.error(f"Error translating RAG context: {e}", exc_info=True)
        return "⚠️ Translation failed. Showing original text below.\n\n" + context

    cache[cache_key] = translated
    return translated


st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("🤖 Agentic AI Chatbot")
st.markdown("""
I can answer your questions about the Official Gazette, current topics, or travel planning.
**You can also upload a document below to ask questions specific to that document.**
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_upload_info" not in st.session_state:
    st.session_state.processed_upload_info = None
if "new_upload_triggered" not in st.session_state:
    st.session_state.new_upload_triggered = False

def handle_file_upload():
    uploaded_file = st.session_state.get("rag_file_uploader")
    if uploaded_file is not None:
        logging.info(f"on_change: New file detected - {uploaded_file.name}")
        try:
            content = uploaded_file.getvalue()
            st.session_state.processed_upload_info = {
                "filename": uploaded_file.name,
                "content": content,
                "type": uploaded_file.type
            }
            st.session_state.new_upload_triggered = True
        except Exception as e:
            logging.error(f"Error reading file content: {e}", exc_info=True)
            st.error(f"An error occurred while reading the file '{uploaded_file.name}'.", icon="⚠️")
            st.session_state.processed_upload_info = None
            st.session_state.new_upload_triggered = False
    else:
        logging.info("on_change: File cleared from widget. Active document context preserved.")
        st.session_state.new_upload_triggered = False 

with st.container(border=True):
    st.subheader("📄 Upload & Manage Document (Agentic RAG)") 
    st.file_uploader(
        "Select a document you want to analyze and ask questions about (PDF, TXT, DOCX etc.)",
        type=["pdf", "txt", "md", "docx"],
        key="rag_file_uploader",
        on_change=handle_file_upload
    )

    active_doc_info = st.session_state.get("processed_upload_info")
    if active_doc_info:
        col1_info, col2_clear = st.columns([4, 1]) 
        with col1_info:
            st.info(f"Active Document Context: **{active_doc_info['filename']}**", icon="ℹ️")
        with col2_clear:
            if st.button("❌ Remove Active Document", key="clear_active_doc_button", help="Only removes the current document context, does not delete chat history."):
                logging.info("User cleared active document context.")
                st.session_state.processed_upload_info = None
                st.session_state.new_upload_triggered = False
                st.success("Active document context removed.", icon="🗑️")
                time.sleep(1)
                st.rerun() 

st.divider()

with st.container(border=False):
    st.subheader("💡 Example Questions:")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("- How are membership fee payments calculated for the entities, institutions, and affiliated partnerships covered under KOSGEB and State Economic Enterprises?")
        st.markdown("- Can you tell me about the appointments made to policy councils under Article 21 of the Presidential Decree?")
    with cols[1]:
        st.markdown("- I want to go from Istanbul to Paris tomorrow and stay for 3 days. My budget is 2000 Euros.")
    with cols[2]:
        st.markdown("- Can you tell me about Turkey's current inflation rate?")
st.divider()

_, clear_col = st.columns([5, 1])
with clear_col:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.processed_upload_info = None
        st.session_state.new_upload_triggered = False
        st.success("Chat history and active document information cleared!", icon="🗑️")
        time.sleep(1)
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            col1_hist, col2_hist = st.columns([4, 1])
            with col1_hist:
                st.caption(f"Source: {message.get('source', 'Unknown')}")
            if message.get("response_time"):
                with col2_hist:
                    st.caption(f"⏱️ {message.get('response_time'):.2f}s")
            if message.get("pdf_path"):
                 try:
                     if Path(message["pdf_path"]).is_file():
                         with open(message["pdf_path"], "rb") as fp_hist:
                             st.download_button(
                                 label="📄 Download Plan (PDF)", data=fp_hist,
                                 file_name=Path(message["pdf_path"]).name, mime="application/pdf",
                                 key=f"pdf_dl_hist_{message.get('source')}_{len(st.session_state.chat_history)}_{message.get('response_time')}"
                             )
                 except Exception as dl_err:
                     logging.warning(f"Error downloading history PDF: {dl_err}")
            if message.get("context"):
                with st.expander("🔍 Context Used (RAG)"):
                    if _is_not_found_answer(message["content"]):
                        st.caption("ℹ️ Context was retrieved above, but the model judged it didn't sufficiently answer your question.")
                    context_key = f"ctx_hist_{message.get('source')}_{len(st.session_state.chat_history)}_{message.get('response_time')}"
                    show_english = st.toggle("🌐 Show in English", key=f"{context_key}_lang")
                    display_context = _translate_context_to_english(message["context"]) if show_english else message["context"]
                    st.text_area("", display_context, height=150, disabled=True, key=context_key)

if user_input := st.chat_input("Type your question here..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Preparing response..."):
        try:
            start_time = time.time()

            formatted_history = []
            for msg in st.session_state.chat_history[:-1]:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(AIMessage(content=msg["content"]))

            graph_input = {
                "query": user_input,
                "chat_history": formatted_history 
            }
            if st.session_state.get("new_upload_triggered"):
                logging.info("New upload flag is True. Directing to Agentic RAG.")
                graph_input["route_directly_to_agentic_rag"] = True
                st.session_state.new_upload_triggered = False
                logging.info("new_upload_triggered flag set to False.")
            else:
                logging.info("New upload flag is False/None. Supervisor will route.")

            logging.info(f"Calling LangGraph... Input Keys: {list(graph_input.keys())}")
            final_state = graph_app.invoke(graph_input)
            logging.info("LangGraph completed.")
            end_time = time.time()
            response_duration = end_time - start_time

            answer = final_state.get("answer", "A problem occurred, couldn't get an answer.")
            source = final_state.get("source", "Unknown")
            context = final_state.get("context")
            pdf_path = final_state.get("pdf_path")
            logging.info(f"Answer generated. Source: {source}. Duration: {response_duration:.2f}s.")

            assistant_response = {
                "role": "assistant", "content": answer, "source": source,
                "context": context, "pdf_path": pdf_path, "response_time": response_duration
            }
            st.session_state.chat_history.append(assistant_response)

            with st.chat_message("assistant"):
                st.markdown(answer)
                col1_resp, col2_resp = st.columns([4,1])
                with col1_resp:
                    st.caption(f"Source: {source}")
                with col2_resp:
                    st.caption(f"⏱️ {response_duration:.2f}s")
                if pdf_path:
                     try:
                         if Path(pdf_path).is_file():
                             with open(pdf_path, "rb") as fp_resp:
                                 st.download_button(
                                     label="📄 Download Plan (PDF)", data=fp_resp,
                                     file_name=Path(pdf_path).name, mime="application/pdf",
                                     key=f"pdf_dl_resp_{len(st.session_state.chat_history)}"
                                 )
                     except Exception as dl_err_resp:
                         logging.warning(f"Error downloading response PDF: {dl_err_resp}")
                         st.error("Couldn't create PDF download button.", icon="⚠️")
                if context:
                    with st.expander("🔍 Context Used (RAG)"):
                        if _is_not_found_answer(answer):
                            st.caption("ℹ️ Context was retrieved above, but the model judged it didn't sufficiently answer your question.")
                        show_english_resp = st.toggle("🌐 Show in English", key=f"ctx_resp_{len(st.session_state.chat_history)}_lang")
                        display_context = _translate_context_to_english(context) if show_english_resp else context
                        st.text_area("Context", display_context, height=200, disabled=True, key=f"ctx_resp_{len(st.session_state.chat_history)}")

        except Exception as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            error_msg_for_user = f"Sorry, an error occurred and I couldn't process your request.\nError Detail: {type(e).__name__}"
            st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, an error occurred.", "source": "System Error"})
            with st.chat_message("assistant"):
                st.error(error_msg_for_user)

with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown(
        """
        This chatbot is designed to answer your questions about **Official Gazette** content,
        **current events/general information**, and **travel planning**.

        - **Official Gazette Questions:** Answers are generated by scanning relevant documents (RAG).
        - **Travel Planning:** Detailed planning and a map are generated.
        - **Document Querying:** Answers your questions about the content of your uploaded document (Agentic RAG). While a document is active, your questions are evaluated in that document's context first.
        - **Other Questions:** Answered using web search or Wikipedia.
        """
    )
    st.divider()
    st.caption("Oğulcan")
    st.caption("AKCA")