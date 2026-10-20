
import sys
import hashlib
import json
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# 1. Project path and imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.chat_manager import ChatManager
from memory.checkpointer import (
    get_checkpointer,
    delete_thread_checkpoints,
)
from ChatBot_workflow import graph
from rag.ingestion import ingest_pdf


# --------------------------------------------------
# 2. Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI Chatbot")
st.caption(
    "Powered by Groq, LangGraph, SQLite, Gemini Embeddings, "
    "ChromaDB, and Streamlit"
)


# --------------------------------------------------
# 3. Initialize chat manager and workflow
# --------------------------------------------------

chat_manager = ChatManager()


@st.cache_resource
def create_workflow():
    checkpointer, connection = get_checkpointer()
    workflow = graph.compile(checkpointer=checkpointer)
    return workflow, checkpointer, connection


workflow, checkpointer, connection = create_workflow()


# --------------------------------------------------
# 4. Initialize active conversation
# --------------------------------------------------

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

threads = chat_manager.load_threads()

if not threads:
    st.session_state.active_thread_id = chat_manager.create_thread()
    threads = chat_manager.load_threads()

elif not any(
    thread["id"] == st.session_state.active_thread_id
    for thread in threads
):
    st.session_state.active_thread_id = threads[0]["id"]


# --------------------------------------------------
# 5. Delete chat
# --------------------------------------------------

def delete_chat(thread_id):
    delete_thread_checkpoints(checkpointer, thread_id)

    remaining_threads = [
        thread
        for thread in chat_manager.load_threads()
        if thread["id"] != thread_id
    ]

    chat_manager.save_threads(remaining_threads)

    if st.session_state.active_thread_id == thread_id:
        if remaining_threads:
            st.session_state.active_thread_id = remaining_threads[0]["id"]
        else:
            st.session_state.active_thread_id = chat_manager.create_thread()


# --------------------------------------------------
# 6. Sidebar: chats and PDF upload
# --------------------------------------------------

with st.sidebar:
    st.title("💬 Your Chats")

    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.active_thread_id = chat_manager.create_thread()
        st.rerun()

    st.divider()
    st.subheader("Chat History")

    threads = chat_manager.load_threads()

    for thread in threads:
        thread_id_item = thread["id"]
        is_active = thread_id_item == st.session_state.active_thread_id
        label = f"● {thread['title']}" if is_active else thread["title"]

        title_col, delete_col = st.columns(
            [0.82, 0.18],
            vertical_alignment="center",
            gap="small",
        )

        with title_col:
            if st.button(
                label,
                key=f"thread_{thread_id_item}",
                use_container_width=True,
            ):
                st.session_state.active_thread_id = thread_id_item
                st.rerun()

        with delete_col:
            if st.button(
                "✕",
                key=f"delete_{thread_id_item}",
                help="Permanently delete this chat",
                use_container_width=True,
            ):
                try:
                    delete_chat(thread_id_item)
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not delete chat: {error}")

    # ----------------------------------------------
    # PDF upload for the active chat
    # ----------------------------------------------

    st.divider()
    st.subheader("📄 Chat Documents")

    active_id = st.session_state.active_thread_id

    uploaded_pdf = st.file_uploader(
        "Upload a PDF for this chat",
        type=["pdf"],
        key=f"pdf_upload_{active_id}",
    )

    if st.button(
        "Process PDF",
        use_container_width=True,
        disabled=uploaded_pdf is None,
    ):
        upload_dir = (
            PROJECT_ROOT / "uploaded_documents" / active_id
        )
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_bytes = uploaded_pdf.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        marker_path = upload_dir / "document.json"

        previous_hash = None
        if marker_path.exists():
            try:
                previous_hash = json.loads(
                    marker_path.read_text(encoding="utf-8")
                ).get("sha256")
            except (OSError, json.JSONDecodeError):
                previous_hash = None

        if previous_hash == file_hash:
            st.info("This PDF has already been processed for this chat.")
        else:
            safe_name = Path(uploaded_pdf.name).name
            pdf_path = upload_dir / safe_name
            pdf_path.write_bytes(file_bytes)

            try:
                with st.spinner(
                    "Extracting PDF text and creating Gemini embeddings..."
                ):
                    pages, chunks = ingest_pdf(
                        pdf_path=str(pdf_path),
                        document_id=active_id,
                    )

                marker_path.write_text(
                    json.dumps({
                        "document_id": active_id,
                        "filename": safe_name,
                        "sha256": file_hash,
                    }),
                    encoding="utf-8",
                )

                st.success(
                    f"Processed {safe_name}\n\n"
                    f"Pages loaded: {pages}\n\n"
                    f"Chunks indexed: {chunks}"
                )

            except Exception as error:
                st.error(f"PDF processing failed: {error}")


# --------------------------------------------------
# 7. Select active thread and configure RAG
# --------------------------------------------------

thread_id = st.session_state.active_thread_id

marker_path = (
    PROJECT_ROOT
    / "uploaded_documents"
    / thread_id
    / "document.json"
)

document_id = None

if marker_path.exists():
    try:
        document_metadata = json.loads(
            marker_path.read_text(encoding="utf-8")
        )
        document_id = document_metadata.get("document_id")
    except (OSError, json.JSONDecodeError):
        document_id = None

config = {
    "configurable": {
        "thread_id": thread_id,
        "document_id": document_id,
    }
}


# --------------------------------------------------
# 8. Load saved messages
# --------------------------------------------------

try:
    saved_state = workflow.get_state(config)
    saved_messages = saved_state.values.get("messages", [])
except Exception as error:
    st.error(f"Could not load conversation: {error}")
    st.stop()


# --------------------------------------------------
# 9. Display conversation history
# --------------------------------------------------

def message_to_text(message):
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    return str(content)


for message in saved_messages:
    if message.type == "human":
        role = "user"
    elif message.type == "ai":
        role = "assistant"
    else:
        continue

    with st.chat_message(role):
        st.markdown(message_to_text(message))


# --------------------------------------------------
# 10. User input and streaming response
# --------------------------------------------------

user_input = st.chat_input("Message your chatbot...")

if user_input:
    chat_manager.update_thread_title(thread_id, user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:

            def stream_response():
                for chunk, metadata in workflow.stream(
                    {"messages": [("human", user_input)]},
                    config=config,
                    stream_mode="messages",
                ):
                    if metadata.get("langgraph_node") != "chatbot":
                        continue

                    content = chunk.content

                    if isinstance(content, str) and content:
                        yield content

                    elif isinstance(content, list):
                        for block in content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                            ):
                                text = block.get("text", "")
                                if text:
                                    yield text

            with st.spinner("Thinking..."):
                st.write_stream(stream_response())

        except Exception as error:
            st.error(f"Something went wrong: {error}")

    st.rerun()
