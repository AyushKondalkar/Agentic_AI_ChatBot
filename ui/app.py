

import sys
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


# --------------------------------------------------
# 2. Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI Chatbot")
st.caption("Powered by Groq, LangGraph, SQLite, and Streamlit")


# --------------------------------------------------
# 3. Initialize chat manager
# --------------------------------------------------

chat_manager = ChatManager()


# --------------------------------------------------
# 4. Create workflow with persistent memory
# --------------------------------------------------

@st.cache_resource
def create_workflow():
    checkpointer, connection = get_checkpointer()
    workflow = graph.compile(checkpointer=checkpointer)

    # Keep the database connection alive.
    return workflow, checkpointer, connection


workflow, checkpointer, connection = create_workflow()


# --------------------------------------------------
# 5. Initialize active conversation
# --------------------------------------------------

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

threads = chat_manager.load_threads()

if not threads:
    st.session_state.active_thread_id = (
        chat_manager.create_thread()
    )
    threads = chat_manager.load_threads()

elif not any(
    thread["id"] == st.session_state.active_thread_id
    for thread in threads
):
    st.session_state.active_thread_id = threads[0]["id"]


# --------------------------------------------------
# 6. Delete one chat from SQLite and JSON
# --------------------------------------------------

def delete_chat(thread_id):
    # Delete persistent LangGraph state first.
    # If this fails, do not remove the sidebar entry.
    delete_thread_checkpoints(checkpointer, thread_id)

    # Remove only the selected chat's metadata.
    remaining_threads = [
        thread
        for thread in chat_manager.load_threads()
        if thread["id"] != thread_id
    ]

    chat_manager.save_threads(remaining_threads)

    # If the active chat was deleted, select another chat.
    if st.session_state.active_thread_id == thread_id:
        if remaining_threads:
            st.session_state.active_thread_id = (
                remaining_threads[0]["id"]
            )
        else:
            st.session_state.active_thread_id = (
                chat_manager.create_thread()
            )


# --------------------------------------------------
# 7. Sidebar: New Chat and Chat History
# --------------------------------------------------

with st.sidebar:
    st.title("💬 Your Chats")

    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.active_thread_id = (
            chat_manager.create_thread()
        )
        st.rerun()

    st.divider()
    st.subheader("Chat History")

    threads = chat_manager.load_threads()

    for thread in threads:
        thread_id_item = thread["id"]
        is_active = (
            thread_id_item == st.session_state.active_thread_id
        )

        label = thread["title"]

        if is_active:
            label = f"● {label}"

        # Title and cross appear on the same row.
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
                type="secondary",
            ):
                st.session_state.active_thread_id = (
                    thread_id_item
                )
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
                    st.error(
                        f"Could not delete chat: {error}"
                    )


# --------------------------------------------------
# 8. Select the active thread
# --------------------------------------------------

thread_id = st.session_state.active_thread_id

config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# --------------------------------------------------
# 9. Load saved messages from LangGraph
# --------------------------------------------------

try:
    saved_state = workflow.get_state(config)
    saved_messages = saved_state.values.get("messages", [])

except Exception as error:
    st.error(f"Could not load conversation: {error}")
    st.stop()


# --------------------------------------------------
# 10. Display conversation history
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
# 11. User input and streaming response
# --------------------------------------------------

user_input = st.chat_input("Message your chatbot...")

if user_input:

    chat_manager.update_thread_title(
        thread_id,
        user_input,
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            def stream_response():
                for chunk, metadata in workflow.stream(
                    {
                        "messages": [
                            ("human", user_input)
                        ]
                    },
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
