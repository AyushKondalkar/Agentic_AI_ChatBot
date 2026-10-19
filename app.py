
import streamlit as st
import uuid

from memory.checkpointer import get_checkpointer
from ChatBot_workflow import graph
from memory.chat_manager import ChatManager

chat_manager = ChatManager()


# --------------------------------------------------
# 1. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic AI Chatbot")
st.caption("Powered by Groq, LangGraph, SQLite, and Streamlit")


# --------------------------------------------------
# 2. Create a unique conversation ID
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# --------------------------------------------------
# 3. Store messages for displaying in the UI
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# LangGraph uses this ID to remember the conversation
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# --------------------------------------------------
# 4. Create the workflow with persistent memory
# --------------------------------------------------

@st.cache_resource
def create_workflow():
    checkpointer, connection = get_checkpointer()

    workflow = graph.compile(
        checkpointer=checkpointer
    )

    # Keep the SQLite connection alive while the
    # cached workflow is being used.
    return workflow, checkpointer, connection


workflow, checkpointer, connection = create_workflow()


# --------------------------------------------------
# 5. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 6. Accept user input
# --------------------------------------------------

user_input = st.chat_input("Ask me anything...")


if user_input:

    # Display and store the user's message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # --------------------------------------------------
    # 7. Generate and stream the assistant's response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        try:
            with st.spinner("Thinking..."):

                def stream_response():
                    for message_chunk, metadata in workflow.stream(
                        {
                            "messages": [
                                ("human", user_input)
                            ]
                        },
                        config=config,
                        stream_mode="messages",
                    ):

                        # Only display output from our chatbot node
                        if metadata.get("langgraph_node") != "chatbot":
                            continue

                        # Stream ordinary text chunks
                        content = message_chunk.content

                        if isinstance(content, str) and content:
                            yield content

                        # Handle content-block lists, if returned
                        elif isinstance(content, list):
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    text = block.get("text", "")
                                    if text:
                                        yield text


                # Show each chunk as it arrives
                answer = st.write_stream(stream_response())


            # Save the complete answer in the UI history
            if answer:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

        except Exception as error:
            st.error(
                f"Something went wrong: {error}"
            )
