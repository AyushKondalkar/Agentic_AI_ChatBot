
import os
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


from memory.checkpointer import get_checkpointer

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key,
)


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot_node(state: ChatbotState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(ChatbotState)

graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)


if __name__ == "__main__":
    with get_checkpointer() as checkpointer:
        chatbot_workflow = graph.compile(
            checkpointer=checkpointer
        )

        config = {
            "configurable": {
                "thread_id": "ayush-chat-1"
            }
        }

        print("Chatbot started! Type 'exit' to stop.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                print("Chatbot: Goodbye!")
                break

            if not user_input:
                continue

            try:
                result = chatbot_workflow.invoke(
                    {"messages": [("human", user_input)]},
                    config=config,
                )

                print("Chatbot:", result["messages"][-1].content)

            except Exception as error:
                print(f"Chatbot error: {error}")
