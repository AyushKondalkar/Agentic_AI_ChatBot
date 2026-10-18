import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


class ChatbotState(TypedDict):
    messages: list[BaseMessage]


def chatbot_node(state: ChatbotState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(ChatbotState)

graph.add_node("chatbot", chatbot_node)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

chatbot_workflow = graph.compile()


if __name__ == "__main__":
    user_input = input("You: ")

    result = chatbot_workflow.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    print("Chatbot:", result["messages"][-1].content)