from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from chatrag.prompts import REACT_PREFIX, REACT_SUFFIX
from langchain.chains.retrieval_qa.base import BaseRetrievalQA


def get_react_chat_agent(llm: ChatOpenAI, qa_retriever: BaseRetrievalQA, verbose: bool = False):
    retriever_description = """Movie search tool. The action input must be just movie topics and description in a natural language sentence""" # noqa
    tools = [
        Tool(
            name = "Search movies",
            func=qa_retriever.run,
            description=retriever_description
        )
    ]

    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_kwargs = {
            "memory_prompts": [chat_history],
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
            "prefix": REACT_PREFIX,
            "suffix": REACT_SUFFIX
        }

    react_chat_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
    )
    return react_chat_agent