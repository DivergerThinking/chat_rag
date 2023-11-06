from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from chatrag.prompts import REACT_PREFIX
from langchain.chains.retrieval_qa.base import BaseRetrievalQA


def get_react_chat_agent(llm: ChatOpenAI, qa_retriever: BaseRetrievalQA, verbose: bool = False):
    retriever_description = """Movie search tool. The action input must be just movie topics and description in a natural language sentence."""  # noqa
    tools = [Tool(name="Search movies", func=qa_retriever.run, description=retriever_description)]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_kwargs = {"system_message": REACT_PREFIX}

    react_chat_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        handle_parsing_errors=True,
    )
    return react_chat_agent
