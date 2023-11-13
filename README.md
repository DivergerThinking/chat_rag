# Retrieval Augmented Generation Document QA

Working PoC for a QA Document with LLM (RAG) pipeline implementation in notebooks. The pipeline covers the following processes:
1. Pre-process CSV dataset
2. Creates a VectorDB and loads CSV dataset with metadata
3. Creates a Retriever able to query the VectorDB and reason over the retrieved data
4. **Extra**: Implements a chatbot able to use the retriever as a tool (agent) interatively


# Installation:

Before launching the notebooks it is needed to install the module. Just run the following comand on your terminal:

`pip install git+https://github.com/DivergerThinking/chat_rag`

If you want to run streamlit too, run the following command too:

`pip install "chatrag[dev] @ git+https://github.com/DivergerThinking/chat_rag"`

The module will be installed in your environment. Make sure you have activated the virtual environment if so. Adter installing it, you will be able to import the module in python as:

`ìmport chatrag`

# Usage:

## Notebook mode
For the OpenAI demos, you just need to set a `.env` file on the root directory with the openai_api_key and org id. Exmaple:
```
OPENAI_API_KEY = yourapikey
OPENAI_ORGANIZATION = yourorgid
```

For Google Vertex AI demo you need to provide a credentials.json got from your Google Cloud Platform account. Check how at:
- retriever_agent_react_poc_gcp.ipynb
- confluence doc

Finally, check the notebooks as a guided usage PoC.

## Streamlit web app mode

1. Open a terminal and move to the repo directory.
2. Run one of the following commands for each app:
`streamlit run src/chatrag/structured_qa_app.py`
`streamlit run streamlit/structured_chat_app/Home.py`
3. The web interface will open in your default browser.
4. Play around and have some fun.
