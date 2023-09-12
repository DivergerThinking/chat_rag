# Retrieval Augmented Generation Chatbot

Working PoC for a QA Document with LLM (RAG+Chat) pipeline implementation in notebooks. The pipeline covers the following processes:
1. Pre-process CSV dataset
2. Creates a VectorDB and loads CSV dataset with metadata
3. Creates a Retriever able to query the VectorDB and reason over the retrieved data
4. Implements a chatbot able to use the retriever as a tool (agent) interatively


# Installation:

Before launching the notebooks it is needed to install the module. Just run the following comand on your terminal:

`pip install git+https://github.com/DivergerThinking/chat_rag_csv`

The module will be installed in your environment. Make sure you have activated the virtual environment if so. Adter installing it, you will be able to import the module in python as:

`ìmport chatrag`

# Usage:

First download the dataset at: \
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Second, take your OpenAI API key and Organization id, encode it in base64 and place it in a `secrets.json` file under the `data` folder.

Finally, check the notebooks as a guided usage PoC.