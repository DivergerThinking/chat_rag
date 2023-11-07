from typing import Dict, Optional

from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chat_models.base import BaseChatModel
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings

from chatrag.csv_meta_loader import CSVMetaLoader
from chatrag.prompts import MOVIE_RETRIEVER_TEMPLATE


def create_retriever_from_csv(
    csv_path: str,
    llm: BaseChatModel,
    metadata_columns_dtypes: Optional[Dict[str, str]] = {"monthly_traffic": "int"},
    n_docs_to_retrieve: int = 20,
    embedding_provider: str = "openai",
):
    if embedding_provider is None or embedding_provider == "openai":
        embedding = OpenAIEmbeddings()
    elif embedding_provider == "vertexai":
        embedding = VertexAIEmbeddings()
    else:
        raise ValueError(f"Invalid embedding provider: {embedding_provider}")
    loader = CSVMetaLoader(csv_path, metadata_columns_dtypes=metadata_columns_dtypes)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    index_creator = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embedding)
    docsearch = index_creator.from_loaders([loader])

    metadata_field_info = [
        AttributeInfo(
            name="vote_average",
            description="The average score given to the movie.",
            type="float",
        )
    ]
    document_content_description = "List of movies with an overview and scoring from a public website."

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=docsearch.vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        enable_limit=False, # TODO: Enable dynamically at func call.
        verbose=True,
        search_kwargs={"k": n_docs_to_retrieve},
    )

    return retriever


def create_retrieval_chain(
    retriever: SelfQueryRetriever,
    llm: BaseChatModel,
    retriever_prompt_template: str = MOVIE_RETRIEVER_TEMPLATE,
):
    retrieval_prompt = PromptTemplate(template=retriever_prompt_template, input_variables=["context", "question"])

    media_retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": retrieval_prompt},
    )

    return media_retriever_chain
