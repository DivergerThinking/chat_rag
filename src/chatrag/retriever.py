from typing import Dict, Optional

from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .csv_meta_loader import CSVMetaLoader
from .prompts import MOVIE_RETRIEVER_TEMPLATE


def create_retriever_from_csv(
    csv_path: str,
    llm: ChatOpenAI,
    metadata_columns_dtypes: Optional[Dict[str, str]] = {"monthly_traffic": "int"},
    n_docs_to_retrieve: int = 20
):
    loader = CSVMetaLoader(csv_path, metadata_columns_dtypes=metadata_columns_dtypes)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    index_creator = VectorstoreIndexCreator(text_splitter=text_splitter)
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
    llm: ChatOpenAI,
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
