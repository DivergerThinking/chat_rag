from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from chatrag.csv_meta_loader import CSVMetaLoader
from typing import Dict, Optional


def create_retriever_from_csv(
    csv_path: str,
    metadata_columns_dtypes: Optional[Dict[str, str]] = {"monthly_traffic": "int"},
    llm: ChatOpenAI = ChatOpenAI(temperature=0, model="gpt-4", max_tokens=2000),
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
        enable_limit=False,
        verbose=True,
        search_kwargs={"k": 20},
    )

    return retriever


def create_retrieval_chain(
    retriever: SelfQueryRetriever,
    llm: ChatOpenAI = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=2000),
):
    template = """Use the following movies data to find the best matches for the user request in the question overview topic. Rules:
    - You can return more than one movie if they are a good match.
    - Answer with movie names of the medias and a short text justifying the choice.
    - The justification must take into account overview topic matching and vote average score (higher=better).

    Media data:
    {context}

    Question overview topic:
    {question}

    Example answer:
    1. Title: [selected movie title]
    - Justification: [Given justification]
    - Score: [vote_average]

    Movies attending to rules and ordered from best to worst:
    """

    retrieval_prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    media_retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": retrieval_prompt},
    )

    return media_retriever_chain
