from typing import Dict, Optional, Callable

from chromadb.errors import InvalidDimensionException
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

from chatrag.csv_meta_loader import CSVMetaLoader

# from chatrag.prompts import MOVIE_RETRIEVER_TEMPLATE


def create_movie_search_tool_from_csv(
    csv_path: str,
    metadata_columns_dtypes: Optional[Dict[str, str]] = None,
    embedding_provider: str = "openai",
    embedding_deployment: Optional[str] = None,
) -> dict[str, Callable]:
    if embedding_provider is None or embedding_provider == "openai":
        if embedding_deployment:
            embedding = OpenAIEmbeddings(model="text-embedding-3-small", deployment=embedding_deployment)
        else:
            embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    # elif embedding_provider == "vertexai":
    #     embedding = VertexAIEmbeddings()
    else:
        raise ValueError(f"Invalid embedding provider: {embedding_provider}")
    loader = CSVMetaLoader(csv_path, metadata_columns_dtypes=metadata_columns_dtypes)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    index_creator = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embedding, vectorstore_cls=Chroma)
    try:
        docsearch = index_creator.from_loaders([loader])
    except InvalidDimensionException:
        Chroma().delete_collection()
        docsearch = index_creator.from_loaders([loader])

    def search_movies(descripcion_pelicula: str, n_recomendaciones: int = 5, puntuacion_minima: float = 0) -> str:
        docs = docsearch.vectorstore.similarity_search(
            descripcion_pelicula, k=n_recomendaciones, filter={"vote_average": {"$gte": puntuacion_minima}}
        )
        return "\n\n".join([doc.page_content for doc in docs])

    return {"search_movies": search_movies}


movie_tool_oai_format = [
    {
        "type": "function",
        "function": {
            "name": "search_movies",
            "description": "Search movies based on a description or synopsis and optionally a minimum rating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "descripcion_pelicula": {
                        "type": "string",
                        "description": "Descripcion o sinopsis del tipo de película a buscar",
                    },
                    "puntuacion_minima": {
                        "type": "number",
                        "description": "Puntuación mínima de la película del 0 al 10.",
                    },
                    "n_recomendaciones": {
                        "type": "integer",
                        "description": "Número de peliculas recomendadas a recibir.",
                    },
                },
                "required": ["descripcion_pelicula"],
            },
        },
    }
]
