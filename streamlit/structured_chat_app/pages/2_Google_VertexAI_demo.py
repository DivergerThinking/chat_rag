# ruff: noqa: E402
import os
import sys
import re

# Fix for streamlit cloud outdated sqlite version
if sys.platform == "linux":
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
import streamlit as st
from langchain.utilities.vertexai import init_vertexai
from langchain.chat_models import ChatVertexAI
# from langchain.globals import set_debug
# set_debug(True)

# Allows streamlit cloud to import self-contained private reopository
root_app_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
module_path = f"{root_app_directory}/src"
sys.path.append(module_path)

from chatrag.react_agent_chat import get_react_chat_agent
from chatrag.retriever import create_retrieval_chain, create_retriever_from_csv

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def load_creds():
    """Converts `gcp_chatrag_client_config env var` to a credential object."""
    creds = None
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_config(
                client_config=eval(st.secrets["gcp_chatrag_client_config"]),
                scopes=SCOPES,
                redirect_uri="urn:ietf:wg:oauth:2.0:oob",
            )
            st.session_state.g_auth_url, _ = flow.authorization_url(prompt="consent")
            st.session_state.flow = flow


def string_to_markdown(text):
    text = re.sub("```", "", text)
    text = re.sub("\n", "<br>", text)
    return text


def get_chat_agent():
    llm = ChatVertexAI(model_name="chat-bison", temperature=0.7, max_output_tokens=2000)
    print("LLM creation worked.")
    retriever = create_retriever_from_csv(
        csv_path=f"{root_app_directory}/data/movies_title_overview_vote.csv",
        metadata_columns_dtypes={"vote_average": "float"},
        llm=llm,
        embedding_provider="vertexai",
    )
    sq_retrieval_chain = create_retrieval_chain(retriever=retriever, llm=llm)
    return get_react_chat_agent(llm, sq_retrieval_chain, verbose=True)


def app():
    if not hasattr(st.session_state, "g_auth_url"):
        st.session_state.g_auth_url = ""
    if not hasattr(st.session_state, "g_auth_creds"):
        st.session_state.g_auth_creds = ""
    if not hasattr(st.session_state, "flow"):
        st.session_state.flow = False
    if not hasattr(st.session_state, "disable_process"):
        st.session_state.disable_process = True
    if not hasattr(st.session_state, "disable_chat"):
        st.session_state.disable_chat = True
    if not hasattr(st.session_state, "recommendation"):
        st.session_state.recommendation = ""
    if not hasattr(st.session_state, "movie_query"):
        st.session_state.movie_query = "Quiero ver un thriller basado en el espacio con puntuación mayor a 7."
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Say hi and your name to start."}]

    st.title("💬 Chat Document QA")
    st.caption("Asistente en formato chatbot que puede consultar documentos estructurados.")

    st.markdown(
        """Esta app muestra cómo interactuar con un asistente basado en modelos de lenguaje que puede consultar un documento estructurado para responder preguntas y elaborar sobre ellas.<br>
                Cuando se habla de documentos estructurados, se refiere a documentos generalmente tipo fila-columna como pueden ser los CSVs o Excels.<br><br>
                Para esta demo, se utiliza un dataset de 5000 peliculas clasificadas y evaluadas en la web imdb. Échale un ojo [aquí](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).""",  # noqa
        unsafe_allow_html=True,
    )

    st.sidebar.title("Configuración")
    st.sidebar.button("Google login", on_click=load_creds)
    if st.session_state.flow:
        st.sidebar.markdown("""Visita el link para conseguir el codigo de autorización:<br>
                                [web de autorización]({})""".format(st.session_state.g_auth_url), unsafe_allow_html=True)
        st.session_state.g_auth_creds = st.sidebar.text_input("Ingrese su código de autorización de google:")
        if st.sidebar.button("Validar"):
            st.session_state.flow.fetch_token(code=st.session_state.g_auth_creds)
            init_vertexai(project="chatrag", location="europe-west9", credentials=st.session_state.flow.credentials)
            ChatVertexAI(model_name="chat-bison", temperature=0.7, max_output_tokens=2000)
            print("LLM creation worked.")
            st.session_state.disable_process = False

    if st.button("Procesar documento y crear el asistente", disabled=st.session_state.disable_process):
        st.session_state.react_chat = get_chat_agent()
        st.session_state.disable_chat = False

    st.markdown(
        """A continuación puedes interactuar con el asistente de forma natural.""",  # noqa
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(disabled=st.session_state.disable_chat):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response_msg = st.session_state.react_chat.run(input=prompt)
        st.session_state.messages.append({"role": "assistant", "content": response_msg})
        st.chat_message("assistant").write(response_msg)


if __name__ == "__main__":
    app()
