# ruff: noqa: E402
import os
import sys
import re

# Fix for streamlit cloud outdated sqlite version
if sys.platform == "linux":
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from openai import OpenAI
import logging

# set logging level to info
logging.getLogger("openai").setLevel(logging.INFO)

# Allows streamlit cloud to import self-contained private reopository
root_app_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
module_path = f"{root_app_directory}/src"
sys.path.append(module_path)

from chatrag.movie_agent import movie_agent
from chatrag.tools import create_movie_search_tool_from_csv
from chatrag.prompts import MOVIE_CHATBOT_TEMPLATE
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_MAP = {":rainbow[GPT-4]": "gpt-4-turbo", "***GPT-3.5***": "gpt-3.5-turbo-0125"}
HTML_BOX_TEMPLATE = """<p style="padding: 0 10px 0 10px; background-color: rgb(240, 242, 246); border-radius: 10px";>
        {text}</p>"""


def string_to_markdown(text):
    text = re.sub("```", "", text)
    text = re.sub("\n", "<br>", text)
    return text


def app():
    if not hasattr(st.session_state, "disable_process"):
        st.session_state.disable_process = True
    if not hasattr(st.session_state, "disable_chat"):
        st.session_state.disable_chat = True
    if not hasattr(st.session_state, "movie_query"):
        st.session_state.movie_query = "Quiero ver un thriller basado en el espacio con puntuación mayor a 7."
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = [{"role": "system", "content": MOVIE_CHATBOT_TEMPLATE}]
    if "messages" not in st.session_state:
        st.session_state["messages"] = None

    st.title("💬 Chat Document QA")
    st.caption("Asistente en formato chatbot que puede consultar documentos estructurados.")

    st.markdown(
        """Esta app muestra cómo interactuar con un asistente basado en modelos de lenguaje que puede consultar un documento estructurado para responder preguntas y elaborar sobre ellas.<br>
                Cuando se habla de documentos estructurados, se refiere a documentos generalmente tipo fila-columna como pueden ser los CSVs o Excels.<br><br>
                Para esta demo, se utiliza un dataset de 5000 peliculas clasificadas y evaluadas en la web imdb. Échale un ojo [aquí](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).""",  # noqa
        unsafe_allow_html=True,
    )

    st.sidebar.title("Configuración")
    radio_model = st.sidebar.radio(
        "Elige qué modelo de OpenAI usar:",
        [":rainbow[GPT-4]", "***GPT-3.5***"],
        captions=[
            "Modelo más potente.",
            "Modelo más barato y rápido.",
        ],
        index=None,
    )
    if radio_model is not None:
        st.session_state.disable_process = False
        st.session_state.openai_model = OPENAI_MODEL_MAP[radio_model]

    if st.button("Procesar documento y habilitar el asistente", disabled=st.session_state.disable_process):
        st.session_state.movie_tool_dict = create_movie_search_tool_from_csv(
            csv_path=f"{root_app_directory}/data/movies_title_overview_vote.csv",
            metadata_columns_dtypes={"vote_average": "float"},
        )
        st.session_state.openai_client = OpenAI()
        st.session_state.react_chat = movie_agent
        st.session_state.disable_chat = False
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Bienvenido al asistente de recomendación de películas, dime tu nombre y empecemos :smiley:",
            }
        ]

    if st.session_state.messages:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(
        disabled=st.session_state.disable_chat, placeholder="Escribe aquí tu mensaje para el chatbot"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response_msg = st.session_state.react_chat(
            history=st.session_state["system_prompt"] + st.session_state.messages,
            oai_client=st.session_state.openai_client,
            tool_name_dict=st.session_state.movie_tool_dict,
            model_name=st.session_state.openai_model,
        )
        st.session_state.messages.append({"role": "assistant", "content": response_msg})
        st.chat_message("assistant").write(response_msg)
        print(st.session_state["system_prompt"] + st.session_state.messages)


if __name__ == "__main__":
    app()
