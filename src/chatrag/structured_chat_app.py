import os
import re
import sys

import streamlit as st
from langchain.chat_models import ChatOpenAI

# Allows streamlit cloud to import self-contained private reopository
module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(module_path)

from chatrag.react_agent_chat import get_react_chat_agent
from chatrag.retriever import create_retrieval_chain, create_retriever_from_csv

OPENAI_MODEL_MAP = {":rainbow[GPT-4]": "gpt-4", "***GPT-3.5***": "gpt-3.5-turbo-16k"}
HTML_BOX_TEMPLATE = """<p style="padding: 0 10px 0 10px; background-color: rgb(240, 242, 246); border-radius: 10px";>
        {text}</p>"""


def string_to_markdown(text):
    text = re.sub("```", "", text)
    text = re.sub("\n", "<br>", text)
    return text


def get_chat_agent():
    llm = ChatOpenAI(
        temperature=0,
        model=st.session_state.openai_model,
        max_tokens=3500,
        openai_api_key=st.session_state.api_key,
        openai_organization=st.session_state.openai_org_id,
    )
    root_app_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    retriever = create_retriever_from_csv(
        csv_path=f"{root_app_directory}/data/movies_title_overview_vote.csv",
        metadata_columns_dtypes={"vote_average": "float"},
        llm=llm,
    )
    sq_retrieval_chain = create_retrieval_chain(retriever=retriever, llm=llm)
    return get_react_chat_agent(llm, sq_retrieval_chain, verbose=True)


def app():
    if not hasattr(st.session_state, "api_key"):
        st.session_state.api_key = ""
    if not hasattr(st.session_state, "openai_org_id"):
        st.session_state.openai_org_id = ""
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
    st.session_state.api_key = st.sidebar.text_input("Ingrese su openai api key:")
    if st.session_state.api_key:
        st.session_state.disable_process = False
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
    else:
        st.session_state.disable_process = True
    st.session_state.openai_org_id = st.sidebar.text_input(
        "Ingrese su openai organization id:", placeholder="Puedes dejarlo vacío"
    )
    os.environ["OPENAI_ORGANIZATION"] = st.session_state.openai_org_id
    radio_model = st.sidebar.radio(
        "Elige qué modelo de OpenAI usar:",
        [":rainbow[GPT-4]", "***GPT-3.5***"],
        captions=[
            "Modelo más potente y caro.",
            "Modelo más barato. Recomendado para tareas de poca complejidad.",
        ],
    )
    if radio_model is not None:
        st.session_state.openai_model = OPENAI_MODEL_MAP[radio_model]

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
        # if not openai_api_key:
        #     st.info("Please add your OpenAI API key to continue.")
        #     st.stop()

        # openai.api_key = openai_api_key
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response_msg = st.session_state.react_chat.run(input=prompt)
        st.session_state.messages.append({"role": "assistant", "content": response_msg})
        st.chat_message("assistant").write(response_msg)


if __name__ == "__main__":
    app()
