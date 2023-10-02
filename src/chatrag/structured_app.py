from chatrag.retriever import create_retriever_from_csv, create_retrieval_chain
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os
import re

OPENAI_MODEL_MAP = {":rainbow[GPT-4]": "gpt-4", "***GPT-3.5***": "gpt-3.5-turbo-16k"}
HTML_BOX_TEMPLATE = """<p style="padding: 0 10px 0 10px; background-color: rgb(240, 242, 246); border-radius: 10px";>
        {text}</p>"""


def string_to_markdown(text):
    text = re.sub("```", "", text)
    text = re.sub("\n", "<br>", text)
    return text


def get_retrieval_chain():
    llm = ChatOpenAI(
        temperature=0,
        model=st.session_state.openai_model,
        max_tokens=2000,
        openai_api_key=st.session_state.api_key,
        openai_organization=st.session_state.openai_org_id,
    )
    root_app_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    retriever = create_retriever_from_csv(
        csv_path=f"{root_app_directory}/data/movies_title_overview_vote.csv",
        metadata_columns_dtypes={"vote_average": "float"},
        llm=llm,
    )
    return create_retrieval_chain(retriever=retriever, llm=llm)


def app():
    if not hasattr(st.session_state, "api_key"):
        st.session_state.api_key = ""
    if not hasattr(st.session_state, "openai_org_id"):
        st.session_state.openai_org_id = ""
    if not hasattr(st.session_state, "disable_process"):
        st.session_state.disable_process = True
    if not hasattr(st.session_state, "disable_recom"):
        st.session_state.disable_recom = True
    if not hasattr(st.session_state, "recommendation"):
        st.session_state.recommendation = ""
    if not hasattr(st.session_state, "movie_query"):
        st.session_state.movie_query = "Quiero ver un thriller basado en el espacio con puntuación mayor a 7."

    st.title("Document QA")
    st.header("Documento estructurado (.csv)")

    st.markdown(
        """Esta app muestra cómo interactuar mediante modelos de lenguaje con un documento estructurado.<br>
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

    if st.button("Procesar documento y crear un Document QA", disabled=st.session_state.disable_process):
        st.session_state.retrieval_chain = get_retrieval_chain()
        st.session_state.disable_recom = False

    st.session_state.movie_query = st.text_input(
        "Qué tipo de película te apetece ver? Puedes especificar la puntuación.",
        st.session_state.movie_query,
        disabled=st.session_state.disable_recom,
    )

    if st.button("Buscar película", disabled=st.session_state.disable_recom):
        st.session_state.recommendation = string_to_markdown(
            st.session_state.retrieval_chain(st.session_state.movie_query)["result"]
        )

    if st.session_state.recommendation:
        mod_md_html = HTML_BOX_TEMPLATE.format(text=st.session_state.recommendation)
        st.markdown(mod_md_html, unsafe_allow_html=True)


if __name__ == "__main__":
    app()
