# ruff: noqa: E402
import os
import sys

# Fix for streamlit cloud outdated sqlite version
if sys.platform == "linux":
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st


def app():
    st.set_page_config(
        page_title="Chat Document QA",
        page_icon="💬",
    )

    st.title("💬 Chat Document QA")
    st.caption("Asistente en formato chatbot que puede consultar documentos estructurados.")

    st.markdown(
        """Esta app muestra cómo interactuar con un asistente basado en modelos de lenguaje que puede consultar un documento estructurado para responder preguntas y elaborar sobre ellas.<br>
                Cuando se habla de documentos estructurados, se refiere a documentos generalmente tipo fila-columna como pueden ser los CSVs o Excels.<br><br>
                Para esta demo, se utiliza un dataset de 5000 peliculas clasificadas y evaluadas en la web imdb. Échale un ojo [aquí](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).""",  # noqa
        unsafe_allow_html=True,
    )

    st.sidebar.title("Selecciona una demo para empezar.")


if __name__ == "__main__":
    app()
