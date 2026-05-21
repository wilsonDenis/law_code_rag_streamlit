import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

from rag import RAG

st.set_page_config(
    page_title="Assistant Code du Travail",
    layout="centered",
)

@st.cache_resource
def load_rag():
    try:
        return RAG()
    except Exception:
        return None

def main():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            color: #1E3A5F;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .stChatMessage {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="main-title">Assistant Code du Travail</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Posez vos questions sur le droit du travail francais. '
        'Les reponses s\'appuient sur les textes officiels du Code du travail.</p>',
        unsafe_allow_html=True,
    )

    rag = load_rag()
    api_ok = bool(os.environ.get("GROQ_API_KEY"))

    if not api_ok:
        st.warning(
            "La cle API Groq n'est pas configuree. "
            "Ajoutez GROQ_API_KEY dans les secrets Streamlit pour activer les reponses."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if rag is None or not api_ok:
                full_response = (
                    "Le service est temporairement indisponible. "
                    "Veuillez verifier que la cle API (GROQ_API_KEY) est correctement configuree."
                )
                st.markdown(full_response)
            else:
                with st.spinner("Recherche en cours..."):
                    try:
                        answer, sources = rag.answer_question(prompt)

                        sources_text = ""
                        if sources:
                            sources_text = (
                                "\n\n---\n**Sources** : Articles "
                                + ", ".join(sources)
                                + " du Code du travail"
                            )

                        full_response = answer + sources_text
                        st.markdown(full_response)

                    except Exception:
                        full_response = (
                            "Une erreur est survenue lors du traitement de votre question. "
                            "Veuillez reessayer dans quelques instants."
                        )
                        st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    with st.sidebar:
        st.markdown("### A propos")
        st.markdown(
            "Cet assistant utilise un systeme RAG "
            "(Retrieval-Augmented Generation) pour repondre "
            "aux questions sur le Code du travail francais."
        )
        st.markdown("---")
        st.markdown(
            "**Modele LLM** : Llama 3.3 70B\n\n"
            "**Embeddings** : distiluse-base-multilingual\n\n"
            "**Base vectorielle** : FAISS"
        )
        st.markdown("---")
        st.caption("Projet pedagogique - RAG Code du travail")

if __name__ == "__main__":
    main()
