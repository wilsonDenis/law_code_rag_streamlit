import os
import streamlit as st

if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

from rag import RAG

st.set_page_config(
    page_title="Assistant Code du Travail",
    page_icon="https://upload.wikimedia.org/wikipedia/fr/thumb/2/22/Minist%C3%A8re_du_Travail.svg/1200px-Minist%C3%A8re_du_Travail.svg.png",
    layout="centered",
)

@st.cache_resource
def load_rag():
    try:
        return RAG()
    except Exception as e:
        return None

def main():
    st.title("Assistant Code du Travail")
    st.markdown(
        "Posez vos questions sur le droit du travail francais. "
        "Les reponses s'appuient sur les textes officiels du Code du travail."
    )

    rag = load_rag()

    api_ok = bool(os.environ.get("GROQ_API_KEY"))

    if not api_ok:
        st.warning(
            "La cle API Groq n'est pas configuree. "
            "Les reponses ne seront pas generees par l'IA. "
            "Ajoutez GROQ_API_KEY dans les secrets Streamlit pour activer les reponses completes."
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
                    "Veuillez verifier que la cle API (GROQ_API_KEY) est correctement configuree "
                    "dans les parametres de l'application."
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
