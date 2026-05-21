"""
rag.py — Assistant Code du travail français (RAG + Groq).
    python rag.py
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dotenv import load_dotenv
from groq import Groq

from vector_db import VectorDB
from agent_moderator import AgentModerator
from config import LLM_MODEL_NAME, VECTOR_DB_NAME


class RAG:
    """Système RAG pour répondre aux questions sur le Code du travail français."""

    def __init__(self) -> None:
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.vector_db = VectorDB(VECTOR_DB_NAME)
        self.moderator = AgentModerator(self.client)

    # ------------------------------------------------------------------ #
    #  Construction du prompt                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_context(self, docs: list[str], metadatas: list[dict]) -> str:
        template = self._read_file("context.txt")
        chunks_text = ""
        for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
            chunks_text += (
                f"[{i}] Article {meta['article']} — {meta['titre']}\n"
                f"    Section : {meta['section']}\n"
                f"    Texte : {doc}\n\n"
            )
        return template.replace("{{Chuncks}}", chunks_text)

    # ------------------------------------------------------------------ #
    #  Réponse                                                             #
    # ------------------------------------------------------------------ #

    def answer_question(self, question: str) -> tuple[str, list[str]]:
        """Retourne (réponse_LLM, liste_articles_sources)."""
        blocked, reason = self.moderator.is_blocked(question)
        if blocked:
            return f"Je ne peux pas répondre à cette question. ({reason})", []

        docs, metadatas = self.vector_db.retrieve(question, k=4)
        context = self._build_context(docs, metadatas)

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question},
            ],
            model=LLM_MODEL_NAME,
        )
        answer = response.choices[0].message.content

        sources = sorted({meta["article"] for meta in metadatas})
        return answer, sources


# ---------------------------------------------------------------------- #
#  Interface en ligne de commande                                         #
# ---------------------------------------------------------------------- #

def main() -> None:
    print("Chargement de la base de connaissances...")
    rag = RAG()
    print("\n■ Système RAG prêt. Tapez 'quit' pour quitter.\n")
    print("─" * 60)

    while True:
        question = input("\n■ Votre question : ").strip()

        if question.lower() in {"quit", "exit", "q"}:
            print("Au revoir !")
            break

        if not question:
            continue

        print("\nRecherche en cours...\n")
        answer, sources = rag.answer_question(question)

        print(answer)
        if sources:
            print(f"\n■ Sources : Articles {', '.join(sources)} du Code du travail")
        print("\n" + "─" * 60)


if __name__ == "__main__":
    main()
