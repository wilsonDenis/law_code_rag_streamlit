
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import re

from vector_db import VectorDB
from config import CORPUS_PATH, VECTOR_DB_NAME



def load_corpus(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def chunker(texte: str, taille_max: int = 500, overlap: int = 50) -> list[str]:

    texte = texte.strip()
    if not texte:
        return []


    if len(texte) <= taille_max:
        return [texte]

    segments = re.split(r"(?=\s+\d+°\s)", texte)

    chunks   = []
    courant  = ""

    for segment in segments:
        if not segment.strip():
            continue
        if len(courant) + len(segment) <= taille_max:
            courant = (courant + segment).strip()
        else:
            if courant:
                chunks.append(courant)
      
            fin = courant[-overlap:] if courant and overlap > 0 else ""
            courant = (fin + " " + segment).strip()

    if courant:
        chunks.append(courant)


    if not chunks:
        debut = 0
        while debut < len(texte):
            fin = min(debut + taille_max, len(texte))
            chunks.append(texte[debut:fin])
            debut += taille_max - overlap

    return chunks



def preparer_chunks(articles: list[dict]) -> list[dict]:

    chunks = []
    for article in articles:
        fragments = chunker(article["texte"], taille_max=500, overlap=50)
        for fragment in fragments:
            chunks.append({
                "texte":         fragment,

                "texte_embedde": (
                    f"Article {article['article']} — {article['titre']} "
                    f"[{article['section']}] : {fragment}"
                ),
                "article": article["article"],
                "titre":   article["titre"],
                "section": article["section"],
            })
    return chunks



def main() -> None:

    for ext in (".faiss", ".json"):
        chemin = VECTOR_DB_NAME + ext
        if os.path.exists(chemin):
            os.remove(chemin)
            print(f"Suppression : {chemin}")

    articles = load_corpus(CORPUS_PATH)
    print(f"{len(articles)} articles chargés depuis {CORPUS_PATH}")

    sections = {a["section"] for a in articles}
    print(f"Thèmes couverts ({len(sections)}) : {', '.join(sorted(sections))}")

    chunks = preparer_chunks(articles)
    print(f"{len(chunks)} chunks produits après découpage\n")

    VectorDB(path=VECTOR_DB_NAME, chunks=chunks)

    print(" Indexation terminée.")
    
    print("Lancez maintenant : python rag.py")


if __name__ == "__main__":
    main()
