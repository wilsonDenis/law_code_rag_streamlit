
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


class VectorDB:


    def __init__(self, path: str, chunks: list[dict] | None = None):

        self.index_path = path + ".faiss"
        self.meta_path  = path + ".json"

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._charger()
        elif chunks:
            self._creer(chunks)
        else:
            raise ValueError(
                "Base introuvable. Fournissez des chunks pour en créer une nouvelle."
            )



    def _creer(self, chunks: list[dict]) -> None:
        print(f"Création de la base FAISS ({len(chunks)} chunks)...")
        print(f"Modèle d'embedding : {EMBEDDING_MODEL_NAME}")

        self.model  = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        self.chunks = chunks

        vecteurs = self._embedder([c["texte_embedde"] for c in chunks])

        dimension   = vecteurs.shape[1]
        self.index  = faiss.IndexFlatIP(dimension)
        self.index.add(vecteurs)

        self._sauvegarder()
        print(f"Index FAISS créé  : {self.index.ntotal} vecteurs, dim={dimension}")



    def _sauvegarder(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"embedding_model": EMBEDDING_MODEL_NAME, "chunks": self.chunks},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Sauvegardé : {self.index_path}  +  {self.meta_path}")

    def _charger(self) -> None:
        print("Chargement de la base FAISS...")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunks    = meta["chunks"]
        model_name     = meta.get("embedding_model", EMBEDDING_MODEL_NAME)
        self.model     = SentenceTransformer(model_name, device="cpu")
        print(f"Modèle d'embedding : {model_name}")
        print(f"{self.index.ntotal} vecteurs chargés (dim={self.index.d})")



    def _embedder(self, textes: list[str]) -> np.ndarray:

        vecteurs = self.model.encode(
            textes,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vecteurs, dtype=np.float32)



    def retrieve(self, question: str, k: int = 4) -> tuple[list[str], list[dict]]:

        vecteur_q       = self._embedder([question])
        scores, indices = self.index.search(vecteur_q, k)

        textes    = []
        metadatas = []
        for idx in indices[0]:
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            textes.append(chunk["texte"])
            metadatas.append({
                "article": chunk["article"],
                "titre":   chunk["titre"],
                "section": chunk["section"],
            })
        return textes, metadatas
