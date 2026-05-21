"""
agent_moderator.py — Garde-fou du RAG : détecte les prompt injections et les questions hors sujet.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

from groq import Groq

from config import MODERATOR_MODEL_NAME


class AgentModerator:
    """Analyse chaque question avant qu'elle n'atteigne le RAG.

    Détecte :
    - Les prompt injections (tentatives de manipulation des instructions système)
    - Les questions hors sujet (sans rapport avec le Code du travail français)
    """

    def __init__(self, client: Groq) -> None:
        self.client = client

    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def moderate(self, question: str) -> dict:
        """Retourne un dict avec is_prompt_injection, is_off_topic et reason."""
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self._read_file("moderator_context.txt")},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            model=MODERATOR_MODEL_NAME,
        )
        return json.loads(response.choices[0].message.content)

    def is_blocked(self, question: str) -> tuple[bool, str]:
        """Retourne (bloqué, raison). Interface simplifiée pour les consommateurs."""
        result = self.moderate(question)
        blocked = result.get("is_prompt_injection", False) or result.get("is_off_topic", False)
        return blocked, result.get("reason", "")
