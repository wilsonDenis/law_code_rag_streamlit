# Assistant Code du Travail

Assistant intelligent pour le droit du travail francais, base sur un systeme RAG (Retrieval-Augmented Generation).

## Fonctionnalites

- Reponses aux questions sur le Code du travail francais
- Recherche semantique dans les articles de loi via FAISS
- Agent moderateur integre contre les prompt injections et les questions hors sujet
- Interface conversationnelle avec historique de session

## Architecture technique

| Composant | Technologie |
|-----------|-------------|
| LLM | Llama 3.3 70B (via Groq) |
| Embeddings | distiluse-base-multilingual-cased-v2 |
| Base vectorielle | FAISS |
| Moderateur | Llama 3.1 8B (via Groq) |
| Interface | Streamlit |

## Installation locale

```bash
pip install -r requirements.txt
```

Creer un fichier `.env` a la racine :
```
GROQ_API_KEY=votre_cle_ici
```

Lancer l'application :
```bash
streamlit run app.py
```

## Structure du projet

```
app.py                  Interface Streamlit
rag.py                  Moteur RAG (retrieval + generation)
vector_db.py            Gestion de la base vectorielle FAISS
agent_moderator.py      Filtrage des questions (injection, hors sujet)
indexation.py           Pipeline de construction de la base
parse_corpus.py         Extraction des articles depuis les archives LEGI
config.py               Configuration des modeles et chemins
context.txt             Prompt systeme du LLM principal
moderator_context.txt   Prompt systeme du moderateur
corpus/                 Articles du Code du travail (JSON)
```