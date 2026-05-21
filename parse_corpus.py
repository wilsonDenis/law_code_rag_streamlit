import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

OUTPUT_PATH     = "corpus/code_travail.json"
CODE_TRAVAIL_ID = "LEGITEXT000006072050"

SECTION_PAR_PREFIXE: dict[str, str] = {
    "L3121": "Durée du travail et heures supplémentaires",
    "L3122": "Durée du travail et heures supplémentaires",
    "L3123": "Durée du travail et heures supplémentaires",
    "L3131": "Durée du travail et heures supplémentaires",
    "L3132": "Durée du travail et heures supplémentaires",
    "L3141": "Congés payés",
    "L3142": "Congés payés",
    "L1221": "Contrat de travail (CDI, CDD)",
    "L1222": "Contrat de travail (CDI, CDD)",
    "L1224": "Contrat de travail (CDI, CDD)",
    "L1241": "Contrat de travail (CDI, CDD)",
    "L1242": "Contrat de travail (CDI, CDD)",
    "L1243": "Contrat de travail (CDI, CDD)",
    "L1244": "Contrat de travail (CDI, CDD)",
    "L1245": "Contrat de travail (CDI, CDD)",
    "L1231": "Licenciement",
    "L1232": "Licenciement",
    "L1233": "Licenciement",
    "L1234": "Licenciement",
    "L1235": "Licenciement",
    "L1251": "Licenciement",
    "L1252": "Licenciement",
    "L3231": "Salaire minimum (SMIC)",
    "L3232": "Salaire minimum (SMIC)",
    "L2311": "Représentation du personnel",
    "L2312": "Représentation du personnel",
    "L2313": "Représentation du personnel",
    "L2314": "Représentation du personnel",
    "L2315": "Représentation du personnel",
    "L2316": "Représentation du personnel",
    "L1132": "Harcèlement et discrimination",
    "L1133": "Harcèlement et discrimination",
    "L1152": "Harcèlement et discrimination",
    "L1153": "Harcèlement et discrimination",
}

_RUPTURE_RANGE = range(11, 20)


def _get_section(numero: str) -> str | None:
    match = re.match(r"(L\d{4})-(\d+)", numero)
    if not match:
        return None
    prefixe, suffixe = match.group(1), int(match.group(2))
    if prefixe == "L1237":
        return "Rupture conventionnelle" if suffixe in _RUPTURE_RANGE else "Licenciement"
    return SECTION_PAR_PREFIXE.get(prefixe)


def _nettoyer(texte: str) -> str:

    texte = re.sub(r"<[^>]+>", " ", texte)
    texte = (texte
             .replace("&nbsp;", " ").replace("&#160;", " ")
             .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"'))
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()


def _extraire_titre_contexte(root: ET.Element) -> str:
    titres = [t.text.strip() for t in root.findall(".//CONTEXTE//TITRE_TM") if t.text]
    return titres[-1] if titres else ""


def parser_fichier(fichier: Path) -> dict | None:

    try:
        root = ET.parse(fichier).getroot()
    except ET.ParseError:
        return None

    if (root.findtext(".//META_ARTICLE/ETAT") or "").upper() != "VIGUEUR":
        return None

    numero = (root.findtext(".//META_ARTICLE/NUM") or "").strip()
    if not numero:
        return None

    section = _get_section(numero)
    if not section:
        return None

    contenu_brut = root.findtext(".//BLOC_TEXTUEL/CONTENU") or ""
    texte = _nettoyer(contenu_brut)
    if not texte:
        return None

    titre = _extraire_titre_contexte(root) or numero

    return {
        "article": numero,
        "titre":   titre,
        "texte":   texte,
        "section": section,
    }


def trouver_dossier_articles(racine: Path) -> Path:

    candidats = list(racine.rglob(CODE_TRAVAIL_ID))
    for c in candidats:
        dossier_articles = c / "article"
        if dossier_articles.exists():
            return dossier_articles
    raise SystemExit(
        f"Impossible de localiser {CODE_TRAVAIL_ID}/article/ dans {racine}.\n"
        "Vérifiez que vous pointez vers le bon dossier."
    )


def main(dossier: str) -> None:
    racine = Path(dossier)
    if not racine.exists():
        raise SystemExit(f"Dossier introuvable : {dossier}")

    dossier_articles = trouver_dossier_articles(racine)
    print(f"Code du travail trouvé : {dossier_articles}\n")

    fichiers = list(dossier_articles.rglob("LEGIARTI*.xml"))
    print(f"{len(fichiers)} fichiers LEGIARTI à parcourir...\n")

    corpus: list[dict] = []
    for fichier in fichiers:
        article = parser_fichier(fichier)
        if article:
            corpus.append(article)

    corpus.sort(key=lambda a: a["article"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    compteur: dict[str, int] = {}
    for a in corpus:
        compteur[a["section"]] = compteur.get(a["section"], 0) + 1

    print(f"{len(corpus)} articles extraits → {OUTPUT_PATH}\n")
    print("Répartition par thème :")
    for section, n in sorted(compteur.items()):
        print(f"  {n:3d}  {section}")
    print("\nLancez maintenant : python indexation.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True, help="Dossier racine de l'archive LEGI extraite")
    main(ap.parse_args().dossier)
