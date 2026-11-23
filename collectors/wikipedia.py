# =============================================================================
# collectors/wikipedia.py — Étape 3b : popularité des joueurs (pages vues Wikipédia).
#
# Pourquoi : un joueur populaire vaut plus cher (maillots, sponsors, audience).
# Les pages vues Wikipédia sont une mesure gratuite, quotidienne depuis
# juillet 2015, et comparable entre joueurs.
#
# Comment :
#   1. Wikidata (la base de données derrière Wikipédia) stocke l'identifiant
#      Transfermarkt de chaque joueur (propriété "P2446"). On s'en sert pour
#      trouver SANS ERREUR la page Wikipédia de chaque joueur, dans 5 langues.
#   2. L'API Wikimedia donne les vues mensuelles de chaque page.
# =============================================================================

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor   # plusieurs téléchargements en parallèle
from urllib.parse import unquote                    # "Kylian_Mbapp%C3%A9" -> "Kylian_Mbappé"

import pandas as pd
import requests
from tqdm import tqdm

import config
from collectors import transfermarkt as tm

# Wikimedia exige un "User-Agent" qui décrit le programme ET donne un moyen de
# contact ; sans contact, l'API bloque très vite (erreur 429).
HEADERS = {"User-Agent": "MarkovChainFootball/0.1 (https://github.com/Nat-404; personal research project) python-requests"}
LANGUAGES = ["en", "fr", "es", "de", "it"]           # les langues des 5 grands championnats + anglais
CACHE = config.CACHE_DIR / "wikipedia"
LINKS_PATH = config.PROCESSED_DIR / "wikipedia_links.parquet"
VIEWS_PATH = config.PROCESSED_DIR / "wikipedia_views.parquet"
START, END = "2015070100", "2026103100"              # période disponible : juillet 2015 -> octobre 2026


def target_players():
    # Tous les joueurs ayant joué au moins un match dans les 5 grands championnats depuis 2014.
    codes = "', '".join(tm.BIG5_CODES)
    return tm.query(
        f"""
        SELECT DISTINCT a.player_id
        FROM appearances a JOIN games g ON a.game_id = CAST(g.game_id AS INTEGER)
        WHERE g.competition_id IN ('{codes}') AND g.season >= '{config.FIRST_SEASON}'
        """
    ).player_id.tolist()


def fetch_links(player_ids, chunk=400):
    # Interroge Wikidata par paquets de 400 joueurs : pour chaque identifiant
    # Transfermarkt, récupère le titre de sa page dans chacune des 5 langues.
    optional = "\n".join(
        f"OPTIONAL {{ ?{l} schema:about ?p ; schema:isPartOf <https://{l}.wikipedia.org/> . }}"
        for l in LANGUAGES
    )
    rows = []
    for i in tqdm(range(0, len(player_ids), chunk), desc="Wikidata"):
        values = " ".join(f'"{pid}"' for pid in player_ids[i:i + chunk])
        query = f"SELECT ?tm {' '.join('?' + l for l in LANGUAGES)} WHERE {{ ?p wdt:P2446 ?tm . {optional} VALUES ?tm {{ {values} }} }}"
        response = requests.get("https://query.wikidata.org/sparql",
                                params={"query": query, "format": "json"}, headers=HEADERS, timeout=120)
        response.raise_for_status()
        for b in response.json()["results"]["bindings"]:
            row = {"player_id": int(b["tm"]["value"])}
            for l in LANGUAGES:
                if l in b:
                    # L'URL complète -> juste le titre de la page, sans encodage
                    row[l] = unquote(b[l]["value"].rsplit("/wiki/", 1)[1])
            rows.append(row)
        time.sleep(1)
    # Un même joueur peut apparaître 2 fois (doublon Wikidata) : on garde la 1re ligne
    links = pd.DataFrame(rows).drop_duplicates("player_id")
    links.to_parquet(LINKS_PATH)
    return links


def _views_one(lang, title):
    # Vues mensuelles d'UNE page dans UNE langue, avec cache sur disque.
    safe = title.replace("/", "%2F")
    cache_file = CACHE / lang / f"{safe[:150]}.json.gz"
    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            return json.load(f)
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           f"{lang}.wikipedia/all-access/user/{requests.utils.quote(safe, safe='')}/monthly/{START}/{END}")
    for attempt in range(5):
        response = requests.get(url, headers=HEADERS, timeout=30)
        time.sleep(0.25)                      # rythme testé sans refus (erreur 429 au-delà)
        if response.status_code == 200:
            items = [{"month": it["timestamp"][:6], "views": it["views"]} for it in response.json()["items"]]
            break
        if response.status_code == 404:      # page sans aucune vue enregistrée
            items = []
            break
        time.sleep(10 * 2 ** attempt)         # erreur 429 "trop de requêtes" : on patiente de plus en plus
    else:
        return None
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_file, "wt", encoding="utf-8") as f:
        json.dump(items, f)
    return items


def fetch_views(links, workers=1):
    # Télécharge les vues de toutes les pages, une à la fois : Wikimedia limite
    # fortement les programmes anonymes (erreur 429 dès 5 requêtes en parallèle).
    # Résultat : une ligne par joueur x langue x mois.
    tasks = [(row.player_id, l, getattr(row, l)) for row in links.itertuples()
             for l in LANGUAGES if isinstance(getattr(row, l, None), str)]

    def work(task):
        pid, lang, title = task
        items = _views_one(lang, title) or []
        return [{"player_id": pid, "lang": lang, "month": it["month"], "views": it["views"]} for it in items]

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(pool.map(work, tasks), total=len(tasks), desc="Pages vues"):
            rows += result
    views = pd.DataFrame(rows)
    views.to_parquet(VIEWS_PATH)
    return views


def build():
    links = fetch_links(target_players())
    print(f"{len(links)} joueurs trouvés sur Wikidata")
    views = fetch_views(links)
    print(f"{len(views):,} lignes joueur x langue x mois")


if __name__ == "__main__":
    build()
