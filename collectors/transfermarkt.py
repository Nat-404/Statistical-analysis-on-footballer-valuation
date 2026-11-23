# =============================================================================
# collectors/transfermarkt.py — Étape 1 : la base Transfermarkt.
#
# Source : projet open source "transfermarkt-datasets" (github.com/dcaribou).
# C'est une base DuckDB : UN seul fichier qui contient 12 tables reliées entre
# elles par des identifiants (player_id, club_id, game_id...).
#
# ⚠ La base est figée au 6 juillet 2026 (le projet a cessé ses mises à jour).
#    L'historique est complet, mais la saison 2026/27 n'y est pas.
# =============================================================================

import duckdb                      # moteur SQL qui lit directement le fichier .duckdb
import pandas as pd                # pour récupérer les résultats sous forme de tableaux
import requests                    # pour télécharger le fichier
from tqdm import tqdm              # barre de progression

import config

# Adresse officielle du fichier (donnée dans le README du projet)
DUCKDB_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"
DUCKDB_PATH = config.RAW_DIR / "transfermarkt.duckdb"   # où on le range chez nous

# Codes Transfermarkt de nos 5 championnats, lus depuis config.py -> ["GB1", "ES1", ...]
BIG5_CODES = [league["transfermarkt"] for league in config.LEAGUES.values()]


def download(force=False):
    # Télécharge la base si on ne l'a pas déjà (sauf si force=True).
    # Ça évite de re-télécharger 210 Mo à chaque exécution.
    if DUCKDB_PATH.exists() and not force:
        print(f"Base déjà présente : {DUCKDB_PATH}")
        return DUCKDB_PATH

    # stream=True : on reçoit le fichier par morceaux au lieu de tout charger en mémoire
    response = requests.get(DUCKDB_URL, stream=True, timeout=60)
    response.raise_for_status()  # stoppe avec une erreur claire si le serveur répond mal

    total = int(response.headers.get("Content-Length", 0))  # taille totale, pour la barre
    tmp_path = DUCKDB_PATH.with_suffix(".part")              # fichier temporaire pendant le téléchargement

    with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Transfermarkt") as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):  # morceaux de 1 Mo
            f.write(chunk)
            bar.update(len(chunk))

    # On ne renomme qu'une fois le téléchargement fini : un fichier coupé
    # en plein milieu ne sera jamais pris pour une base valide.
    tmp_path.rename(DUCKDB_PATH)
    return DUCKDB_PATH


def connect():
    # Ouvre la base en LECTURE SEULE : impossible de modifier les données brutes par erreur.
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


def query(sql, params=None):
    # Exécute une requête SQL et renvoie le résultat sous forme de tableau pandas.
    # "with" ferme la connexion automatiquement à la fin.
    with connect() as con:
        return con.execute(sql, params or []).df()


def list_tables():
    # Liste les tables de la base avec leur nombre de lignes.
    with connect() as con:
        names = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        rows = [(name, con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in names]
    return pd.DataFrame(rows, columns=["table", "lignes"])


def describe(table):
    # Liste les colonnes d'une table et leur type (texte, nombre, date...).
    return query(f"DESCRIBE {table}")[["column_name", "column_type"]]


def find_player(name):
    # Cherche un joueur par son nom (insensible aux majuscules et aux accents grâce à strip_accents).
    # Renvoie les principaux champs de sa fiche.
    return query(
        """
        SELECT player_id, name, date_of_birth, country_of_citizenship, position, sub_position,
               foot, height_in_cm, current_club_name, contract_expiration_date,
               market_value_in_eur, highest_market_value_in_eur, international_caps, last_season
        FROM players
        WHERE strip_accents(lower(name)) LIKE '%' || strip_accents(lower(?)) || '%'
        ORDER BY market_value_in_eur DESC NULLS LAST
        """,
        [name],
    )


def player_valuations(player_id):
    # Historique complet des valeurs marchandes d'un joueur, de la plus ancienne à la plus récente.
    return query(
        "SELECT date, market_value_in_eur FROM player_valuations WHERE player_id = ? ORDER BY date",
        [player_id],
    )


def player_seasons(player_id):
    # Résumé saison par saison d'un joueur, toutes compétitions confondues :
    # matchs, minutes, buts, passes décisives, cartons.
    # On relie "appearances" (une ligne par match joué) à "games" (pour connaître la saison).
    return query(
        """
        SELECT g.season,
               a.player_club_id                 AS club_id,
               COUNT(*)                         AS matchs,
               SUM(a.minutes_played)            AS minutes,
               SUM(a.goals)                     AS buts,
               SUM(a.assists)                   AS passes_d,
               SUM(a.yellow_cards)              AS jaunes,
               SUM(a.red_cards)                 AS rouges
        FROM appearances a
        JOIN games g USING (game_id)
        WHERE a.player_id = ?
        GROUP BY g.season, a.player_club_id
        ORDER BY g.season
        """,
        [player_id],
    )


# Si on lance ce fichier directement (python -m collectors.transfermarkt), il télécharge la base.
if __name__ == "__main__":
    download()
    print(list_tables())
