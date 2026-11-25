# =============================================================================
# collectors/tournaments.py — Les grands tournois de sélections (FotMob).
#
# Problème : Transfermarkt liste bien les matchs des Coupes du monde, Euros,
# etc. depuis 2006, mais les FEUILLES DE MATCH des joueurs n'y figurent qu'à
# partir de 2025/26. Sans elles, "a joué avec sa sélection" ne serait connu
# que pour la Coupe du monde 2026 : la variable serait confondue avec l'année.
#
# Solution : on récupère sur FotMob les compositions de TOUS les grands
# tournois depuis 2014 (même source pour toutes les années = mesure cohérente) :
#   Coupe du monde, Euro, Copa América, Coupe d'Afrique, Coupe d'Asie.
# Puis on relie chaque joueur FotMob à son identifiant Transfermarkt :
#   1. par la table de correspondance déjà construite (matchs de club) ;
#   2. sinon par nom + NATIONALITÉ + année de naissance (l'âge donné par
#      FotMob est l'âge le jour du match). Un seul candidat exigé.
#
# Sortie : data/processed/tournaments.parquet, une ligne par joueur et par
# match de tournoi (date, minutes jouées).
# =============================================================================

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

import config
from collectors import fotmob as fm
from collectors import matching
from collectors import transfermarkt as tm

OUT_PATH = config.PROCESSED_DIR / "tournaments.parquet"

# Identifiant FotMob -> nom ; les éditions sont lues dans l'API (allAvailableSeasons)
TOURNAMENTS = {77: "Coupe du monde", 50: "Euro", 44: "Copa América", 289: "Coupe d'Afrique", 290: "Coupe d'Asie"}
FIRST_YEAR = config.FIRST_SEASON          # 2014 : début de la période étudiée

# Noms de pays qui diffèrent entre FotMob (équipe) et Transfermarkt (nationalité)
COUNTRY_ALIASES = {"usa": "united states", "korea republic": "korea south", "south korea": "korea south",
                   "ivory coast": "cote d ivoire", "dr congo": "dr congo", "congo dr": "dr congo",
                   "bosnia herzegovina": "bosnia herzegovina", "czechia": "czech republic",
                   "turkiye": "turkey", "cape verde": "cape verde", "ir iran": "iran"}


def editions():
    # Liste des (tournoi, édition) à collecter, de 2014 à aujourd'hui.
    out = []
    for league_id in TOURNAMENTS:
        data = fm._get_json("leagues", {"id": league_id}, fm.CACHE / "leagues" / f"{league_id}_editions.json.gz")
        for label in data.get("allAvailableSeasons", []):
            if FIRST_YEAR <= int(label) <= config.CURRENT_SEASON:
                out.append((league_id, label))
    return out


def edition_matches(league_id, label):
    # Matchs terminés d'une édition (même cache que les championnats).
    cache = fm.CACHE / "leagues" / f"{league_id}_{label}_final.json.gz"
    data = fm._get_json("leagues", {"id": league_id, "season": label}, cache)
    rows = []
    for m in data["fixtures"]["allMatches"]:
        if m["status"].get("finished") and not m["status"].get("cancelled"):
            rows.append({"match_id": int(m["id"]), "date": m["status"]["utcTime"]})
    return pd.DataFrame(rows)


def collect():
    # Télécharge toutes les compositions (avec reprise automatique via le cache).
    rows = []
    for league_id, label in tqdm(editions(), desc="Tournois"):
        matches = edition_matches(league_id, label)
        for m in tqdm(matches.itertuples(), total=len(matches), desc=f"{TOURNAMENTS[league_id]} {label}", leave=False):
            try:
                match = fm.get_match(m.match_id)
            except RuntimeError as error:
                tqdm.write(f"⚠ match {m.match_id} sauté ({error})")
                continue
            if match is None or "general" not in match:
                continue
            minutes = {p["player_id"]: p.get("minutes_played") for p in fm.parse_player_stats(match)}
            for p in fm.parse_lineups(match):
                if p["role"] == "unavailable":
                    continue
                rows.append({"match_id": m.match_id, "date": m.date, "tournament": TOURNAMENTS[league_id],
                             "edition": label, "fm_player_id": p["player_id"], "player_name": p["player_name"],
                             "age": p["age"], "team": _team_name(match, p["team_id"]),
                             "minutes": minutes.get(p["player_id"]) if minutes else lineup_minutes(p)})
        tqdm.write(f"✓ {TOURNAMENTS[league_id]} {label} : {len(matches)} matchs")
    df = pd.DataFrame(rows)
    df["minutes"] = pd.to_numeric(df.minutes, errors="coerce").fillna(0)
    return df


def lineup_minutes(p):
    # Pour les matchs anciens sans statistiques individuelles (Copa 2015, CAN 2015-2017,
    # Coupe d'Asie 2015-2019, une partie du Mondial 2014), on déduit les minutes de la
    # composition : un titulaire joue jusqu'à sa sortie (ou 90), un remplaçant depuis son entrée.
    out = p["sub_out_minute"] if p["sub_out_minute"] is not None else 90
    if p["role"] == "starters":
        return min(out, 90)
    if p["sub_in_minute"] is not None:
        return max(min(out, 90) - p["sub_in_minute"], 1)
    return 0


def _team_name(match, team_id):
    lineup = match["content"].get("lineup") or {}
    for side in ("homeTeam", "awayTeam"):
        if (lineup.get(side) or {}).get("id") == team_id:
            return lineup[side].get("name")
    return None


def _country(name):
    n = matching.normalize(name)
    return COUNTRY_ALIASES.get(n, n)


def link_players(apps):
    # Relie chaque joueur FotMob à Transfermarkt (correspondance existante, sinon nom+pays+âge).
    known = matching.fm_to_tm()
    apps["tm_player_id"] = apps.fm_player_id.map(known)
    # Année de naissance déduite de l'âge le jour du match (un même joueur peut avoir
    # un âge renseigné dans un tournoi et pas dans un autre : on prend la médiane).
    unknown = apps[apps.tm_player_id.isna()].assign(
        year=lambda d: pd.to_datetime(d.date).dt.year,
        birth_year=lambda d: pd.to_datetime(d.date).dt.year - d.age)
    todo = (unknown.groupby("fm_player_id")
            .agg(player_name=("player_name", "first"), team=("team", "first"),
                 birth_year=("birth_year", "median"), year=("year", "min")).reset_index())
    players = tm.query("SELECT player_id, name, date_of_birth, country_of_citizenship FROM players "
                       "WHERE date_of_birth IS NOT NULL")
    players["norm"] = players.name.map(matching.normalize)
    players["country"] = players.country_of_citizenship.fillna("").map(_country)
    players["birth_year"] = pd.to_datetime(players.date_of_birth).dt.year
    taken = set(known.values())
    found = {}
    for p in todo.itertuples():
        # Candidats : même nationalité, pas déjà reliés, nom très proche, et
        #   - âge connu : année de naissance à 1 an près (anniversaire avant/après le match) ;
        #   - âge inconnu (Euros 2016 et 2020) : âge plausible au tournoi (17-40 ans) et
        #     exigence plus forte sur le nom (95 % au lieu de 90 %).
        pool = players[(players.country == _country(p.team)) & ~players.player_id.isin(taken)]
        if p.birth_year == p.birth_year:
            pool, threshold = pool[(pool.birth_year - p.birth_year).abs() <= 1], 0.90
        else:
            pool, threshold = pool[(p.year - pool.birth_year).between(17, 40)], 0.95
        if pool.empty:
            continue
        norm = matching.normalize(p.player_name)
        sims = pool.norm.map(lambda n: max(fuzz.token_set_ratio(norm, n), fuzz.ratio(norm, n)) / 100)
        good = pool[sims >= threshold]
        if len(good) == 1:
            found[p.fm_player_id] = int(good.player_id.iloc[0])
    apps.loc[apps.tm_player_id.isna(), "tm_player_id"] = apps.fm_player_id.map(found)
    return apps


def build():
    apps = link_players(collect())
    apps["date"] = pd.to_datetime(apps.date).dt.tz_localize(None)
    apps.to_parquet(OUT_PATH)
    played = apps[apps.minutes > 0]
    print(f"✓ {played.match_id.nunique()} matchs de tournoi, {played.fm_player_id.nunique()} joueurs, "
          f"{played.tm_player_id.notna().mean():.0%} des apparitions reliées à Transfermarkt")
    return apps


def load():
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":
    build()
