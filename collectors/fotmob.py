# =============================================================================
# collectors/fotmob.py — Étape 2 : statistiques avancées par match (FotMob / Opta).
#
# FotMob remplace Understat (hors ligne depuis novembre 2026). Ses données
# viennent d'Opta, comme l'ancien FBref. Pour CHAQUE match, on récupère :
#   - les stats de chaque joueur : minutes, xG, xA, passes, duels, tacles,
#     arrêts, buts évités (xGOT concédés - buts), note FotMob...
#   - chaque tir : minute, xG, xG cadré, situation (jeu, penalty, corner...)
#   - la composition : titulaires, remplaçants (minute d'entrée), absents
#     (blessure / suspension) -> une source de blessures match par match !
#
# Disponibilité vérifiée : stats de base depuis 2015/16, xG depuis 2018/19,
# 5 grands championnats + leurs 2es divisions.
#
# Principe du cache : chaque réponse est enregistrée sur le disque (compressée).
# Si on relance, on relit le fichier au lieu de redemander au site.
# Le téléchargement peut donc être interrompu et repris sans rien perdre.
# =============================================================================

import gzip                         # compression des fichiers en cache (~5x plus petits)
import json                         # lecture / écriture du format JSON (le format des réponses FotMob)
import re                           # expressions régulières, pour nettoyer les noms de stats
import time                         # pour attendre entre deux requêtes
import unicodedata                  # pour retirer les accents des noms de stats

import pandas as pd
import requests
from tqdm import tqdm

import config

API = "https://www.fotmob.com/api/data"                 # adresse de l'API (vérifiée à l'étape 2)
HEADERS = {"User-Agent": "Mozilla/5.0"}                 # on se présente comme un navigateur classique
CACHE = config.CACHE_DIR / "fotmob"                     # dossier du cache FotMob
OUT = config.PROCESSED_DIR / "fotmob"                   # dossier des tableaux nettoyés

# Téléchargement massif limité aux saisons récentes (le reste vient de Transfermarkt) :
# ~26 000 matchs au lieu de ~39 000. Les matchs récents du joueur analysé sont,
# eux, récupérés à la demande au moment de l'analyse.
# Vérifié après collecte : les statistiques de match (passes, duels, défense...)
# existent dès 2018/19, mais les xG (carte des tirs) seulement à partir de
# 2020/21 en 1re division, 2021/22 en D2 anglaise et allemande, 2025/26 pour les
# autres D2. Avant, les colonnes xG sont traitées comme inconnues (quarterly.py).
FIRST_SEASON_D1 = 2018
FIRST_SEASON_D2 = 2021   # 2es divisions : 2021/22 et après, pour limiter le volume

# Une session réutilise la même connexion pour toutes les requêtes (plus rapide)
_session = requests.Session()
_session.headers.update(HEADERS)
_last_request = 0.0      # moment de la dernière requête, pour respecter le délai


# -----------------------------------------------------------------------------
# 1. Téléchargement avec cache
# -----------------------------------------------------------------------------
def _get_json(endpoint, params, cache_file):
    # Renvoie la réponse JSON : depuis le cache si elle existe, sinon depuis FotMob.
    global _last_request
    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            return json.load(f)

    # Politesse : on attend que le délai minimum soit écoulé depuis la dernière requête
    wait = config.REQUEST_DELAY_SECONDS - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)

    # Jusqu'à 4 tentatives, en attendant de plus en plus longtemps (5 s, 10 s, 20 s)
    # si le serveur refuse ou ne répond pas : c'est le "backoff exponentiel".
    for attempt in range(4):
        try:
            response = _session.get(f"{API}/{endpoint}", params=params, timeout=30)
            _last_request = time.time()
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and data.get("error"):
                    return None          # ex. « Data not found » : on ne met pas une erreur en cache
                break
            if response.status_code == 404:   # la ressource n'existe pas : inutile d'insister
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * 2 ** attempt)
    else:
        # Les 4 tentatives ont échoué : on signale l'erreur sans écrire de cache
        raise RuntimeError(f"FotMob ne répond pas : {endpoint} {params}")

    # Écriture dans un fichier temporaire puis renommage : un fichier
    # coupé en plein milieu ne sera jamais pris pour un cache valide.
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.rename(cache_file)
    return data


def season_label(season):
    # Notre convention (2019) -> convention FotMob ("2019/2020")
    return f"{season}/{season + 1}"


def get_league_season(league_id, season):
    # Page d'un championnat pour une saison : calendrier complet des matchs.
    # La saison en cours n'est jamais mise en cache définitivement (elle évolue),
    # on la stocke sous un nom daté du jour.
    suffix = time.strftime("%Y%m%d") if season >= config.CURRENT_SEASON else "final"
    cache_file = CACHE / "leagues" / f"{league_id}_{season}_{suffix}.json.gz"
    return _get_json("leagues", {"id": league_id, "season": season_label(season)}, cache_file)


def get_match(match_id):
    # Détail complet d'un match (stats joueurs, tirs, compositions).
    # Rangé dans un sous-dossier par tranche de 1000 pour éviter un dossier géant.
    cache_file = CACHE / "matches" / str(int(match_id) // 1000) / f"{match_id}.json.gz"
    return _get_json("matchDetails", {"matchId": match_id}, cache_file)


def list_matches(league_id, season):
    # Liste des matchs TERMINÉS d'un championnat pour une saison, sous forme de tableau.
    data = get_league_season(league_id, season)
    rows = []
    for m in data["fixtures"]["allMatches"]:
        status = m["status"]
        if status.get("finished") and not status.get("cancelled"):
            rows.append({
                "match_id": int(m["id"]),
                "date": status["utcTime"],
                "round": m.get("round"),
                "home_id": int(m["home"]["id"]), "home": m["home"]["name"],
                "away_id": int(m["away"]["id"]), "away": m["away"]["name"],
                "score": status.get("scoreStr"),
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 2. Transformation d'un match en tableaux
# -----------------------------------------------------------------------------
def _slug(label):
    # "Expected goals (xG)" -> "expected_goals_xg" : un nom de colonne propre.
    text = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_player_stats(match):
    # Une ligne par joueur ayant joué, avec toutes ses statistiques du match.
    general, content = match["general"], match["content"]
    rows = []
    for player in (content.get("playerStats") or {}).values():
        row = {
            "match_id": int(general["matchId"]),
            "player_id": player["id"],
            "player_name": player["name"],
            "team_id": player["teamId"],
            "is_goalkeeper": player.get("isGoalkeeper", False),
            "position_id": player.get("positionId"),   # poste occupé CE match (polyvalence)
        }
        # Les stats sont rangées par groupes ("Top stats", "Attack", "Defense"...).
        # On les met toutes à plat. Certaines ont une valeur ET un total
        # (ex. passes réussies 22 sur 26) : on garde les deux.
        for group in player.get("stats", []):
            for label, item in group.get("stats", {}).items():
                stat = item.get("stat") or {}
                key = _slug(label)
                if key == "shotmap":          # c'est un graphique, pas une stat
                    continue
                row[key] = stat.get("value")
                if stat.get("total") is not None:
                    row[f"{key}_total"] = stat["total"]
        rows.append(row)
    return rows


def parse_shots(match):
    # Une ligne par tir, avec sa minute (utile pour le "xG sur le terrain").
    match_id = int(match["general"]["matchId"])
    keep = ["playerId", "teamId", "min", "minAdded", "eventType", "expectedGoals",
            "expectedGoalsOnTarget", "situation", "shotType", "isOnTarget", "isOwnGoal",
            "isFromInsideBox", "period", "x", "y"]
    shots = (match["content"].get("shotmap") or {}).get("shots") or []
    return [{"match_id": match_id, **{k: s.get(k) for k in keep}} for s in shots]


def parse_lineups(match):
    # Une ligne par joueur de la feuille de match : titulaire, remplaçant ou absent.
    # Pour les remplaçants on note la minute d'entrée ; pour les absents, la raison.
    match_id = int(match["general"]["matchId"])
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = (match["content"].get("lineup") or {}).get(side) or {}
        for role in ("starters", "subs", "unavailable"):
            for p in team.get(role) or []:
                perf = p.get("performance") or {}
                sub_events = perf.get("substitutionEvents") or []
                unavailable = p.get("unavailability") or {}
                rows.append({
                    "match_id": match_id,
                    "team_id": team.get("id"),
                    "player_id": p.get("id"),
                    "player_name": p.get("name"),
                    "role": role,                                   # starters / subs / unavailable
                    "age": p.get("age"),
                    "country": p.get("countryCode"),
                    "market_value_fotmob": p.get("marketValue"),
                    "sub_in_minute": next((e["time"] for e in sub_events if e.get("type") == "subIn"), None),
                    "sub_out_minute": next((e["time"] for e in sub_events if e.get("type") == "subOut"), None),
                    "absence_type": unavailable.get("type"),        # "injury", "suspension"...
                    "absence_id": unavailable.get("injuryId"),
                    "expected_return": unavailable.get("expectedReturn"),
                })
    return rows


# -----------------------------------------------------------------------------
# 3. Collecte d'une saison complète
# -----------------------------------------------------------------------------
def collect_league_season(league_key, league_id, season):
    # Télécharge tous les matchs d'un championnat / saison, puis enregistre
    # 4 tableaux parquet : matchs, stats joueurs, tirs, compositions.
    matches = list_matches(league_id, season)
    players, shots, lineups = [], [], []
    for match_id in tqdm(matches["match_id"], desc=f"{league_key} {season_label(season)}", leave=False):
        try:
            match = get_match(match_id)
        except RuntimeError as error:
            # Panne passagère : on note le match et on continue. Comme il n'est pas
            # en cache, il sera retenté automatiquement au prochain lancement.
            tqdm.write(f"⚠ match {match_id} sauté ({error})")
            continue
        if match is None or "general" not in match:   # match sans données chez FotMob
            continue
        players += parse_player_stats(match)
        shots += parse_shots(match)
        lineups += parse_lineups(match)

    folder = OUT / league_key
    folder.mkdir(parents=True, exist_ok=True)
    matches.assign(league=league_key, season=season).to_parquet(folder / f"{season}_matches.parquet")
    pd.DataFrame(players).to_parquet(folder / f"{season}_players.parquet")
    pd.DataFrame(shots).to_parquet(folder / f"{season}_shots.parquet")
    pd.DataFrame(lineups).to_parquet(folder / f"{season}_lineups.parquet")
    return len(matches)


def load_season(league_key, season):
    # Relit une saison : depuis les fichiers parquet si la saison est complète,
    # sinon directement depuis les matchs déjà en cache (saison en cours de
    # téléchargement). Renvoie 4 tableaux : matchs, joueurs, tirs, compositions.
    folder = OUT / league_key
    if (folder / f"{season}_players.parquet").exists():
        return tuple(pd.read_parquet(folder / f"{season}_{name}.parquet")
                     for name in ("matches", "players", "shots", "lineups"))
    league = {**config.LEAGUES, **config.SECOND_DIVISIONS}[league_key]
    matches = list_matches(league["fotmob"], season)
    players, shots, lineups, kept = [], [], [], []
    for match_id in matches["match_id"]:
        cache_file = CACHE / "matches" / str(int(match_id) // 1000) / f"{match_id}.json.gz"
        if not cache_file.exists():          # pas encore téléchargé : on le saute
            continue
        match = get_match(match_id)
        players += parse_player_stats(match)
        shots += parse_shots(match)
        lineups += parse_lineups(match)
        kept.append(match_id)
    matches = matches[matches.match_id.isin(kept)].assign(league=league_key, season=season)
    return matches, pd.DataFrame(players), pd.DataFrame(shots), pd.DataFrame(lineups)


def all_targets():
    # Liste de tous les (championnat, saison) à collecter.
    # On parcourt les saisons de la plus récente à la plus ancienne :
    # les données les plus utiles arrivent en premier.
    targets = []
    for season in range(config.CURRENT_SEASON, FIRST_SEASON_D1 - 1, -1):
        for key, league in config.LEAGUES.items():
            targets.append((key, league["fotmob"], season))
        if season >= FIRST_SEASON_D2:
            for key, league in config.SECOND_DIVISIONS.items():
                targets.append((key, league["fotmob"], season))
    return targets


def collect_all():
    # Collecte complète. Peut être interrompue (Ctrl+C) et relancée :
    # tout ce qui est déjà en cache est relu instantanément.
    for key, league_id, season in tqdm(all_targets(), desc="FotMob"):
        n = collect_league_season(key, league_id, season)
        tqdm.write(f"✓ {key} {season_label(season)} : {n} matchs")


# python -m collectors.fotmob            -> collecte complète
# python -m collectors.fotmob FRA 2023   -> une seule saison (pour tester)
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        key, season = sys.argv[1], int(sys.argv[2])
        league = {**config.LEAGUES, **config.SECOND_DIVISIONS}[key]
        print(collect_league_season(key, league["fotmob"], season), "matchs")
    else:
        collect_all()
