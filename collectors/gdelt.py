# =============================================================================
# collectors/gdelt.py — Contrôle presse : incidents hors terrain (GDELT).
#
# GDELT analyse en continu la presse mondiale et classe chaque article par
# THÈMES (arrestation, crime, drogue, agression...). Son API "DOC" donne, jour
# par jour depuis le 1er janvier 2017 et jusqu'à aujourd'hui, la part des
# articles du monde entier qui mentionnent un nom ET un de ces thèmes. Un pic
# signale un incident : pour Mason Greenwood, le pic tombe le 30 janvier 2022,
# jour de son arrestation.
#
# Utilisation : À LA DEMANDE, au moment où l'on analyse un joueur (Main.py) :
# deux requêtes seulement (la courbe 2017 -> aujourd'hui, puis les derniers
# articles), donc des données à jour au jour près.
#
# Pourquoi ce n'est PAS une variable du modèle : pour estimer l'effet d'un
# incident sur la valeur, il faudrait l'historique de TOUS les joueurs (~8 500
# requêtes). L'API gratuite limite à une requête toutes les 5 secondes et bride
# fortement au-delà (2 joueurs en 30 minutes lors de l'essai) : impossible à
# cette échelle sans passer par GDELT sur Google BigQuery. Sans cet historique,
# l'effet ne peut pas être validé sur le passé : le contrôle presse est donc
# AFFICHÉ dans la fiche, à titre d'information, sans modifier la simulation.
#
# Précautions :
#   - noms d'un seul mot ("Rodri", "Pedri") : trop ambigus, non interrogés ;
#   - un homonyme peut produire un faux pic : les dates et les titres sont
#     donnés pour que l'analyste vérifie lui-même.
#
#     python -m collectors.gdelt "Mason Greenwood"   -> test en ligne de commande
# =============================================================================

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API = "https://api.gdeltproject.org/api/v2/doc/doc"
START = datetime(2017, 1, 1, tzinfo=timezone.utc)     # début de l'historique de l'API DOC
THEMES = ["ARREST", "CRIME_COMMON_ROBBERY", "SEXUAL_ASSAULT", "TAX_FNCACT_POLICE",
          "DRUNK_DRIVING", "CRIME_ILLEGAL_DRUGS"]
HEADERS = {"User-Agent": "MarkovChainFootball/0.1 (https://github.com/Nat-404; personal research project) python-requests"}
PAUSE_SECONDS = 6            # l'API impose au plus une requête toutes les 5 secondes
MAX_WAIT_SECONDS = 150       # au-delà, on renonce (la fiche indique "contrôle indisponible")
PEAK_SHARE = 0.02            # pic notable : >= 0,02 % de tous les articles du monde ce jour-là
RECENT_DAYS = 90             # fenêtre "récente"


def query(name):
    themes = " OR ".join(f"theme:{t}" for t in THEMES)
    return f'"{name}" ({themes})'


def _get(params):
    # Une requête, avec patience en cas de refus "trop de requêtes" (429), bornée dans le temps.
    waited, delay = 0, 10
    while True:
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:              # réponse texte : requête jugée invalide par l'API
                    return {}
        except requests.RequestException:
            pass
        if waited + delay > MAX_WAIT_SECONDS:
            raise RuntimeError("GDELT ne répond pas (limite de requêtes)")
        time.sleep(delay)
        waited, delay = waited + delay, min(delay * 2, 60)


def press_check(name, today=None):
    # Contrôle presse d'un joueur : courbe quotidienne 2017 -> aujourd'hui, pics
    # notables, activité des 90 derniers jours et titres récents à vérifier.
    if len(name.split()) < 2:
        return {"status": "skipped", "reason": "single-word name, too ambiguous"}
    today = today or datetime.now(timezone.utc)
    fmt = "%Y%m%d%H%M%S"
    timeline = _get({"query": query(name), "mode": "timelinevol", "format": "json",
                     "startdatetime": START.strftime(fmt), "enddatetime": today.strftime(fmt)})
    days = ((timeline.get("timeline") or [{}])[0].get("data")) or []
    # (série vide si aucun article : index de dates explicite pour que les comparaisons restent valides)
    series = pd.Series([d["value"] for d in days], dtype=float,
                       index=pd.DatetimeIndex([pd.Timestamp(d["date"][:8]) for d in days]))
    recent_start = pd.Timestamp((today - timedelta(days=RECENT_DAYS)).date())
    peaks = series[series >= PEAK_SHARE].sort_values(ascending=False)
    # Pics séparés d'au moins 30 jours (un même incident produit plusieurs jours de couverture)
    distinct = []
    for day, value in peaks.items():
        if all(abs((day - d).days) > 30 for d, _ in distinct):
            distinct.append((day, float(value)))
    recent = series[series.index >= recent_start]

    # Titres récents : seulement s'il y a eu de la couverture ces 90 derniers jours
    # (une requête de moins dans le cas le plus courant, l'API étant lente)
    articles = []
    if (recent > 0).any():
        time.sleep(PAUSE_SECONDS)
        articles = _get({"query": query(name), "mode": "artlist", "format": "json", "maxrecords": 5,
                         "sort": "datedesc", "startdatetime": recent_start.strftime(fmt),
                         "enddatetime": today.strftime(fmt)}).get("articles") or []
    return {
        "status": "ok", "checked_on": today.date(), "days_covered": int(len(series)),
        "peaks": sorted(distinct[:5]),                      # (date, part des articles en %)
        "recent_days_with_coverage": int((recent > 0).sum()),
        "recent_max": float(recent.max()) if len(recent) else 0.0,
        "recent_peak": bool(len(recent) and recent.max() >= PEAK_SHARE),
        "headlines": [{"date": a.get("seendate", "")[:8], "title": a.get("title", ""), "domain": a.get("domain", "")}
                      for a in articles],
    }


def summary_en(check):
    # Une phrase (anglais) pour la fiche.
    if check is None:
        return None
    if check["status"] == "skipped":
        return "Press check (GDELT): not run, the player's name is a single word and too ambiguous."
    if check["status"] == "error":
        return "Press check (GDELT): unavailable at the time of the simulation (service not responding)."
    if not check["peaks"]:
        past = "no notable spike of incident-related press coverage since January 2017"
    else:
        dates = ", ".join(d.strftime("%b %Y") for d, _ in check["peaks"])
        past = f"spikes of incident-related press coverage in {dates}"
    n = check["recent_days_with_coverage"]
    if check["recent_peak"]:
        recent = f"a spike in the last {RECENT_DAYS} days ({n} days with coverage)"
    elif n:
        recent = f"no spike in the last {RECENT_DAYS} days ({n} days with minor coverage)"
    else:
        recent = f"no such coverage in the last {RECENT_DAYS} days"
    return (f"Press check (GDELT, {check['checked_on']:%d %b %Y}): {past}; {recent}. "
            "Information only, not an input of the simulation (its effect on value cannot be validated); "
            "a namesake can cause false spikes.")


def run(name):
    # Appel protégé pour Main.py : ne bloque jamais l'analyse.
    try:
        return press_check(name)
    except RuntimeError as error:
        return {"status": "error", "reason": str(error)}


if __name__ == "__main__":
    import sys
    result = run(" ".join(sys.argv[1:]) or "Mason Greenwood")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(summary_en(result))
