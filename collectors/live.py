# =============================================================================
# collectors/live.py — Récupération À LA DEMANDE des données récentes d'un joueur.
#
# Transfermarkt s'arrête en juillet 2026. Quand on analyse un joueur, on va
# chercher sur FotMob ses matchs de la saison en cours (2026/27), puis on met
# à jour son état de départ dans la chaîne de Markov cachée :
#     état connu en juillet 2026  --(transition)-->  état prédit pour T1 26/27
#     + observations réelles de T1 (minutes, blessures, performance)
#     = état mis à jour (c'est le "filtrage" : la même opération que l'étape
#       AVANT de l'algorithme avant-arrière, appliquée aux nouvelles données)
# =============================================================================

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

import config
from collectors import fotmob as fm
from collectors.matching import normalize
from models import hmm
from models import performance as pf

ALL_LEAGUES = {**config.LEAGUES, **config.SECOND_DIVISIONS}


def refresh_current_season(league_keys=None):
    # Télécharge la page du jour de chaque championnat (calendrier à jour) et
    # les matchs terminés pas encore en cache. Rapide : seuls les nouveaux matchs
    # sont téléchargés.
    for key in league_keys or ALL_LEAGUES:
        league = ALL_LEAGUES[key]
        for match_id in fm.list_matches(league["fotmob"], config.CURRENT_SEASON).match_id:
            fm.get_match(match_id)


def current_season(league_keys=None):
    # Toutes les stats FotMob de la saison en cours, toutes ligues confondues.
    frames = []
    for key in league_keys or ALL_LEAGUES:
        matches, players, _, lineups = fm.load_season(key, config.CURRENT_SEASON)
        if players.empty:
            continue
        dates = pd.to_datetime(matches.set_index("match_id")["date"]).dt.tz_localize(None)
        frames.append((matches.assign(league=key),
                       players.assign(date=players.match_id.map(dates), league=key),
                       lineups.assign(date=lineups.match_id.map(dates), league=key)))
    return (pd.concat([f[i] for f in frames], ignore_index=True) for i in range(3))


def find_fotmob_id(name, age, lineups):
    # Le joueur dans les compositions FotMob : nom très proche ET âge cohérent.
    cand = lineups.drop_duplicates("player_id")[["player_id", "player_name", "age"]].dropna(subset=["age"])
    target = normalize(name)
    sims = cand.player_name.map(lambda n: max(fuzz.token_set_ratio(target, normalize(n)),
                                              fuzz.ratio(target, normalize(n))) / 100)
    # FotMob arrondit l'âge à l'année inférieure : on compare à la partie entière
    ok = cand[(sims >= 0.9) & ((cand.age - np.floor(age)).abs() <= 1)]
    return int(ok.player_id.iloc[0]) if len(ok) == 1 else None


def quarter_observations(fm_id, matches, players, lineups, pos_group):
    # Observations du joueur par trimestre de la saison en cours :
    # part de temps de jeu, matchs manqués sur blessure, indice de performance.
    me = players[players.player_id == fm_id]
    if me.empty:
        return pd.DataFrame()
    team = me.team_id.mode().iloc[0]
    league = me.league.mode().iloc[0]
    tm_ = matches[(matches.league == league) & ((matches.home_id == team) | (matches.away_id == team))]
    tm_dates = pd.to_datetime(tm_.date).dt.tz_localize(None)
    quarter_of = {m: q for q, ms in config.QUARTERS.items() for m in ms}

    # Référence de performance : joueurs du top 5 avec >= 270 minutes cette saison, même famille de poste
    big5 = players[players.league.isin(config.LEAGUES)]
    rows = []
    for q in sorted(set(tm_dates.dt.month.map(quarter_of))):
        in_q = tm_dates.dt.month.map(quarter_of) == q
        team_games = int(in_q.sum())
        mine = me[me.date.dt.month.map(quarter_of) == q]
        minutes = float(pd.to_numeric(mine.minutes_played, errors="coerce").fillna(0).sum())
        absences = lineups[(lineups.player_id == fm_id) & (lineups.role == "unavailable")
                           & (lineups.date.dt.month.map(quarter_of) == q)]
        rows.append({"quarter": q, "team_games": team_games, "minutes": minutes,
                     "share": min(minutes / max(team_games * 90, 1), 1.0),
                     "missed_injury": int((absences.absence_type == "injury").sum()),
                     "matches": len(mine), "goals": float(mine.goals.fillna(0).sum()),
                     "xg": float(pd.to_numeric(mine.expected_goals_xg, errors="coerce").fillna(0).sum())})
    obs = pd.DataFrame(rows)
    # Même règle que dans la chaîne de Markov : un titulaire (>= 50 % du temps de
    # jeu au trimestre précédent) qui disparaît des feuilles de match (< 5 %) sans
    # avoir changé de club est très probablement blessé, même si FotMob ne le dit pas.
    prev_share = obs.share.shift().fillna(1.0)
    vanished = (obs.share < 0.05) & (prev_share >= 0.5)
    obs.loc[vanished, "missed_injury"] = obs.loc[vanished, ["missed_injury", "team_games"]].max(axis=1)
    obs["perf"] = _season_perf(fm_id, big5, me, pos_group)    # même indice sur toute la saison en cours
    return obs


def _season_perf(fm_id, big5, me, pos_group):
    # Indice de performance de la saison en cours (z-scores par rapport au top 5),
    # calculé avec les mêmes critères que l'étape 6.
    from features.quarterly import FOTMOB_SUMS
    cols = {k: v for k, v in FOTMOB_SUMS.items() if v in big5.columns}
    agg = (big5.rename(columns={v: k for k, v in cols.items()})
           .assign(**{k: lambda d, k=k: pd.to_numeric(d[k], errors="coerce").fillna(0) for k in cols})
           .groupby("player_id")[list(cols)].sum())
    agg["goals"] = 0
    agg["assists"] = 0
    agg["minutes"] = agg.fm_minutes
    metrics = pf.compute_metrics(agg)
    ref = agg.fm_minutes >= 270
    weights = pf.CRITERIA.get(pos_group, {})
    if fm_id not in agg.index or not weights:
        return np.nan
    total, wsum = 0.0, 0.0
    for metric, w in weights.items():
        if metric not in metrics:
            continue
        ref_values = metrics.loc[ref, metric]
        value = metrics.at[fm_id, metric]
        if pd.isna(value) or ref_values.std() == 0:
            continue
        z = np.clip((value - ref_values.mean()) / ref_values.std(), -3, 3)
        total += w * z
        wsum += abs(w)
    return total / wsum if wsum else np.nan


def filter_update(start_probs, obs, params, cls):
    # Met à jour la probabilité de chaque état avec les nouvelles observations,
    # trimestre par trimestre : prédiction (transition) puis correction (émission).
    probs = np.asarray(start_probs, dtype=float)
    for row in obs.itertuples():
        probs = probs @ params["trans"][cls]                                   # prédiction
        like = np.ones(len(probs))
        b = int(np.digitize(row.share, hmm.SHARE_BINS[1:-1]))
        like *= params["share"][:, b]                                           # temps de jeu
        if not np.isnan(row.perf) and row.minutes >= 90:
            w = min(row.minutes / hmm.FULL_WEIGHT_MINUTES, 1)
            sd = params["perf_sd"] / np.sqrt(w)
            like *= np.exp(-0.5 * ((row.perf - params["perf_mu"]) / sd) ** 2) / sd  # performance
        p_inj = params["injury"]
        like *= p_inj if row.missed_injury > 0 else (1 - p_inj)                 # blessure
        probs = probs * like
        probs = probs / probs.sum()                                             # correction
    return probs


def update_player(name, age, pos_group, start_probs, params, cls, refresh=True):
    # Point d'entrée : renvoie (probabilités d'état mises à jour, observations, trimestres ajoutés).
    if refresh:
        refresh_current_season()
    matches, players, lineups = current_season()
    fm_id = find_fotmob_id(name, age, lineups)
    if fm_id is None:
        return start_probs, pd.DataFrame(), 0
    obs = quarter_observations(fm_id, matches, players, lineups, pos_group)
    # On ne garde que les trimestres assez avancés (au moins 3 matchs de l'équipe)
    obs = obs[obs.team_games >= 3].sort_values("quarter", key=lambda q: q.map({1: 0, 2: 1, 3: 2, 4: 3}))
    if obs.empty:
        return start_probs, obs, 0
    return filter_update(start_probs, obs, params, cls), obs, len(obs)
