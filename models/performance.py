# =============================================================================
# models/performance.py — Étape 6 : indice de performance + facteur club.
#
# A. INDICE DE PERFORMANCE (par poste)
#    Chaque poste est jugé sur ses propres critères (un défenseur central n'est
#    pas noté sur ses xG). Pour chaque critère, on calcule une valeur "par 90
#    minutes", puis un z-score : combien d'écarts-types au-dessus ou en dessous
#    de la moyenne des joueurs du même poste, la même saison, dans les 5 grands
#    championnats. 0 = joueur moyen du poste ; +1 = nettement au-dessus ; +2 = élite.
#    L'indice est la moyenne pondérée des z-scores.
#    Sans données FotMob (avant 2018/19 en 1re division, 2021/22 en 2e division),
#    on se rabat sur les buts + passes décisives de Transfermarkt pour les
#    joueurs offensifs. Les stats xG n'existent qu'à partir de 2020/21 : avant,
#    l'indice repose sur les autres critères (passes, duels, actions défensives).
#    Critère collectif "on/off" (features/onoff.py : différence d'xG de l'équipe
#    avec lui sur le terrain moins sans lui) : calculé (onoff_90), mais NON
#    retenu dans l'indice. Testé en validation 2020 -> 2026 et 2023 -> 2026
#    (python -m evaluation.ablation onoff), il n'améliore pas les prévisions.
#
# B. FACTEUR CLUB (le contexte)
#    Un joueur d'une équipe dominante a des stats gonflées ; un joueur d'une
#    équipe faible, des stats écrasées. On estime un modèle à effets mixtes :
#        indice = effet propre du JOUEUR + b1 x force de son équipe
#                 + b2 x force des adversaires + effet du CHAMPIONNAT + bruit
#    Les joueurs qui changent de club permettent de séparer l'effet joueur de
#    l'effet équipe. On retire ensuite la part due au contexte : c'est
#    l'"indice ajusté", qui mesure le joueur lui-même.
# =============================================================================

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config
from features import quarterly as qt

OUT_PATH = config.PROCESSED_DIR / "performance.parquet"
CONTEXT_PATH = config.PROCESSED_DIR / "performance_context.json"   # coefficients du facteur club
MIN_MINUTES_REFERENCE = 270     # minutes minimales pour entrer dans la population de référence

# Regroupement des postes Transfermarkt en 6 familles
POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB", "Right-Back": "FB",
    "Defensive Midfield": "MID", "Central Midfield": "MID", "Left Midfield": "MID", "Right Midfield": "MID",
    "Attacking Midfield": "AMW", "Left Winger": "AMW", "Right Winger": "AMW",
    "Centre-Forward": "FW", "Second Striker": "FW",
}

# Critères et poids par famille de postes. Un poids négatif = moins c'est mieux.
# "per90" : divisé par les minutes ; les autres sont des pourcentages de réussite.
CRITERIA = {
    "GK":  {"goals_prevented_90": 0.6, "save_pct": 0.2, "pass_pct": 0.1, "long_pct": 0.1},
    "CB":  {"aerial_pct": 0.15, "tkl_int_90": 0.15, "clr_blk_90": 0.10, "dribbled_past_90": -0.10,
            "pass_pct": 0.20, "final_third_90": 0.15, "errors_90": -0.05, "xg_xa_90": 0.10},
    "FB":  {"xa_90": 0.20, "chances_90": 0.15, "final_third_90": 0.15, "dribbles_90": 0.10,
            "tkl_int_90": 0.15, "dribbled_past_90": -0.10, "pass_pct": 0.15},
    "MID": {"pass_pct": 0.15, "final_third_90": 0.15, "xa_90": 0.15, "chances_90": 0.10,
            "recoveries_90": 0.15, "tkl_int_90": 0.15, "duel_pct": 0.05, "npxg_90": 0.10},
    "AMW": {"npxg_90": 0.25, "xa_90": 0.25, "chances_90": 0.10, "dribbles_90": 0.15,
            "touches_box_90": 0.10, "recoveries_90": 0.05, "pass_pct": 0.10},
    "FW":  {"npxg_90": 0.40, "xa_90": 0.15, "shots_90": 0.10, "touches_box_90": 0.10,
            "big_chances_90": 0.10, "aerial_won_90": 0.05, "xgot_90": 0.10},
}


def compute_metrics(df):
    # Transforme les totaux du trimestre en taux par 90 minutes et en pourcentages.
    m = pd.DataFrame(index=df.index)
    per90 = 90 / df["fm_minutes"].where(df["fm_minutes"] > 0) if "fm_minutes" in df else None

    def get(col):
        return df[col] if col in df else pd.Series(np.nan, index=df.index)

    def ratio(a, b):
        return get(a) / get(b).where(get(b) > 0)

    # Les stats xG se rapportent aux seules minutes jouées dans des matchs avec xG
    per90_xg = 90 / get("xg_minutes").where(get("xg_minutes") > 0)
    if per90 is not None:
        m["goals_prevented_90"] = (get("xgot_faced") - get("goals_conceded_gk_xg")) * per90_xg
        m["save_pct"] = get("saves") / (get("saves") + get("goals_conceded_gk")).where(lambda s: s > 0)
        m["pass_pct"] = ratio("passes_acc", "passes_tot")
        m["long_pct"] = ratio("long_acc", "long_tot")
        m["aerial_pct"] = ratio("aerial_won", "aerial_tot")
        m["duel_pct"] = get("duels_won") / (get("duels_won") + get("duels_lost")).where(lambda s: s > 0)
        m["tkl_int_90"] = (get("tackles") + get("interceptions")) * per90
        m["clr_blk_90"] = (get("clearances") + get("blocks")) * per90
        m["dribbled_past_90"] = get("dribbled_past") * per90
        m["final_third_90"] = get("final_third_passes") * per90
        m["errors_90"] = get("errors_to_goal") * per90
        m["xg_xa_90"] = (get("xg") + get("xa")) * per90_xg
        m["xa_90"] = get("xa") * per90_xg
        m["npxg_90"] = get("npxg") * per90_xg
        m["xgot_90"] = get("xgot") * per90_xg
        m["chances_90"] = get("chances_created") * per90
        m["big_chances_90"] = get("big_chances_created") * per90
        m["dribbles_90"] = get("dribbles_acc") * per90
        m["touches_box_90"] = get("touches_box") * per90
        m["recoveries_90"] = get("recoveries") * per90
        m["shots_90"] = get("shots") * per90
        m["aerial_won_90"] = get("aerial_won") * per90
    if "oo_on_min" in df:
        from features import onoff
        m["onoff_90"] = onoff.onoff_xgd_90(df)
    # Repli Transfermarkt : buts + passes décisives par 90 minutes
    m["ga_90"] = (df["goals"] + df["assists"]) * 90 / df["minutes"].where(df["minutes"] > 0)
    return m


def _zscore_within(values, ref_mask, groups):
    # z-score de chaque valeur par rapport aux joueurs de référence du même groupe
    # (même famille de postes, même saison). Limité à +/-3 pour éviter qu'un
    # trimestre aberrant (2 matchs, 2 buts) n'écrase tout.
    ref = values.where(ref_mask)
    mean = ref.groupby(groups).transform("mean")
    std = ref.groupby(groups).transform("std")
    return ((values - mean) / std).clip(-3, 3)


def performance_index(df):
    df = df.copy()
    df["pos_group"] = df.sub_position.map(POSITION_GROUPS)
    metrics = compute_metrics(df)
    big5 = df.league_id.isin([l["transfermarkt"] for l in config.LEAGUES.values()])
    fm_ref = big5 & (df.get("fm_minutes", pd.Series(0, index=df.index)).fillna(0) >= MIN_MINUTES_REFERENCE)
    tm_ref = big5 & (df.minutes >= MIN_MINUTES_REFERENCE)
    groups = [df.pos_group, df.season]

    index = pd.Series(np.nan, index=df.index)
    for group, weights in CRITERIA.items():
        rows = df.pos_group == group
        total = pd.Series(0.0, index=df.index)
        weight_sum = pd.Series(0.0, index=df.index)
        for metric, w in weights.items():
            if metric not in metrics:
                continue
            z = _zscore_within(metrics[metric], fm_ref, groups)
            total = total + (w * z).where(z.notna() & rows, 0)
            weight_sum = weight_sum + pd.Series(abs(w), index=df.index).where(z.notna() & rows, 0)
        index[rows] = (total / weight_sum.where(weight_sum > 0))[rows]
    # Une moyenne de plusieurs z-scores a un écart-type INFÉRIEUR à 1 (environ 0,6 ici),
    # alors que l'indice de repli (buts + passes D) est un z-score simple (écart-type 1).
    # Sans correction, les saisons sans FotMob (2014-2017) paraîtraient deux fois plus
    # dispersées et la chaîne de Markov cachée prendrait cette dispersion pour des
    # états "Fort" / "Élite". On re-standardise donc l'indice composite : 0 = joueur
    # moyen du poste, 1 = un écart-type de la population de référence, comme le repli.
    index = _zscore_within(index, fm_ref, groups)
    df["perf_source"] = np.where(index.notna(), "fotmob", None)

    # Repli : buts + passes D (offensifs seulement ; pour les défenseurs et
    # gardiens, ce chiffre ne dit presque rien de leur niveau)
    offensive = df.pos_group.isin(["MID", "AMW", "FW"])
    fallback = _zscore_within(metrics["ga_90"], tm_ref, groups)
    use_fallback = index.isna() & offensive & (df.minutes > 0)
    index[use_fallback] = fallback[use_fallback]
    df.loc[use_fallback, "perf_source"] = "transfermarkt"

    df["perf"] = index
    # minutes qui "portent" l'indice (sert à savoir quelle confiance lui accorder)
    df["perf_minutes"] = np.where(df.perf_source == "fotmob", df.get("fm_minutes", 0), df.minutes)
    return df


def fit_context(df):
    # Modèle à effets mixtes : effet aléatoire par joueur (son niveau propre),
    # effets fixes pour la force de son équipe, des adversaires et le championnat.
    d = df[(df.perf.notna()) & (df.perf_minutes >= MIN_MINUTES_REFERENCE)
           & df.own_elo.notna() & df.opp_elo.notna()].copy()
    # Les championnats trop peu représentés (< 500 lignes) sont regroupés dans "autre" :
    # trop peu de données pour estimer leur effet de façon fiable.
    counts = d.league_id.value_counts()
    d["league"] = d.league_id.where(d.league_id.map(counts) >= 500, "autre").fillna("autre")
    d["own_c"] = (d.own_elo - 1600) / 100          # en centaines de points Elo, centré
    d["opp_c"] = (d.opp_elo - 1600) / 100
    # L'optimiseur essaie en chemin des points où la variance de l'effet joueur est
    # nulle, d'où un avertissement "covariance singulière" sans conséquence : on le
    # masque, mais on VÉRIFIE que l'estimation finale a convergé avec une variance > 0.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Random effects covariance is singular")
        model = smf.mixedlm("perf ~ own_c + opp_c + C(league)", d, groups=d["player_id"],
                            re_formula="1").fit(method=["lbfgs", "powell"], maxiter=500)
    if not model.converged or float(model.cov_re.iloc[0, 0]) <= 0:
        raise RuntimeError("Le modèle de contexte n'a pas convergé correctement")
    return model


def adjust(df, model):
    # Retire la part de l'indice due au contexte (équipe, adversaires, championnat).
    params = model.params
    own = ((df.own_elo - 1600) / 100).fillna(0)
    opp = ((df.opp_elo - 1600) / 100).fillna(0)
    league_effect = df.league_id.fillna("autre").map(
        lambda l: params.get(f"C(league)[T.{l}]", 0.0)).astype(float)
    # Référence = championnat de base du modèle (effet 0). On recentre sur la
    # moyenne des 5 grands championnats pour que 0 reste "joueur moyen du top 5".
    big5 = [l["transfermarkt"] for l in config.LEAGUES.values()]
    league_ref = np.mean([params.get(f"C(league)[T.{l}]", 0.0) for l in big5])
    context = params["own_c"] * own + params["opp_c"] * opp + (league_effect - league_ref)
    out = df.copy()
    out["perf_context"] = context.where(df.perf.notna())
    out["perf_adj"] = df.perf - out.perf_context
    return out


def build():
    df = pd.read_parquet(qt.OUT_PATH)
    df = performance_index(df)
    model = fit_context(df)
    df = adjust(df, model)
    keep = ["player_id", "season", "quarter", "t", "pos_group", "perf", "perf_adj", "perf_context",
            "perf_source", "perf_minutes"]
    df[keep].to_parquet(OUT_PATH)
    import json
    CONTEXT_PATH.write_text(json.dumps({"own": float(model.params["own_c"]), "opp": float(model.params["opp_c"]),
                                        "n": int(model.nobs)}))
    return df, model


if __name__ == "__main__":
    df, model = build()
    print(model.summary())
