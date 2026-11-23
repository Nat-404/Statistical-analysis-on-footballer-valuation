# =============================================================================
# models/valuation.py — Étape 9 : de l'état du joueur à sa valeur en euros.
#
# On explique le LOGARITHME de la valeur marchande Transfermarkt par :
#   - l'état du joueur (chaîne de Markov cachée) et sa performance récente
#   - son âge, selon sa famille de poste (un gardien de 31 ans vaut plus,
#     relativement, qu'un ailier de 31 ans)
#   - la force de son club (Elo) et son championnat
#   - la durée de contrat restante (connue seulement pour la dernière saison)
#   - sa popularité (pages vues Wikipédia), sa sélection nationale,
#     sa nationalité (UE ou non), son pied, sa taille
#   - l'ANNÉE : ce coefficient mesure l'INFLATION du marché (tous les prix
#     montent ou baissent ensemble). On le retire pour la "valeur sans inflation".
# Pourquoi le logarithme ? Les valeurs vont de 50 k€ à 200 M€ : en log, un
# même effet (ex. +1 an de contrat) agit en POURCENTAGE, ce qui est réaliste.
#
# Ce que le modèle n'explique pas (réputation, agent, "hype"...) est le
# RÉSIDU du joueur. Il est persistant mais s'estompe avec le temps : on mesure
# cette persistance (coefficient rho) pour la reproduire dans la simulation.
# =============================================================================

import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config
from features import quarterly as qt
from models import hmm
from models import performance as pf

MODEL_PATH = config.PROCESSED_DIR / "valuation.pkl"

EU = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
      "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
      "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
      "Spain", "Sweden"}
BIG5 = {l["transfermarkt"]: k for k, l in config.LEAGUES.items()}

FORMULA = (
    "log_value ~ p_Blessé + p_Rotation + p_Titulaire + p_Fort + p_Élite"
    " + perf_recent + share_recent"
    " + C(family) * (age_c + I(age_c**2) + I(age_c**3))"
    " + elo_c + C(league)"
    " + c_lt1 + c_1_2 + c_2_3"
    " + log_views + has_views + national + eu + left_foot + height_c"
    " + C(season)"
)
# Même formule, avec en plus la valeur d'il y a un an (log_lag)
DYNAMIC_FORMULA = FORMULA.replace("log_value ~ ", "log_value ~ log_lag + ")


def build_features(df):
    # Prépare toutes les variables explicatives à partir de la table trimestrielle
    # + états HMM. Les moyennes "récentes" portent sur les 4 derniers trimestres.
    df = df.sort_values(["player_id", "t"]).copy()
    g = df.groupby("player_id")
    df["perf_recent"] = g.perf_adj.transform(lambda s: s.rolling(4, min_periods=1).mean()).fillna(0)
    df["share_recent"] = g.minutes_share.transform(lambda s: s.rolling(4, min_periods=1).mean()).fillna(0)
    views = df["wiki_views"] if "wiki_views" in df else pd.Series(np.nan, index=df.index)
    df["log_views"] = np.log1p(views.groupby(df.player_id).transform(
        lambda s: s.rolling(4, min_periods=1).mean())).fillna(0)
    df["has_views"] = views.notna().astype(int)
    df["national"] = (g.minutes_national.transform(lambda s: s.rolling(8, min_periods=1).sum()) > 0).astype(int)
    df["family"] = df.pos_group.map({"GK": "GK", "CB": "DEF", "FB": "DEF", "MID": "MID", "AMW": "ATT", "FW": "ATT"})
    df["age_c"] = df.age - 26
    df["elo_c"] = (df.own_elo.fillna(g.own_elo.ffill()).fillna(1450) - 1600) / 100
    minutes_d2 = df["minutes_d2"] if "minutes_d2" in df else pd.Series(0, index=df.index)
    other = pd.Series(np.where(minutes_d2.fillna(0) > 0, "D2", "autre"), index=df.index)
    df["league"] = df.league_id.map(BIG5).astype(object).where(df.league_id.isin(BIG5), other)
    df["eu"] = df.nationality.isin(EU).astype(int)
    df["left_foot"] = (df.foot == "left").astype(int)
    df["height_c"] = ((df.height.fillna(181) - 181) / 10) * df.family.isin(["GK", "DEF"])
    # Contrat : seule la date de fin ACTUELLE est connue -> utilisable pour la
    # dernière saison seulement. Piège : "contrat connu" = "saison 2025", ce qui
    # se confondrait avec l'effet année (inflation). On CENTRE donc les
    # indicateurs de contrat sur la saison récente : ils mesurent l'écart entre
    # contrats courts et longs SANS déplacer le niveau moyen de l'année.
    years = (pd.to_datetime(df.contract_expiration_date) - df.q_end).dt.days / 365.25
    df["contract_years"] = years
    recent = (df.season >= config.CURRENT_SEASON - 1) & years.notna()
    for col, (lo, hi) in CONTRACT_BINS.items():
        raw = ((years > lo) & (years <= hi)).astype(float)
        df[col] = np.where(recent, raw - raw[recent].mean(), 0.0)
    return df


# Tranches de contrat restant (années) ; la référence implicite est "plus de 3 ans"
CONTRACT_BINS = {"c_lt1": (-99, 1), "c_1_2": (1, 2), "c_2_3": (2, 3)}


def contract_dummies(years, centering):
    # Indicateurs de contrat pour la simulation, centrés comme à l'entraînement.
    return {col: float(lo < years <= hi) - centering[col] for col, (lo, hi) in CONTRACT_BINS.items()}


def load_training(cutoff_t=None):
    q = pd.read_parquet(qt.OUT_PATH)
    p = pd.read_parquet(pf.OUT_PATH)[["player_id", "t", "pos_group", "perf_adj"]]
    s = pd.read_parquet(hmm.STATES_PATH)[["player_id", "t"] + [f"p_{n}" for n in config.STATES]]
    df = q.merge(p, on=["player_id", "t"]).merge(s, on=["player_id", "t"])
    if cutoff_t is not None:
        df = df[df.t <= cutoff_t]
    df = build_features(df)
    # Valeur "fraîche" : publiée pendant le trimestre (sinon c'est une valeur ancienne)
    df["fresh"] = (df.market_value > 0) & (pd.to_datetime(df.value_date) >= df.q_start)
    df["log_value"] = np.log(df.market_value.where(df.market_value > 0))
    return df


def fit(cutoff_t=None):
    df = load_training(cutoff_t)
    train = df[df.fresh & df.family.notna()].copy()
    train["season"] = train.season.astype(str)
    model = smf.ols(FORMULA, train).fit()
    # Moyennes de centrage des indicateurs de contrat (à réappliquer en simulation)
    recent = (df.season >= config.CURRENT_SEASON - 1) & df.contract_years.notna()
    centering = {col: float(((df.contract_years > lo) & (df.contract_years <= hi))[recent].mean())
                 for col, (lo, hi) in CONTRACT_BINS.items()}

    # MODÈLE DYNAMIQUE (celui utilisé par la simulation).
    # Le modèle "en niveau" ci-dessus explique mal le prix des stars (il lui
    # manque leur talent, que les données ne mesurent qu'en partie). Le marché,
    # lui, part du prix ACTUEL et l'ajuste. On fait pareil :
    #     log(valeur dans 1 an) = a + b x log(valeur aujourd'hui) + effets des
    #                             caractéristiques dans 1 an (état, perf, âge...)
    # b proche de 1 = forte persistance ; (1 - b) = retour progressif vers la
    # valeur "expliquée" par le profil du joueur.
    df = df.sort_values(["player_id", "t"])
    lag = df[["player_id", "t", "market_value"]].assign(t=df.t + 4).rename(columns={"market_value": "lag_value"})
    dyn = df.merge(lag, on=["player_id", "t"], how="left")
    dyn = dyn[dyn.fresh & dyn.family.notna() & (dyn.lag_value > 0)].copy()
    dyn["log_lag"] = np.log(dyn.lag_value)
    dyn["season"] = dyn.season.astype(str)
    dynamic = smf.ols(DYNAMIC_FORMULA, dyn).fit()
    # Écart entre la valeur réelle et le modèle dynamique : persiste-t-il d'une année sur l'autre ?
    dyn["resid"] = dynamic.resid
    nxt = dyn[["player_id", "t", "resid"]].assign(t=dyn.t - 4)
    pairs = dyn.merge(nxt, on=["player_id", "t"], suffixes=("", "_next"))
    resid_persistence = float(np.polyfit(pairs.resid, pairs.resid_next, 1)[0])

    # Indice d'inflation : coefficient de chaque saison (par rapport à la 1re)
    seasons = sorted(train.season.unique())
    index = {int(s): model.params.get(f"C(season)[T.{s}]", 0.0) for s in seasons}
    # Tendance annuelle moyenne sur TOUTE la période (plus stable qu'une fenêtre courte)
    steps = np.diff([index[s] for s in sorted(index)])
    inflation = {"index": index, "drift": float(np.mean(steps)), "vol": float(np.std(steps))}
    return {"model": model, "dynamic": dynamic, "annual_sd": float(np.sqrt(dynamic.scale)),
            "resid_persistence": resid_persistence, "inflation": inflation,
            "contract_centering": centering, "last_season": int(max(seasons))}


def build():
    result = fit()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


def load():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    r = build()
    print(r["model"].rsquared, r["rho"], r["inflation"])
