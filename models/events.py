# =============================================================================
# models/events.py — Étape 8 : les événements de carrière.
#
# Ce qui peut arriver à un joueur en dehors de son "niveau" (géré par la
# chaîne de Markov cachée) :
#   1. TRANSFERT / PRÊT : probabilité de changer de club à chaque trimestre,
#      selon son état, son âge et la période (mercato d'été / d'hiver).
#      Et si il part : vers un club plus fort ou plus faible ? (écart d'Elo)
#   2. ÉVOLUTION DU CLUB : la force du club bouge d'un trimestre à l'autre.
#   3. SÉLECTION NATIONALE : probabilité de jouer en sélection, par état/âge.
#   4. DISCIPLINE : cartons rouges -> matchs de suspension.
#   5. CONTRAT : prolongation ou départ libre. AUCUNE source gratuite ne donne
#      l'historique des contrats : ces probabilités sont des HYPOTHÈSES,
#      affichées comme telles et modifiables ci-dessous.
# Tout est mesuré sur les vraies données (sauf le point 5).
# =============================================================================

import pickle

import numpy as np
import pandas as pd

import config
from collectors.transfermarkt import BIG5_CODES
from features import quarterly as qt
from models import hmm

MODEL_PATH = config.PROCESSED_DIR / "events.pkl"
K = len(config.STATES)
AGE_BINS = hmm.AGE_BINS

# --- HYPOTHÈSES (pas de données historiques de contrats) ---------------------
# Probabilité, par an, qu'un joueur à moins de 18 mois de la fin de son
# contrat prolonge, selon son état. Durée de la prolongation : 3 ans (4 ans
# avant 25 ans). Un contrat qui arrive à échéance = départ libre.
CONTRACT_ASSUMPTIONS = {
    "renewal_prob_per_year": {"Blessé": 0.35, "Marginal": 0.15, "Rotation": 0.30,
                              "Titulaire": 0.50, "Fort": 0.55, "Élite": 0.60},
    "renewal_years_young": 4, "renewal_years": 3, "young_age": 25,
    "new_contract_years_after_transfer": 4,
}
SUSPENSION_MATCHES_PER_RED = 1.5


def _age_bin(age):
    return np.digitize(age, AGE_BINS[1:-1])


def fit(cutoff_t=None, states=None):
    q = pd.read_parquet(qt.OUT_PATH)
    s = (states if states is not None else pd.read_parquet(hmm.STATES_PATH))[["player_id", "t", "state", "tier"]]
    df = q.merge(s, on=["player_id", "t"]).sort_values(["player_id", "t"])
    if cutoff_t is not None:
        df = df[df.t <= cutoff_t]
    g = df.groupby("player_id")
    df["prev_state"] = g.state.shift()
    df["prev_elo"] = g.own_elo.shift()
    df["next_elo"] = g.own_elo.shift(-1)
    df["age_bin"] = _age_bin(df.age.fillna(25))
    d = df[df.prev_state.notna()].copy()
    d["prev_state"] = d.prev_state.astype(int)

    # 1. Probabilité de transfert : table état x tranche d'âge x trimestre (lissée vers la moyenne)
    n_age = len(AGE_BINS) - 1
    overall = d.groupby("quarter").moved.mean()
    move = np.tile(overall.sort_index().to_numpy(), (K, n_age, 1))   # sans données = moyenne
    for (st, ab, qq), grp in d.groupby(["prev_state", "age_bin", "quarter"]):
        prior = overall[qq]
        move[st, ab, qq - 1] = (grp.moved.sum() + 20 * prior) / (len(grp) + 20)

    # Écart d'Elo à l'arrivée (club d'arrivée - club de départ), par état au moment du départ
    movers = d[d.moved & d.prev_elo.notna()]
    arrival = movers.own_elo.fillna(movers.next_elo)
    gap = (arrival - movers.prev_elo).dropna()
    elo_jump = {st: gap[movers.loc[gap.index, "prev_state"] == st].to_numpy() for st in range(K)}
    elo_jump = {st: (v if len(v) >= 30 else gap.to_numpy()) for st, v in elo_jump.items()}

    # 2. Variation trimestrielle de l'Elo du club sans transfert
    stay = d[~d.moved & d.prev_elo.notna() & d.own_elo.notna()]
    elo_sd = float((stay.own_elo - stay.prev_elo).std())

    # 3. Sélection nationale. Transfermarkt ne contient que les matchs de GRANDS
    #    TOURNOIS (Coupe du monde, Euro, Copa América...). On mesure donc la proba
    #    de jouer un tournoi, état x âge, sur les seuls trimestres de tournoi
    #    (ceux où au moins 1 % des joueurs ont joué en sélection).
    df["plays_national"] = df.minutes_national > 0
    share_by_quarter = df.groupby("t").plays_national.mean()
    tournament_t = share_by_quarter[share_by_quarter > 0.01].index
    tq = df[df.t.isin(tournament_t)]
    base = tq.plays_national.mean()
    national = np.full((K, n_age), base)        # combinaison sans données = moyenne générale
    for (st, ab), grp in tq.groupby(["state", "age_bin"]):
        national[int(st), ab] = (grp.plays_national.sum() + 20 * base) / (len(grp) + 20)

    # 4. Discipline : cartons rouges par 1 000 minutes (moyenne générale)
    red_rate = float(df.red.sum() / df.minutes.sum() * 1000)

    # 5. DESTINATION d'un transfert : rester dans le même championnat, rejoindre un
    #    autre grand championnat, ou quitter le top 5 (Arabie saoudite, MLS, Turquie,
    #    2e division...). Mesuré par tranche d'âge : à 31 ans, on part plus souvent hors top 5.
    big5 = set(BIG5_CODES)
    df["league_cat"] = np.where(df.league_id.isin(big5), df.league_id, "autre")
    df["next_cat"] = df.groupby("player_id").league_cat.shift(-2)       # 2 trimestres après le départ
    #    ... et selon le NIVEAU DE TALENT (une star part rarement hors du top 5).
    m = df[df.moved & df.league_cat.isin(big5) & df.next_cat.notna()]

    def shares(g):
        return np.array([(g.next_cat == g.league_cat).sum(),
                         (g.next_cat.isin(big5) & (g.next_cat != g.league_cat)).sum(),
                         (g.next_cat == "autre").sum()], dtype=float)

    base_dest = shares(m) / max(len(m), 1)
    dest = np.zeros((n_age, hmm.N_TIERS, 3))                             # [même, autre top 5, hors top 5]
    for ab in range(n_age):
        for tr in range(hmm.N_TIERS):
            counts = shares(m[(m.age_bin == ab) & (m.tier == tr)])
            dest[ab, tr] = (counts + 20 * base_dest) / (counts.sum() + 20)

    # Retour dans le top 5 d'un joueur parti hors top 5 (prêts, retours...), par âge
    out_movers = df[df.moved & (df.league_cat == "autre") & df.next_cat.notna()
                    & df.player_id.isin(df.loc[df.league_cat.isin(big5), "player_id"])]
    base_ret = out_movers.next_cat.isin(big5).mean() if len(out_movers) else 0.2
    comeback = np.array([(out_movers[out_movers.age_bin == ab].next_cat.isin(big5).sum() + 20 * base_ret)
                         / ((out_movers.age_bin == ab).sum() + 20) for ab in range(n_age)])

    # 6. Seuils de valeur des niveaux de talent (60e et 90e centiles par tranche
    #    d'âge, sur les 3 dernières saisons connues) : servent à recalculer le
    #    niveau de talent pendant la simulation à partir de la valeur simulée.
    recent = df[(df.season >= df.season.max() - 2) & (df.market_value > 0)]
    tier_cuts = np.array([recent[recent.age_bin == ab].market_value.quantile(hmm.TIER_CUTS).to_numpy()
                          if (recent.age_bin == ab).sum() > 50 else [np.inf, np.inf]
                          for ab in range(n_age)])

    return {"move": move, "elo_jump": elo_jump, "elo_sd": elo_sd, "national": national,
            "red_rate_per_1000": red_rate, "contract": CONTRACT_ASSUMPTIONS,
            "tournament_quarters": TOURNAMENTS, "destination": dest, "comeback": comeback,
            "tier_cuts": tier_cuts}


# Grands tournois à venir dans l'horizon, en (saison, trimestre) :
# Euro 2028 (juin-juillet 2028 -> saison 2027, T4), Coupe du monde 2030 (saison 2029, T4).
TOURNAMENTS = {(2027, 4): "Euro 2028", (2029, 4): "Coupe du monde 2030"}


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
    print(r["elo_sd"], r["red_rate_per_1000"])
