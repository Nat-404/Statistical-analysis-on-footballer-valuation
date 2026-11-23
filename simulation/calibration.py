# =============================================================================
# simulation/calibration.py — Calibration de la dérive de la simulation.
#
# La validation sur le passé a montré que la simulation est trop optimiste
# d'environ 6 à 12 % par an, et que l'écart s'accumule. Le modèle de valeur est
# juste quand on lui donne les VRAIES caractéristiques du joueur ; ce sont les
# caractéristiques SIMULÉES (états, temps de jeu...) qui dérivent.
#
# Correction : sur des centaines de cas RÉELS de la période d'entraînement, on
# simule 1 an et on compare à ce qui s'est vraiment passé. L'écart moyen, par
# tranche d'âge et niveau de talent, est la "dérive" à corriger : on l'ajoute
# à chaque trimestre simulé. Cette correction n'utilise que des données
# antérieures à la date de coupure (aucune information du futur).
# =============================================================================

import pickle

import numpy as np
import pandas as pd

import config
from models import hmm
from simulation import monte_carlo as mc

CALIBRATION_PATH = config.PROCESSED_DIR / "calibration.pkl"


def calibrate(ctx, cutoff_t, n_rows=900, n_sims=300, seed=0):
    rng = np.random.default_rng(seed)
    panel = ctx.panel
    big5 = list(config.LEAGUES)
    # Cas réels : valeur connue à t et un an plus tard, les deux avant la coupure
    base = panel[(panel.t <= cutoff_t - 4) & (panel.market_value > 0) & panel.league.isin(big5)
                 & panel.age.between(17, 36) & panel.tier.notna()][["player_id", "t", "age", "tier", "market_value"]]
    later = panel[["player_id", "t", "market_value"]].assign(t=panel.t - 4).rename(columns={"market_value": "v4"})
    cases = base.merge(later, on=["player_id", "t"])
    cases = cases[cases.v4 > 0]
    cases["age_bin"] = np.digitize(cases.age, hmm.AGE_BINS[1:-1])
    # Échantillon équilibré entre tranches d'âge et niveaux de talent
    cases = pd.concat([g.sample(min(len(g), n_rows // 15), random_state=seed)
                       for _, g in cases.groupby(["age_bin", "tier"])])

    # Variation "réelle" hors inflation : on retire l'évolution du marché sur l'année
    index = ctx.val["inflation"]["index"]
    last_season = max(index)

    def market_change(t):
        s0, s1 = t // 4, (t + 4) // 4
        return index.get(min(s1, last_season), 0) - index.get(min(s0, last_season), 0)

    rows = []
    for c in cases.itertuples():
        sim = mc.simulate(c.player_id, ctx, n=n_sims, steps=4, origin_t=c.t, resid_scale=1.0,
                          seed=int(rng.integers(1e9)), drift=False)
        simulated = float(np.median(np.log(sim["value_real"][:, 3] / c.market_value)))
        actual = float(np.log(c.v4 / c.market_value)) - market_change(c.t)
        rows.append({"age_bin": c.age_bin, "tier": int(c.tier), "gap": actual - simulated})
    gaps = pd.DataFrame(rows)

    # Table de dérive annuelle (âge x talent), lissée vers la moyenne de l'âge
    drift = np.zeros((len(hmm.AGE_BINS) - 1, hmm.N_TIERS))
    overall = gaps.gap.median()
    for ab in range(drift.shape[0]):
        g_age = gaps[gaps.age_bin == ab]
        age_mean = (g_age.gap.sum() + 10 * overall) / (len(g_age) + 10)
        for tr in range(hmm.N_TIERS):
            g = g_age[g_age.tier == tr].gap
            drift[ab, tr] = (g.sum() + 10 * age_mean) / (len(g) + 10)
    return drift, gaps


def build(ctx=None):
    ctx = ctx or mc.Context()
    drift, gaps = calibrate(ctx, int(ctx.panel.t.max()))
    with open(CALIBRATION_PATH, "wb") as f:
        pickle.dump(drift, f)
    return drift, gaps


def load():
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, "rb") as f:
            return pickle.load(f)
    return None
