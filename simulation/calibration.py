# =============================================================================
# simulation/calibration.py — Calibration de la dérive de la simulation.
#
# La validation sur le passé a montré que la simulation, sans correction, est
# trop optimiste et que l'écart s'accumule. Le modèle de valeur est juste quand
# on lui donne les VRAIES caractéristiques du joueur ; ce sont les
# caractéristiques SIMULÉES (états, temps de jeu...) qui dérivent.
#
# Correction : sur des centaines de cas RÉELS de la période d'entraînement, on
# simule 1 an et on compare à ce qui s'est vraiment passé. L'écart (rétréci vers
# la moyenne de l'âge), par tranche d'ÂGE et tranche de VALEUR, est la "dérive"
# à corriger : on l'ajoute à chaque trimestre simulé, selon la valeur simulée du
# moment. Une tranche de valeur absolue (et non le rang du joueur) corrige un
# défaut mesuré en validation : sans elle, les joueurs chers étaient surestimés
# et les moins chers sous-estimés. Cette correction n'utilise que des données
# antérieures à la date de coupure (aucune information du futur).
# =============================================================================

import pickle

import numpy as np
import pandas as pd

import config
from models import hmm

CALIBRATION_PATH = config.PROCESSED_DIR / "calibration.pkl"
VALUE_BANDS = [2e6, 5e6, 20e6, 50e6]      # limites des tranches de valeur (€) : 5 tranches
SHRINK = 10                                # poids (en nombre de cas) de la moyenne vers laquelle on rétrécit


def value_band(value):
    # Tranche de valeur (0 = moins de 2 M€ ... 4 = plus de 50 M€), vectorisé.
    return np.digitize(value, VALUE_BANDS)


def calibrate(ctx, cutoff_t, n_rows=1500, n_sims=300, seed=0):
    from simulation import monte_carlo as mc      # import ici : monte_carlo importe déjà ce module
    rng = np.random.default_rng(seed)
    panel = ctx.panel
    big5 = list(config.LEAGUES)
    # Cas réels : valeur connue à t et un an plus tard, les deux avant la coupure
    base = panel[(panel.t <= cutoff_t - 4) & (panel.market_value > 0) & panel.league.isin(big5)
                 & panel.age.between(17, 36)][["player_id", "t", "age", "market_value"]]
    later = panel[["player_id", "t", "market_value"]].assign(t=panel.t - 4).rename(columns={"market_value": "v4"})
    cases = base.merge(later, on=["player_id", "t"])
    cases = cases[cases.v4 > 0]
    cases["age_bin"] = np.digitize(cases.age, hmm.AGE_BINS[1:-1])
    cases["band"] = value_band(cases.market_value)
    # Échantillon équilibré entre tranches d'âge et de valeur
    n_cells = (len(hmm.AGE_BINS) - 1) * (len(VALUE_BANDS) + 1)
    cases = pd.concat([g.sample(min(len(g), n_rows // n_cells), random_state=seed)
                       for _, g in cases.groupby(["age_bin", "band"])])

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
        rows.append({"age_bin": c.age_bin, "band": int(c.band), "gap": actual - simulated})
    gaps = pd.DataFrame(rows)

    # Table de dérive annuelle (âge x tranche de valeur) : moyenne des écarts de
    # chaque case, rétrécie vers la moyenne de son âge, elle-même rétrécie vers
    # l'écart médian global (les cases avec peu de cas restent prudentes).
    drift = np.zeros((len(hmm.AGE_BINS) - 1, len(VALUE_BANDS) + 1))
    overall = gaps.gap.median()
    for ab in range(drift.shape[0]):
        g_age = gaps[gaps.age_bin == ab]
        age_mean = (g_age.gap.sum() + SHRINK * overall) / (len(g_age) + SHRINK)
        for band in range(drift.shape[1]):
            g = g_age[g_age.band == band].gap
            drift[ab, band] = (g.sum() + SHRINK * age_mean) / (len(g) + SHRINK)
    return drift, gaps


def build(ctx=None):
    from simulation import monte_carlo as mc
    ctx = ctx or mc.Context()
    drift, gaps = calibrate(ctx, int(ctx.panel.t.max()))
    with open(CALIBRATION_PATH, "wb") as f:
        pickle.dump({"drift": drift, "gaps": gaps}, f)
    return drift, gaps


def load(full=False):
    # full=True : renvoie aussi les écarts mesurés (cas réels), cités dans le papier
    if not CALIBRATION_PATH.exists():
        return None
    with open(CALIBRATION_PATH, "rb") as f:
        saved = pickle.load(f)
    if not isinstance(saved, dict) or saved["drift"].shape[1] != len(VALUE_BANDS) + 1:
        return None           # ancien format (âge x niveau de talent) : incompatible, à recalculer
    return saved if full else saved["drift"]
