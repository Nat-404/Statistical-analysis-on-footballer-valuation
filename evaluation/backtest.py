# =============================================================================
# evaluation/backtest.py — Étape 13 : validation sur le passé.
#
# On se replace à l'été 2020 (fin de la saison 2019/20) :
#   1. on ré-entraîne TOUS les modèles avec les seules données d'avant cette
#      date (sinon le modèle "connaîtrait" le futur : ce serait tricher) ;
#   2. pour plusieurs centaines de joueurs, on simule 6 ans (24 trimestres) ;
#   3. on compare aux VRAIES valeurs de juin 2026.
#
# Ce qu'on vérifie :
#   - COUVERTURE : la fourchette à 80 % doit contenir la vraie valeur dans
#     environ 80 % des cas (pareil pour 50 % et 90 %). Si c'est 95 %, les
#     fourchettes sont trop larges ; si c'est 60 %, trop étroites.
#   - BIAIS : la médiane prédite est-elle trop haute ou trop basse en moyenne ?
#     (au total, et par niveau de valeur : les stars sont-elles sous-estimées ?)
#   - LARGEUR : pour une même couverture, plus c'est étroit, mieux c'est.
# On choisit ensuite le niveau de bruit (resid_scale) qui donne la meilleure
# couverture : c'est la "calibration" des fourchettes.
#
# Limite à connaître : on ne peut tester que les joueurs qui ont encore une
# valeur en 2026 (ceux qui ont arrêté leur carrière sont exclus).
# =============================================================================

import numpy as np
import pandas as pd

import config
from models import collective, events, hmm, valuation
from simulation import monte_carlo as mc

ORIGIN_T = (config.BACKTEST_ORIGIN - 1) * 4 + 3       # mai-juillet 2020 = saison 2019, T4
TARGET_T = ORIGIN_T + config.N_STEPS                  # mai-juillet 2026


def fit_until(cutoff_t):
    # Ré-entraîne toute la chaîne de modèles sur les données d'avant cutoff_t.
    params, states, res = hmm.fit(cutoff_t, verbose=False)
    return {"cutoff_t": cutoff_t, "hmm_params": params, "states": states, "resilience": res,
            "valuation": valuation.fit(cutoff_t, states), "events": events.fit(cutoff_t, states),
            "collective": collective.fit(cutoff_t)}


def test_players(n_players=400, seed=0):
    # Joueurs testés : en top 5 à l'origine, 18-32 ans, valeur >= 1 M€,
    # et dont on connaît la vraie valeur 6 ans plus tard.
    q = pd.read_parquet("data/processed/quarterly.parquet")
    at_origin = q[(q.t == ORIGIN_T) & (q.market_value >= 1e6) & q.age.between(18, 32)
                  & q.league_id.isin([l["transfermarkt"] for l in config.LEAGUES.values()])]
    at_target = q[(q.t == TARGET_T) & (q.market_value > 0)][["player_id", "market_value"]]
    pool = at_origin.merge(at_target, on="player_id", suffixes=("", "_2026"))
    # Échantillon stratifié par niveau de valeur, pour bien tester les stars aussi
    pool["band"] = pd.cut(pool.market_value, [0, 5e6, 20e6, 50e6, 1e10], labels=["<5M", "5-20M", "20-50M", ">50M"])
    sample = pd.concat([g.sample(min(len(g), n_players // 4), random_state=seed)
                        for _, g in pool.groupby("band", observed=True)])
    return sample[["player_id", "name", "age", "band", "market_value", "market_value_2026"]]


def run(n_players=400, n_sims=2000, scales=(0.25, 0.5, 0.75, 1.0), fitted=None):
    fitted = fitted or fit_until(ORIGIN_T)
    ctx = mc.Context(fitted)
    players = test_players(n_players)
    rows = []
    for p in players.itertuples():
        for scale in scales:
            try:
                sim = mc.simulate(p.player_id, ctx, n=n_sims, origin_t=ORIGIN_T, resid_scale=scale)
            except ValueError:
                continue
            final = sim["value_nominal"][:, -1]               # valeur en euros courants (avec inflation)
            truth = p.market_value_2026
            row = {"player_id": p.player_id, "name": p.name, "band": p.band, "scale": scale,
                   "value_2020": p.market_value, "truth": truth, "median": np.median(final),
                   "pit": float(np.mean(final <= truth))}   # rang de la vérité dans la distribution
            for b in config.CONFIDENCE_BANDS:
                lo, hi = np.quantile(final, [(1 - b) / 2, 1 - (1 - b) / 2])
                row[f"in_{int(b * 100)}"] = lo <= truth <= hi
                row[f"width_{int(b * 100)}"] = np.log(hi / lo)
            rows.append(row)
    return pd.DataFrame(rows), fitted


def report(results):
    # Tableau de synthèse par niveau de bruit, puis par niveau de valeur.
    r = results.assign(log_error=np.log(results["median"] / results.truth))
    by_scale = r.groupby("scale").agg(
        couverture_50=("in_50", "mean"), couverture_80=("in_80", "mean"), couverture_90=("in_90", "mean"),
        largeur_80=("width_80", "mean"), biais_log=("log_error", "mean"), joueurs=("player_id", "nunique"))
    best = (by_scale.couverture_80 - 0.80).abs().idxmin()
    by_band = r[r.scale == best].groupby("band", observed=True).agg(
        couverture_80=("in_80", "mean"), biais_log=("log_error", "mean"), joueurs=("player_id", "nunique"))
    return by_scale, by_band, best


if __name__ == "__main__":
    results, _ = run()
    results.to_parquet(config.PROCESSED_DIR / "backtest.parquet")
    by_scale, by_band, best = report(results)
    print(by_scale.round(3))
    print(f"\nMeilleur niveau de bruit : {best}")
    print(by_band.round(3))
