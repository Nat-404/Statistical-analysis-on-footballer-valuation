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
# NIVEAU DE MARCHÉ. L'inflation future est imprévisible (en 2020, la tendance
# 2015-2019 annonçait +10 %/an ; le marché a fait environ +4 %/an) et inégale
# selon l'âge (la hausse 2023-2025 a surtout porté sur les jeunes ; la valeur
# d'un joueur de 25-27 ans est restée stable en médiane). On enregistre donc
# trois comparaisons :
#   - principale : prévision aux prix du départ contre vraie valeur, sans
#     ajustement (le marché des âges testés est resté à peu près plat) ;
#   - "_deflated" : vraie valeur dégonflée par l'indice général du marché
#     (inflation observée, tous âges confondus) ;
#   - "_nominal" : prévision avec l'inflation estimée au départ, en euros courants.
#
# Limite à connaître : on ne peut tester que les joueurs qui ont encore une
# valeur en 2026 (ceux qui ont arrêté leur carrière sont exclus).
#
# Seconde validation, plus courte : départ à l'été 2023, horizon 3 ans
# (python -m evaluation.backtest --origin 2023). Elle sert à juger les
# variables qui n'existent qu'à partir de 2020/21 (xG, impact on/off) : la
# validation 2020 ne peut pas les voir, puisqu'elles n'existent pas avant.
# =============================================================================

import numpy as np
import pandas as pd

import config
from models import collective, events, hmm, valuation
from simulation import monte_carlo as mc

ORIGIN_T = (config.BACKTEST_ORIGIN - 1) * 4 + 3       # mai-juillet 2020 = saison 2019, T4
TARGET_T = ORIGIN_T + config.N_STEPS                  # mai-juillet 2026


def fit_until(cutoff_t):
    # Ré-entraîne toute la chaîne de modèles sur les données d'avant cutoff_t,
    # y compris la calibration de la dérive (elle aussi limitée au passé).
    from simulation import calibration
    params, states, res = hmm.fit(cutoff_t, verbose=False)
    fitted = {"cutoff_t": cutoff_t, "hmm_params": params, "states": states, "resilience": res,
              "valuation": valuation.fit(cutoff_t, states), "events": events.fit(cutoff_t, states),
              "collective": collective.fit(cutoff_t), "drift": None}
    fitted["drift"], _ = calibration.calibrate(mc.Context(fitted), cutoff_t)
    return fitted


def origin_from_season(season):
    # Départ à la fin de la saison (season-1)/(season) : trimestre mai-juillet de "season".
    return (season - 1) * 4 + 3


def test_players(n_players=400, seed=0, origin_t=ORIGIN_T, target_t=TARGET_T):
    # Joueurs testés : en top 5 à l'origine, 18-32 ans, valeur >= 1 M€,
    # et dont on connaît la vraie valeur à l'horizon.
    q = pd.read_parquet("data/processed/quarterly.parquet")
    at_origin = q[(q.t == origin_t) & (q.market_value >= 1e6) & q.age.between(18, 32)
                  & q.league_id.isin([l["transfermarkt"] for l in config.LEAGUES.values()])]
    at_target = q[(q.t == target_t) & (q.market_value > 0)][["player_id", "market_value"]]
    pool = at_origin.merge(at_target, on="player_id", suffixes=("", "_2026"))
    # Échantillon stratifié par niveau de valeur, pour bien tester les stars aussi
    pool["band"] = pd.cut(pool.market_value, [0, 5e6, 20e6, 50e6, 1e10], labels=["<5M", "5-20M", "20-50M", ">50M"])
    sample = pd.concat([g.sample(min(len(g), n_players // 4), random_state=seed)
                        for _, g in pool.groupby("band", observed=True)])
    return sample[["player_id", "name", "age", "band", "market_value", "market_value_2026"]]


def run(n_players=400, n_sims=2000, scales=(0.25, 0.5, 0.75, 1.0), fitted=None, origin_t=ORIGIN_T,
        steps=config.N_STEPS):
    fitted = fitted or fit_until(origin_t)
    ctx = mc.Context(fitted)
    players = test_players(n_players, origin_t=origin_t, target_t=origin_t + steps)
    market_change = realized_market_change(origin_t, origin_t + steps)
    rows = []
    for p in players.itertuples():
        for scale in scales:
            try:
                sim = mc.simulate(p.player_id, ctx, n=n_sims, steps=steps, origin_t=origin_t, resid_scale=scale)
            except ValueError:
                continue
            row = {"player_id": p.player_id, "name": p.name, "band": p.band, "scale": scale,
                   "value_2020": p.market_value, "market_change": market_change}
            # 1. Prix du départ contre vraie valeur (critère principal)
            row.update(scores(sim["value_real"][:, -1], p.market_value_2026))
            # 2. Prix du départ contre vraie valeur dégonflée par l'indice général du marché
            row.update({f"{k}_deflated": v for k, v in
                        scores(sim["value_real"][:, -1], p.market_value_2026 * np.exp(-market_change)).items()})
            # 3. En euros courants : prévision avec inflation simulée contre vraie valeur
            row.update({f"{k}_nominal": v for k, v in
                        scores(sim["value_nominal"][:, -1], p.market_value_2026).items()})
            rows.append(row)
    return pd.DataFrame(rows), fitted


def scores(final, truth):
    # Médiane, rang de la vérité (PIT) et couverture des intervalles d'une distribution simulée.
    out = {"truth": truth, "median": float(np.median(final)), "pit": float(np.mean(final <= truth))}
    for b in config.CONFIDENCE_BANDS:
        lo, hi = np.quantile(final, [(1 - b) / 2, 1 - (1 - b) / 2])
        out[f"in_{int(b * 100)}"] = lo <= truth <= hi
        out[f"width_{int(b * 100)}"] = np.log(hi / lo)
    return out


def realized_market_change(origin_t, target_t):
    # Inflation RÉELLEMENT observée entre le départ et l'horizon (en log), d'après les
    # effets saison du modèle de valeur estimé sur toutes les données.
    index = valuation.load()["inflation"]["index"]
    return float(index[target_t // 4] - index[origin_t // 4])


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


def output_path(origin_season=None):
    name = "backtest.parquet" if origin_season in (None, config.BACKTEST_ORIGIN) else f"backtest_{origin_season}.parquet"
    return config.PROCESSED_DIR / name


def horizon_steps(origin_season):
    # Horizon : jusqu'à juin 2026 (dernière valeur connue), au plus 6 ans
    return min(config.N_STEPS, (config.CURRENT_SEASON - origin_season) * 4)


if __name__ == "__main__":
    import sys
    season = int(sys.argv[sys.argv.index("--origin") + 1]) if "--origin" in sys.argv else config.BACKTEST_ORIGIN
    results, _ = run(origin_t=origin_from_season(season), steps=horizon_steps(season))
    results.to_parquet(output_path(season))
    by_scale, by_band, best = report(results)
    print(by_scale.round(3))
    print(f"\nMeilleur niveau de bruit : {best}")
    print(by_band.round(3))
