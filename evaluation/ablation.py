# =============================================================================
# evaluation/ablation.py — Une amélioration sert-elle vraiment ?
#
# On refait la validation en AJOUTANT un ingrédient candidat au modèle, toutes
# choses égales par ailleurs, et on compare au modèle de production. Si la
# validation n'est pas meilleure avec l'ingrédient, il n'a pas sa place.
#
#     python -m evaluation.ablation onoff                 -> avec le critère "on/off", 2020 -> 2026
#     python -m evaluation.ablation onoff --origin 2023   -> même chose, validation 2023 -> 2026
#
# Résultat (novembre 2026) : l'on/off ne change pas l'erreur des prévisions
# (écart moyen par joueur < 0,002 en log, non significatif) : non retenu.
#
# Les fichiers de la variante sont écrits dans data/processed/ablation/ : les
# modèles de production ne sont pas touchés.
# =============================================================================

import sys

import config
from evaluation import backtest as bt
from models import performance as pf

FOLDER = config.PROCESSED_DIR / "ablation"


ONOFF_WEIGHT = 0.10          # poids testé pour les joueurs de champ (les autres poids sont renormalisés)


def with_onoff(season):
    # Indice de performance recalculé AVEC le critère on/off, puis validation
    # complète (tous les modèles ré-estimés) sur ce nouvel indice.
    for group, weights in pf.CRITERIA.items():
        if group != "GK":
            weights["onoff_90"] = ONOFF_WEIGHT
    pf.OUT_PATH = FOLDER / "performance_with_onoff.parquet"
    pf.CONTEXT_PATH = FOLDER / "performance_context_with_onoff.json"
    pf.build()
    results, _ = bt.run(origin_t=bt.origin_from_season(season), steps=bt.horizon_steps(season))
    return results


VARIANTS = {"onoff": with_onoff}


if __name__ == "__main__":
    FOLDER.mkdir(parents=True, exist_ok=True)
    name = sys.argv[1]
    season = int(sys.argv[sys.argv.index("--origin") + 1]) if "--origin" in sys.argv else config.BACKTEST_ORIGIN
    results = VARIANTS[name](season)
    suffix = "" if season == config.BACKTEST_ORIGIN else f"_{season}"
    results.to_parquet(FOLDER / f"backtest_with_{name}{suffix}.parquet")
    by_scale, by_band, best = bt.report(results)
    print(by_scale.round(3))
    print(f"\nMeilleur niveau de bruit : {best}")
    print(by_band.round(3))
