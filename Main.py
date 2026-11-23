# =============================================================================
# Main.py — Point d'entrée : analyse complète d'un joueur.
#
#     python Main.py "Bukayo Saka"          -> rapport HTML dans outputs/
#     python Main.py --check                -> vérifie l'installation
#
# Étapes : 1) trouver le joueur  2) charger les modèles entraînés
#          3) simuler 10 000 carrières sur 6 ans  4) écrire le rapport
# =============================================================================

import importlib
import sys
import unicodedata

import config


def check_installation():
    # Vérifie dossiers et bibliothèques (l'ancienne étape 0).
    for folder in [config.RAW_DIR, config.PROCESSED_DIR, config.CACHE_DIR, config.OUTPUTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in ["pandas", "numpy", "pyarrow", "duckdb", "requests", "scipy", "statsmodels",
                 "sklearn", "plotly", "rapidfuzz", "tqdm"]:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    print("✓ Installation complète" if not missing else f"⚠ Il manque : {', '.join(missing)}")


def _normalize(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def find_player(ctx, query):
    # Cherche le joueur par son nom (sans tenir compte des accents ni des majuscules).
    # S'il y a plusieurs résultats, on prend celui dont la valeur actuelle est la plus élevée.
    latest = ctx.panel.sort_values("t").groupby("player_id").tail(1)
    hits = latest[latest.name.fillna("").map(_normalize).str.contains(_normalize(query), regex=False)]
    if hits.empty:
        sys.exit(f"Aucun joueur trouvé pour « {query} »")
    if len(hits) > 1:
        print("Plusieurs joueurs correspondent :")
        for row in hits.sort_values("market_value", ascending=False).head(5).itertuples():
            print(f"  - {row.name} ({row.age:.0f} ans, {row.market_value / 1e6:.1f} M€)")
    return int(hits.sort_values("market_value", ascending=False).player_id.iloc[0])


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--check":
        check_installation()
        return
    from reporting import report
    from simulation import monte_carlo as mc

    print("Chargement des modèles…")
    ctx = mc.Context()
    player_id = find_player(ctx, " ".join(sys.argv[1:]))
    print(f"Simulation de {config.N_SIMULATIONS:,} carrières…")
    path, sim = report.build_report(player_id, ctx)
    print(f"✓ Rapport : {path}")


if __name__ == "__main__":
    main()
