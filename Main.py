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

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    offline = "--offline" in sys.argv          # --offline : pas de récupération FotMob à la demande
    print("Chargement des modèles…")
    ctx = mc.Context()
    player_id = find_player(ctx, " ".join(args))

    # Données récentes (saison en cours) récupérées à la demande sur FotMob
    live = None
    if not offline:
        from collectors import live as live_data
        from models import hmm
        last = ctx.panel[ctx.panel.player_id == player_id].sort_values("t").iloc[-1]
        probs = last[[f"f_{s}" for s in config.STATES]].to_numpy(dtype=float)
        print("Récupération des matchs 2026/27 sur FotMob…")
        updated, obs, added = live_data.update_player(
            # âge au début de la saison 2026/27 (FotMob donne l'âge arrondi à l'année inférieure)
            last["name"], last.age + 0.25, last.pos_group, probs / probs.sum(), ctx.hmm["params"],
            int(last.cls) if last.cls == last.cls else 0)
        if added:
            print(f"  {added} trimestre(s) 2026/27 intégré(s) : {int(obs.matches.sum())} matchs, "
                  f"{obs.minutes.sum():.0f} minutes")
            live = {"probs": updated, "obs": obs}
        else:
            print("  aucune donnée récente trouvée (départ en juillet 2026)")

    print(f"Simulation de {config.N_SIMULATIONS:,} carrières…")
    sim = mc.simulate(player_id, ctx, live=live)
    from reporting import fiche
    pdf = fiche.build_fiche(player_id, ctx, sim, live=live)
    print(f"✓ Fiche de projection (2 pages, anglais) : {pdf}")
    if "--html" in sys.argv:                    # page web interactive détaillée, en option
        path, _ = report.build_report(player_id, ctx, live=live)
        print(f"✓ Rapport interactif : {path}")


if __name__ == "__main__":
    main()
