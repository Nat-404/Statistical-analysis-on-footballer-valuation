# =============================================================================
# Main.py — Point d'entrée : analyse complète d'un joueur.
#
#     python Main.py "Bukayo Saka"          -> fiche PDF de 2 pages dans outputs/
#     python Main.py "Bukayo Saka" --offline -> sans aucune requête en ligne
#     python Main.py --check                -> vérifie l'installation
#
# Version fenêtre (champ "joueur" + bouton "Simulation", PDF dans Téléchargements) :
# double-cliquer sur "Football Valuation.app" (voir app/simulation_app.py).
#
# Au lancement (sauf --offline), deux sources sont interrogées pour CE joueur :
#   - FotMob : ses matchs de la saison en cours (met à jour son état de départ) ;
#   - GDELT  : contrôle presse des incidents hors terrain, 2017 -> aujourd'hui
#              (affiché dans la fiche, sans modifier la simulation).
#
# Étapes : 1) trouver le joueur  2) charger les modèles entraînés
#          3) simuler 10 000 carrières sur 6 ans  4) écrire la fiche PDF
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
        raise LookupError(f"Aucun joueur trouvé pour « {query} »")
    if len(hits) > 1:
        print("Plusieurs joueurs correspondent :")
        for row in hits.sort_values("market_value", ascending=False).head(5).itertuples():
            print(f"  - {row.name} ({row.age:.0f} ans, {row.market_value / 1e6:.1f} M€)")
    return int(hits.sort_values("market_value", ascending=False).player_id.iloc[0])


def unique_path(folder, name):
    # "Rio_Ngumoha_projection.pdf", ou "... (2).pdf" si le fichier existe déjà
    base = f"{name.replace(' ', '_')}_projection"
    path, k = folder / f"{base}.pdf", 2
    while path.exists():
        path, k = folder / f"{base} ({k}).pdf", k + 1
    return path


def analyse(query=None, player_id=None, offline=False, out_dir=None, progress=None, html=False):
    # Analyse complète d'un joueur ; renvoie le chemin de la fiche PDF.
    # query : nom recherché ; player_id : identifiant Transfermarkt (prioritaire).
    # out_dir : dossier de la fiche (par défaut outputs/, nom fixe écrasé à chaque fois).
    # progress(étape, fraction) : appelé à chaque étape (sert à la barre
    # d'avancement de la fenêtre, app/simulation_app.py). Étapes, dans l'ordre :
    # "models", "fotmob", "press", "simulation" (fraction 0 -> 1), "pdf".
    from simulation import monte_carlo as mc

    def step(name, fraction=0.0):
        if progress:
            progress(name, fraction)

    step("models")
    print("Chargement des modèles…")
    ctx = mc.Context()
    player_id = player_id if player_id is not None else find_player(ctx, query)
    player_name = ctx.panel[ctx.panel.player_id == player_id].sort_values("t").iloc[-1]["name"]
    path = unique_path(out_dir, player_name) if out_dir else None

    # Données récentes (saison en cours) récupérées à la demande sur FotMob
    live = None
    if not offline:
        from collectors import live as live_data
        last = ctx.panel[ctx.panel.player_id == player_id].sort_values("t").iloc[-1]
        probs = last[[f"f_{s}" for s in config.STATES]].to_numpy(dtype=float)
        step("fotmob")
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

    # Contrôle presse (GDELT), à la demande : données à jour au jour près
    press = None
    if not offline:
        from collectors import gdelt
        step("press")
        print("Contrôle presse GDELT (peut prendre 1 à 5 minutes, l'API limite les requêtes)…")
        press = gdelt.run(player_name)
        print("  " + (gdelt.summary_en(press) or ""))

    print(f"Simulation de {config.N_SIMULATIONS:,} carrières…")
    step("simulation")
    sim = mc.simulate(player_id, ctx, live=live, progress=lambda f: step("simulation", f))
    step("pdf")
    from reporting import fiche
    pdf = fiche.build_fiche(player_id, ctx, sim, live=live, press=press, path=path)
    print(f"✓ Fiche de projection (2 pages, anglais) : {pdf}")
    if html:                                    # page web interactive détaillée, en option
        from reporting import report
        page, _ = report.build_report(player_id, ctx, live=live)
        print(f"✓ Rapport interactif : {page}")
    return pdf


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--check":
        check_installation()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    try:
        # --offline : aucune requête en ligne (FotMob, GDELT) ; --html : page web en plus
        analyse(" ".join(args), offline="--offline" in sys.argv, html="--html" in sys.argv)
    except LookupError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
