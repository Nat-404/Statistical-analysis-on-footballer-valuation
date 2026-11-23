# =============================================================================
# Main.py — Point d'entrée du programme.
# Pour l'instant (étape 0), il vérifie simplement que l'installation est
# correcte. À la fin du projet, il lancera l'analyse complète d'un joueur :
#     python Main.py "Nom du joueur"
# =============================================================================

import importlib  # permet d'importer une bibliothèque à partir de son nom (texte)

import config     # notre fichier de réglages


def check_folders():
    # Crée les dossiers de données s'ils n'existent pas encore.
    # exist_ok=True : pas d'erreur si le dossier existe déjà.
    # parents=True : crée aussi les dossiers parents manquants.
    for folder in [config.RAW_DIR, config.PROCESSED_DIR, config.CACHE_DIR, config.OUTPUTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    print("✓ Dossiers de données en place")


def check_libraries():
    # Essaie d'importer chaque bibliothèque nécessaire.
    # Si l'une manque, on l'ajoute à la liste "missing" au lieu de planter.
    # (Le nom d'import diffère parfois du nom d'installation : scikit-learn -> sklearn)
    libraries = ["pandas", "numpy", "pyarrow", "duckdb", "soccerdata", "requests",
                 "scipy", "statsmodels", "sklearn", "plotly", "matplotlib", "tqdm"]
    missing = []
    for name in libraries:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")  # récupère la version si elle existe
            print(f"  ✓ {name:<12} {version}")
        except ImportError:
            missing.append(name)
            print(f"  ✗ {name:<12} MANQUANTE")
    return missing


def show_config():
    # Affiche un résumé des réglages, pour vérifier d'un coup d'œil.
    print("\nRéglages du projet :")
    print(f"  Championnats : {', '.join(l['name'] for l in config.LEAGUES.values())}")
    print(f"  Saisons      : {config.FIRST_SEASON}/{config.FIRST_SEASON + 1} -> {config.CURRENT_SEASON}/{config.CURRENT_SEASON + 1}")
    print(f"  Horizon      : {config.HORIZON_YEARS} ans = {config.N_STEPS} trimestres")
    print(f"  Simulations  : {config.N_SIMULATIONS:,} carrières par joueur")
    print(f"  États Markov : {' < '.join(config.STATES)}")


# Ce bloc ne s'exécute que si on lance directement "python Main.py"
# (et pas si un autre fichier fait "import Main").
if __name__ == "__main__":
    print("=== Vérification de l'installation ===\n")
    check_folders()
    print("\nBibliothèques :")
    missing = check_libraries()
    show_config()
    if missing:
        print(f"\n⚠ Il manque : {', '.join(missing)} -> lance : pip install -r requirements.txt")
    else:
        print("\n✓ Tout est prêt pour l'étape 1.")
