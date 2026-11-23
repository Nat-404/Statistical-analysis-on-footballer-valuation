# =============================================================================
# config.py — TOUS les réglages du projet au même endroit.
# Si tu veux changer l'horizon, les championnats ou le nombre de simulations,
# c'est ICI et nulle part ailleurs. Les autres fichiers lisent ces valeurs.
# =============================================================================

from pathlib import Path  # Path permet de manipuler des chemins de fichiers proprement (Mac/Windows)

# -----------------------------------------------------------------------------
# 1. Chemins des dossiers
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent  # dossier racine du projet (là où se trouve ce fichier)
DATA_DIR = ROOT_DIR / "data"                # toutes les données
RAW_DIR = DATA_DIR / "raw"                  # données brutes, telles que téléchargées (jamais modifiées)
PROCESSED_DIR = DATA_DIR / "processed"      # données nettoyées et transformées, prêtes pour les modèles
CACHE_DIR = DATA_DIR / "cache"              # copies locales des pages web : chaque page n'est téléchargée qu'une fois
OUTPUTS_DIR = ROOT_DIR / "outputs"          # résultats finaux : rapports HTML, graphiques

# -----------------------------------------------------------------------------
# 2. Périmètre : les 5 grands championnats
# Chaque source de données nomme les championnats différemment, donc on garde
# la correspondance ici. La clé (ex. "ENG") est NOTRE nom interne.
# -----------------------------------------------------------------------------
LEAGUES = {
    "ENG": {"name": "Premier League", "transfermarkt": "GB1", "understat": "ENG-Premier League", "fotmob": 47, "clubelo": "ENG"},
    "ESP": {"name": "La Liga",        "transfermarkt": "ES1", "understat": "ESP-La Liga",        "fotmob": 87, "clubelo": "ESP"},
    "ITA": {"name": "Serie A",        "transfermarkt": "IT1", "understat": "ITA-Serie A",        "fotmob": 55, "clubelo": "ITA"},
    "GER": {"name": "Bundesliga",     "transfermarkt": "L1",  "understat": "GER-Bundesliga",     "fotmob": 54, "clubelo": "GER"},
    "FRA": {"name": "Ligue 1",        "transfermarkt": "FR1", "understat": "FRA-Ligue 1",        "fotmob": 53, "clubelo": "FRA"},
}
# Note : les identifiants FotMob seront vérifiés à l'étape 3.

# -----------------------------------------------------------------------------
# 3. Saisons étudiées
# Une saison est désignée par son année de DÉBUT : 2019 = saison 2019/20.
# Understat commence en 2014/15, c'est donc notre point de départ commun.
# -----------------------------------------------------------------------------
FIRST_SEASON = 2014    # première saison utilisée pour apprendre les modèles
CURRENT_SEASON = 2026  # saison en cours (2026/27) : point de départ des prévisions

# -----------------------------------------------------------------------------
# 4. Découpage trimestriel d'une saison (aligné sur le calendrier du football)
# Chaque trimestre est la liste des mois (1 = janvier ... 12 = décembre).
# -----------------------------------------------------------------------------
QUARTERS = {
    1: [8, 9, 10],   # T1 : août-octobre    -> début de saison, fin du mercato d'été
    2: [11, 12, 1],  # T2 : novembre-janvier -> enchaînement des matchs, mercato d'hiver
    3: [2, 3, 4],    # T3 : février-avril   -> phases finales européennes, matchs à enjeux
    4: [5, 6, 7],    # T4 : mai-juillet     -> fin de saison, tournois, mercato d'été
}

# -----------------------------------------------------------------------------
# 5. Simulation Monte Carlo
# -----------------------------------------------------------------------------
HORIZON_YEARS = 6                          # horizon de prévision demandé : 6 ans
N_STEPS = HORIZON_YEARS * len(QUARTERS)    # nombre de pas de simulation : 6 x 4 = 24 trimestres
N_SIMULATIONS = 10_000                     # nombre de carrières simulées par joueur
RANDOM_SEED = 42                           # graine aléatoire : rend les résultats reproductibles
                                           # (même graine = mêmes tirages = mêmes résultats)

# Niveaux de confiance affichés dans les graphiques en éventail (bandes autour de la médiane)
CONFIDENCE_BANDS = [0.50, 0.80, 0.90]

# -----------------------------------------------------------------------------
# 6. États de la chaîne de Markov cachée (le "niveau réel" du joueur)
# L'ordre compte : du plus bas au plus haut. Les seuils exacts seront appris
# sur les données à l'étape 8 ; ici on ne fixe que la liste.
# -----------------------------------------------------------------------------
STATES = [
    "Blessé",      # absent longue durée
    "Marginal",    # très peu de temps de jeu, ou prêté
    "Rotation",    # joue régulièrement sans être indiscutable
    "Titulaire",   # titulaire, niveau moyen pour son poste
    "Fort",        # titulaire, top 25 % de son poste
    "Élite",       # top 5 % de son poste dans les 5 grands championnats
]

# -----------------------------------------------------------------------------
# 7. Validation sur le passé (backtest)
# On se place à BACKTEST_ORIGIN, on prédit 6 ans plus tard, et on compare au réel.
# -----------------------------------------------------------------------------
BACKTEST_ORIGIN = 2019  # on "oublie" tout ce qui s'est passé après la saison 2019/20

# -----------------------------------------------------------------------------
# 8. Politesse envers les sites web (scraping)
# On attend entre deux requêtes pour ne pas surcharger les serveurs.
# -----------------------------------------------------------------------------
REQUEST_DELAY_SECONDS = 3.0
