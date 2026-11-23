# =============================================================================
# models/revenue.py — Étape 11 : revenus générés par le joueur pour son club.
#
# ⚠ C'est le volet le MOINS précis du projet : aucune donnée publique ne dit
#    combien de maillots se vendent par joueur, ni quelle part des sponsors
#    vient de lui. On construit donc des SCÉNARIOS à partir de chiffres publics
#    (sources ci-dessous) et d'hypothèses explicites, modifiables ici.
#
# 4 sources de revenus, par saison simulée :
#   1. Primes sportives & droits TV : les points qu'il apporte (étape 10)
#      x la valeur d'un point au classement (primes de classement des droits TV)
#      + ses victoires en plus en Ligue des champions (si son club la joue).
#   2. Maillots : maillots vendus à son nom x prix x part du club (royalties).
#   3. Sponsoring & image : part des revenus commerciaux du club attribuable
#      au joueur, selon son statut.
#   4. Réseaux sociaux & audience : part "numérique" liée à sa popularité.
# =============================================================================

import numpy as np

import config

# --- CHIFFRES PUBLICS (sources) ---------------------------------------------
# UEFA, distribution 2024-27 : 2,1 M€ par victoire, 0,7 M€ par nul en phase de ligue.
UCL_WIN_EUR = 2.1e6
UCL_LEAGUE_PHASE_MATCHES = 8
# Premier League 2024/25 : environ 2,66 M£ par place au classement (≈ 3,1 M€).
# Une place vaut environ 3 points en milieu de tableau -> ≈ 1,0 M€ par point.
EUR_PER_POINT = {"ENG": 1.0e6, "ESP": 0.45e6, "GER": 0.35e6, "ITA": 0.40e6, "FRA": 0.20e6, "autre": 0.10e6}
# Royalties maillots : le club touche environ 7,5 % à 20 % des ventes -> 12,5 % retenu.
SHIRT_PRICE_EUR = 95
SHIRT_ROYALTY = 0.125
# Deloitte Football Money League 2026 : revenus commerciaux moyens du top 20 ≈ 265 M€,
# Real Madrid 594 M€. Niveau commercial approximatif du club selon sa force Elo :
COMMERCIAL_BY_ELO = [(1950, 450e6), (1850, 280e6), (1750, 130e6), (1650, 60e6), (0, 20e6)]

# --- HYPOTHÈSES (aucune donnée publique) -------------------------------------
# Maillots vendus par saison à son nom, selon son état
SHIRTS_BY_STATE = {"Blessé": 20_000, "Marginal": 2_000, "Rotation": 10_000,
                   "Titulaire": 40_000, "Fort": 150_000, "Élite": 600_000}
# Part des revenus commerciaux du club attribuable au joueur
SPONSOR_SHARE_BY_STATE = {"Blessé": 0.004, "Marginal": 0.0, "Rotation": 0.002,
                          "Titulaire": 0.006, "Fort": 0.015, "Élite": 0.05}
# Part "numérique" (réseaux sociaux, audience), en % du sponsoring attribué
DIGITAL_SHARE_OF_SPONSOR = 0.25
# Seuil Elo à partir duquel on suppose que le club joue la Ligue des champions
UCL_ELO_THRESHOLD = 1780


def club_commercial(elo):
    # Revenus commerciaux approximatifs d'un club selon sa force Elo (tableau ci-dessus)
    out = np.full(np.shape(elo), COMMERCIAL_BY_ELO[-1][1], dtype=float)
    for threshold, value in reversed(COMMERCIAL_BY_ELO):
        out = np.where(elo >= threshold, value, out)
    return out


def season_revenue(states, points, club_elo, league="ENG", popularity_mult=1.0):
    # Revenus d'une saison pour N simulations à la fois.
    # states : état dominant de la saison (indices) ; points : points apportés ;
    # club_elo : Elo moyen du club sur la saison.
    names = np.array(config.STATES)[states]
    per_point = EUR_PER_POINT.get(league, EUR_PER_POINT["autre"])
    sporting = np.maximum(points, 0) * per_point
    # En Ligue des champions : points apportés -> victoires en plus (1 victoire ≈ 3 points)
    ucl = (club_elo >= UCL_ELO_THRESHOLD) * np.maximum(points, 0) / 3 \
        * UCL_LEAGUE_PHASE_MATCHES / 38 * UCL_WIN_EUR
    shirts = np.vectorize(SHIRTS_BY_STATE.get)(names) * SHIRT_PRICE_EUR * SHIRT_ROYALTY * popularity_mult
    sponsor = np.vectorize(SPONSOR_SHARE_BY_STATE.get)(names) * club_commercial(club_elo) * popularity_mult
    digital = sponsor * DIGITAL_SHARE_OF_SPONSOR
    return {"Primes & droits TV": sporting + ucl, "Maillots": shirts,
            "Sponsoring & image": sponsor - digital, "Réseaux sociaux & audience": digital}
