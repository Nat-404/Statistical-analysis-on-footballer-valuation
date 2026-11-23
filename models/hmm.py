# =============================================================================
# models/hmm.py — Étape 7 : la chaîne de Markov cachée (HMM) et la résilience.
#
# L'IDÉE
#   Chaque trimestre, le joueur est dans un ÉTAT réel qu'on ne voit pas
#   directement : Blessé, Marginal, Rotation, Titulaire, Fort, Élite.
#   Ce qu'on VOIT, ce sont des indices bruités de cet état :
#     - sa part du temps de jeu (minutes / minutes possibles)
#     - son indice de performance ajusté du contexte (étape 6)
#     - s'il a manqué des matchs pour blessure (FotMob)
#   Le modèle apprend deux choses à la fois :
#     1. les ÉMISSIONS : à quoi ressemblent les observations dans chaque état
#        (ex. un "Titulaire" joue 60 à 90 % des minutes, indice moyen ~0)
#     2. les TRANSITIONS : la probabilité de passer d'un état à l'autre d'un
#        trimestre au suivant, SELON L'ÂGE et la FAMILLE DE POSTE
#        (28 matrices : 7 tranches d'âge x 4 familles)
#
# L'ALGORITHME (Baum-Welch, une forme d'"EM")
#   On part d'une première affectation par règles simples, puis on répète :
#   (E) avec les paramètres actuels, calculer pour chaque trimestre la
#       probabilité d'être dans chaque état (algorithme "avant-arrière") ;
#   (M) recalculer émissions et transitions à partir de ces probabilités.
#   Chaque tour améliore la vraisemblance ; on s'arrête quand ça ne bouge plus.
#
# LA RÉSILIENCE (le "caractère")
#   Après un trimestre de baisse, certains joueurs rebondissent plus souvent
#   que la moyenne, d'autres s'enfoncent. On compte, pour chaque joueur, ses
#   rebonds réels par rapport aux rebonds attendus pour son âge et son poste.
#   Ratio > 1 = plus résilient que la moyenne. Ratio lissé (bayésien) pour
#   ne pas surinterpréter 2 ou 3 épisodes.
# =============================================================================

import pickle

import numpy as np
import pandas as pd

import config
from features import quarterly as qt
from models import performance as pf

K = len(config.STATES)                           # 6 états
BLESSE, MARGINAL, ROTATION, TITULAIRE, FORT, ELITE = range(K)
AGE_BINS = [0, 20, 23, 26, 29, 32, 35, 99]       # tranches : <20, 20-22, 23-25, 26-28, 29-31, 32-34, 35+
FAMILIES = {"GK": 0, "CB": 1, "FB": 1, "MID": 2, "AMW": 3, "FW": 3}   # gardien, défense, milieu, attaque
# Niveau de TALENT : sans lui, la chaîne n'a pas de mémoire (une star blessée
# redeviendrait un remplaçant comme un autre). Mesuré par le rang de sa valeur
# marchande parmi les joueurs de son âge à la même date (information connue
# à ce moment-là, sans regarder le futur) : 0 = reste, 1 = top 10-40 %, 2 = top 10 %.
TIER_CUTS = [0.60, 0.90]
N_TIERS = len(TIER_CUTS) + 1
N_AGES = len(AGE_BINS) - 1
N_CLASSES = N_AGES * 4 * N_TIERS


def class_index(age_bin, family, tier):
    # Numéro de la matrice de transition : tranche d'âge x famille de poste x talent
    return (age_bin * 4 + family) * N_TIERS + tier
SHARE_BINS = [-0.01, 0.05, 0.30, 0.60, 0.85, 1.01]   # 5 tranches de part de temps de jeu
PRIOR_STRENGTH = 30          # poids du lissage des matrices vers la matrice moyenne par âge
FULL_WEIGHT_MINUTES = 1080   # 12 matchs complets : l'indice de perf compte à 100 %
MODEL_PATH = config.PROCESSED_DIR / "hmm.pkl"
STATES_PATH = config.PROCESSED_DIR / "hmm_states.parquet"


# -----------------------------------------------------------------------------
# 1. Préparation des séquences
# -----------------------------------------------------------------------------
def load_panel(cutoff_t=None):
    # Assemble la table trimestrielle et l'indice de performance.
    # cutoff_t : pour la validation sur le passé, on coupe les données à une date.
    q = pd.read_parquet(qt.OUT_PATH)
    p = pd.read_parquet(pf.OUT_PATH)[["player_id", "t", "pos_group", "perf_adj", "perf_minutes"]]
    df = q.merge(p, on=["player_id", "t"], how="left")
    if cutoff_t is not None:
        df = df[df.t <= cutoff_t]
    df = df[df.pos_group.notna() & df.age.notna()].sort_values(["player_id", "t"]).reset_index(drop=True)

    # Observation 1 : tranche de part de temps de jeu (manquante si le club
    # n'est pas dans un championnat couvert : on ne sait pas combien il pouvait jouer)
    df["share_bin"] = pd.cut(df.minutes_share, SHARE_BINS, labels=False)
    # Observation 2 : indice de perf, avec un poids selon les minutes jouées
    df["perf_w"] = np.clip(df.perf_minutes.fillna(0) / FULL_WEIGHT_MINUTES, 0, 1)
    df.loc[df.perf_minutes.fillna(0) < 90, "perf_adj"] = np.nan
    # Observation 3 : a-t-il manqué au moins un match pour blessure ? (FotMob)
    df["injured_obs"] = (df["missed_injury"] > 0).astype(float) if "missed_injury" in df else np.nan
    if "missed_injury" in df:
        df.loc[df.missed_injury.isna(), "injured_obs"] = np.nan
    # Indicateur de repli (Transfermarkt) quand FotMob ne dit rien :
    #   - titulaire (>= 50 %) au trimestre précédent, quasi 0 minute maintenant,
    #     sans transfert, dans un championnat couvert -> très probablement blessé (1)
    #   - joue au moins 30 % des minutes -> pas blessé (0)
    prev_share = df.groupby("player_id").minutes_share.shift()
    proxy = pd.Series(np.nan, index=df.index)
    proxy[(df.minutes_share < 0.05) & (prev_share >= 0.5) & ~df.moved] = 1.0
    proxy[df.minutes_share >= 0.30] = 0.0
    df["injured_obs"] = df["injured_obs"].fillna(proxy)
    # Classe de transition : tranche d'âge x famille de poste x niveau de talent
    df["age_bin"] = np.digitize(df.age, AGE_BINS[1:-1])
    pct = df.groupby(["season", "age_bin"]).market_value.rank(pct=True).fillna(0)
    df["tier"] = np.digitize(pct, TIER_CUTS)
    df["cls"] = class_index(df.age_bin, df.pos_group.map(FAMILIES).astype(int), df.tier)
    return df


def initial_states(df):
    # Première affectation par règles simples (point de départ de l'EM).
    # Seuils de performance : 75e et 95e centiles chez les titulaires ("Fort" et "Élite").
    starters = df.minutes_share >= 0.6
    q75, q95 = df.loc[starters, "perf_adj"].quantile([0.75, 0.95])
    s = np.full(len(df), MARGINAL)
    s[df.minutes_share.between(0.30, 0.60)] = ROTATION
    s[starters] = TITULAIRE
    s[starters & (df.perf_adj >= q75)] = FORT
    s[starters & (df.perf_adj >= q95)] = ELITE
    s[(df.injured_obs == 1) & (df.minutes_share.fillna(0) < 0.3)] = BLESSE
    # Chute brutale sans explication (titulaire -> 0 minute) : probablement blessé
    prev_share = df.groupby("player_id").minutes_share.shift()
    s[(df.minutes_share.fillna(1) < 0.05) & (prev_share >= 0.6)] = BLESSE
    return s


# -----------------------------------------------------------------------------
# 2. Paramètres : émissions et transitions
# -----------------------------------------------------------------------------
def m_step(df, gamma, xi_by_class, first_by_class):
    # Recalcule tous les paramètres à partir des probabilités d'états.
    # gamma[i, k] = probabilité que la ligne i soit dans l'état k.
    params = {}

    # Émission 1 : distribution des tranches de temps de jeu par état (+1 de lissage)
    share = np.ones((K, len(SHARE_BINS) - 1))
    ok = df.share_bin.notna().to_numpy()
    for b in range(len(SHARE_BINS) - 1):
        share[:, b] += gamma[ok & (df.share_bin == b).to_numpy()].sum(axis=0)
    # Titulaire, Fort et Élite jouent autant : ils partagent la MÊME distribution
    # de temps de jeu. Seule la performance les distingue (c'est leur définition).
    share[TITULAIRE:] = share[TITULAIRE:].sum(axis=0)
    params["share"] = share / share.sum(axis=1, keepdims=True)

    # Émission 2 : performance ~ Normale(moyenne_k, écart-type_k), pondérée par les minutes
    perf = df.perf_adj.to_numpy()
    w = (df.perf_w.to_numpy() * ~np.isnan(perf))[:, None] * gamma
    x = np.nan_to_num(perf)[:, None]
    mu = (w * x).sum(0) / np.maximum(w.sum(0), 1e-9)
    var = (w * (x - mu) ** 2).sum(0) / np.maximum(w.sum(0), 1e-9)
    params["perf_mu"], params["perf_sd"] = mu, np.sqrt(np.maximum(var, 0.05))

    # Émission 3 : probabilité d'avoir manqué un match sur blessure, par état
    inj = df.injured_obs.to_numpy()
    okj = ~np.isnan(inj)
    params["injury"] = ((gamma[okj] * inj[okj, None]).sum(0) + 1) / (gamma[okj].sum(0) + 2)
    # Un joueur "Marginal" est écarté, pas blessé : on plafonne sa probabilité de
    # blessure observée, sinon "Marginal" et "Blessé" deviennent indiscernables.
    params["injury"][MARGINAL] = min(params["injury"][MARGINAL], 0.10)

    # Transitions par classe, lissées vers la moyenne de la même tranche d'âge
    # et du même niveau de talent (tous postes confondus)
    counts = xi_by_class + 1e-3
    grid = counts.reshape(N_AGES, 4, N_TIERS, K, K)
    pooled = grid.sum(axis=1, keepdims=True)                                # somme sur les postes
    pooled = pooled / pooled.sum(axis=-1, keepdims=True)
    prior = np.broadcast_to(pooled, grid.shape).reshape(N_CLASSES, K, K) * PRIOR_STRENGTH
    trans = counts + prior
    params["trans"] = trans / trans.sum(axis=2, keepdims=True)

    init = first_by_class + 1
    params["init"] = init / init.sum(axis=1, keepdims=True)
    return _enforce_order(params)


def _enforce_order(params):
    # Garantit Titulaire < Fort < Élite en performance moyenne. Si l'EM les a
    # inversés, on renumérote ces 3 états (émissions ET transitions) pour que
    # chaque nom corresponde toujours à son niveau.
    top = np.arange(TITULAIRE, K)
    order = top[np.argsort(params["perf_mu"][top])]
    if np.array_equal(order, top):
        return params
    perm = np.arange(K)
    perm[top] = order
    for key in ("share", "perf_mu", "perf_sd", "injury"):
        params[key] = params[key][perm]
    params["trans"] = params["trans"][:, perm][:, :, perm]
    params["init"] = params["init"][:, perm]
    return params


def log_emissions(df, params):
    # Log-probabilité de chaque observation dans chaque état (lignes x états).
    # Une observation manquante ne contribue pas (log 1 = 0).
    n = len(df)
    le = np.zeros((n, K))
    sb = df.share_bin.to_numpy()
    ok = ~np.isnan(sb)
    le[ok] += np.log(params["share"][:, sb[ok].astype(int)].T)

    perf, w = df.perf_adj.to_numpy(), df.perf_w.to_numpy()
    okp = ~np.isnan(perf) & (w > 0)
    # Peu de minutes = indice moins fiable : on élargit la variance (division par le poids)
    sd = params["perf_sd"][None, :] / np.sqrt(w[okp, None])
    z = (perf[okp, None] - params["perf_mu"][None, :]) / sd
    le[okp] += -0.5 * z ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)

    inj = df.injured_obs.to_numpy()
    okj = ~np.isnan(inj)
    p = params["injury"][None, :]
    le[okj] += np.where(inj[okj, None] == 1, np.log(p), np.log(1 - p))
    return le


# -----------------------------------------------------------------------------
# 3. Algorithme avant-arrière (forward-backward), joueur par joueur
# -----------------------------------------------------------------------------
def forward_backward(le, cls, params):
    # le : log-émissions (T x K) d'un joueur ; cls : classe de transition à chaque t.
    # Renvoie gamma (T x K), xi (T-1 x K x K) et la log-vraisemblance.
    # On travaille avec des probabilités "remises à l'échelle" à chaque pas
    # pour éviter que les nombres deviennent trop petits pour l'ordinateur.
    T = len(le)
    e = np.exp(le - le.max(axis=1, keepdims=True))
    shift = le.max(axis=1)
    A = params["trans"][cls]                       # A[t] : matrice utilisée pour aller de t-1 à t
    alpha = np.zeros((T, K))
    c = np.zeros(T)
    alpha[0] = params["init"][cls[0]] * e[0]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):                          # passe AVANT : ce qu'on sait jusqu'à t
        alpha[t] = (alpha[t - 1] @ A[t]) * e[t]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):                 # passe ARRIÈRE : ce que le futur apprend sur t
        beta[t] = A[t + 1] @ (e[t + 1] * beta[t + 1])
        beta[t] /= c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = (alpha[:-1, :, None] * A[1:] * (e[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
    loglik = np.log(c).sum() + shift.sum()
    return gamma, xi, alpha, loglik


def e_step(df, params, groups):
    # Applique l'avant-arrière à tous les joueurs et cumule les statistiques.
    le = log_emissions(df, params)
    cls = df.cls.to_numpy()
    gamma = np.zeros((len(df), K))
    filtered = np.zeros((len(df), K))
    xi_by_class = np.zeros((N_CLASSES, K, K))
    first_by_class = np.zeros((N_CLASSES, K))
    total = 0.0
    for idx in groups:                             # idx : lignes d'un joueur, dans l'ordre du temps
        g, xi, alpha, ll = forward_backward(le[idx], cls[idx], params)
        gamma[idx], filtered[idx] = g, alpha
        np.add.at(xi_by_class, cls[idx][1:], xi)
        first_by_class[cls[idx][0]] += g[0]
        total += ll
    return gamma, filtered, xi_by_class, first_by_class, total


def _player_groups(df):
    # Liste des indices de lignes de chaque joueur (consécutives, triées par t).
    # On coupe aussi une séquence s'il y a un "trou" de plus d'un trimestre.
    breaks = (df.player_id != df.player_id.shift()) | (df.t - df.t.shift() != 1)
    seq_id = breaks.cumsum().to_numpy()
    _, starts = np.unique(seq_id, return_index=True)
    bounds = list(starts) + [len(df)]
    return [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


# -----------------------------------------------------------------------------
# 4. Résilience
# -----------------------------------------------------------------------------
def resilience(df, gamma, xi_list, params, prior=3.0):
    # Après une baisse (état t-1 < état t-2), le joueur remonte-t-il (état t > état t-1) ?
    # On compare ses remontées OBSERVÉES (selon le modèle) aux remontées ATTENDUES
    # avec les matrices moyennes de son âge et de son poste.
    rows = []
    up = np.triu(np.ones((K, K)), 1)               # masque des transitions vers un état supérieur
    for idx, xi in xi_list:
        if len(idx) < 3:
            continue
        g = gamma[idx]
        level = g @ np.arange(K)                    # niveau moyen attendu à chaque trimestre
        observed, expected = 0.0, 0.0
        for t in range(2, len(idx)):
            if level[t - 1] < level[t - 2] - 0.5:   # il vient de baisser d'au moins un demi-état
                A = params["trans"][df.cls.iat[idx[t]]]
                observed += (xi[t - 1] * up).sum()               # prob. réelle d'être remonté
                expected += (g[t - 1] @ (A * up)).sum()          # prob. attendue de remonter
        rows.append({"player_id": df.player_id.iat[idx[0]], "observed_rebounds": observed,
                     "expected_rebounds": expected})
    r = pd.DataFrame(rows).groupby("player_id").sum().reset_index()
    # Lissage bayésien : avec peu d'épisodes, le ratio reste proche de 1
    r["resilience"] = (r.observed_rebounds + prior) / (r.expected_rebounds + prior)
    return r


# -----------------------------------------------------------------------------
# 5. Entraînement complet
# -----------------------------------------------------------------------------
def fit(cutoff_t=None, max_iter=80, tol=2e-5, verbose=True):
    df = load_panel(cutoff_t)
    groups = _player_groups(df)

    # Point de départ : affectation par règles -> gamma "dur" (0 ou 1)
    s0 = initial_states(df)
    gamma = np.eye(K)[s0]
    xi_by_class = np.zeros((N_CLASSES, K, K))
    first_by_class = np.zeros((N_CLASSES, K))
    cls = df.cls.to_numpy()
    for idx in groups:
        np.add.at(xi_by_class, (cls[idx][1:], s0[idx][:-1], s0[idx][1:]), 1)
        first_by_class[cls[idx][0], s0[idx][0]] += 1
    params = m_step(df, gamma, xi_by_class, first_by_class)

    previous = -np.inf
    for it in range(max_iter):
        gamma, filtered, xi_by_class, first_by_class, ll = e_step(df, params, groups)
        params = m_step(df, gamma, xi_by_class, first_by_class)
        if verbose:
            print(f"itération {it + 1:>2} : log-vraisemblance = {ll:,.0f}")
        if abs(ll - previous) < tol * abs(ll):
            break
        previous = ll

    # Dernier passage pour récupérer les xi par joueur (résilience)
    le = log_emissions(df, params)
    xi_list = []
    for idx in groups:
        g, xi, _, _ = forward_backward(le[idx], cls[idx], params)
        xi_list.append((idx, xi))
    res = resilience(df, gamma, xi_list, params)

    states = df[["player_id", "t", "season", "quarter", "age", "pos_group", "tier", "cls"]].copy()
    for k, name in enumerate(config.STATES):
        states[f"p_{name}"] = gamma[:, k]           # probabilité lissée (avec le futur)
        states[f"f_{name}"] = filtered[:, k]        # probabilité filtrée (passé seulement)
    states["state"] = gamma.argmax(axis=1)
    return params, states, res


def build():
    params, states, res = fit()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"params": params, "resilience": res}, f)
    states.to_parquet(STATES_PATH)
    return params, states, res


def load():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build()
