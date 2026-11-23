# =============================================================================
# simulation/monte_carlo.py — Étape 12 : la simulation Monte Carlo.
#
# On simule N = 10 000 carrières possibles du joueur, trimestre par trimestre,
# sur 24 trimestres (6 ans). Les 10 000 carrières avancent EN MÊME TEMPS
# (calcul vectorisé avec numpy : un tableau de 10 000 cases par variable).
#
# À chaque trimestre, pour chaque carrière simulée :
#   1. ÉTAT   : tirage du nouvel état selon la matrice de transition de son âge
#               et de son poste (chaîne de Markov), ajustée par sa résilience
#   2. JEU    : tirage de son temps de jeu et de sa performance selon son état
#   3. ÉVÉNEMENTS : transfert éventuel (mercato), évolution de son club,
#               contrat (prolongation / départ libre), tournoi, suspensions
#   4. VALEUR : modèle de valeur (étape 9) + son écart persistant de réputation
#               + l'inflation du marché (une trajectoire aléatoire par carrière)
#   5. ÉQUIPE et REVENUS : points apportés, puis revenus par saison
# Le point de départ est son DERNIER état connu (probabilités "filtrées" :
# uniquement avec le passé, sans regarder le futur).
# =============================================================================

import numpy as np
import pandas as pd

import config
from models import collective, events, hmm, revenue, valuation

K = len(config.STATES)
FAMILY_INDEX = hmm.FAMILIES
QUARTER_MATCHES = 13          # matchs de championnat par trimestre, en moyenne
CLUB_MEAN_REVERSION = 0.15    # vitesse à laquelle l'Elo d'un club revient vers son niveau habituel
RESID_SCALE = 0.5             # part du bruit de réputation simulé (calibrée à l'étape 13)


class Context:
    # Charge une fois tous les modèles entraînés et les données des joueurs.
    # fitted : modèles entraînés en mémoire (validation sur le passé) ; sinon
    # on charge les modèles enregistrés sur le disque.
    def __init__(self, fitted=None):
        if fitted:
            self.hmm = {"params": fitted["hmm_params"], "resilience": fitted["resilience"]}
            self.val, self.ev, self.col = fitted["valuation"], fitted["events"], fitted["collective"]
            states = fitted["states"]
            self.panel = valuation.load_training(fitted["cutoff_t"], states)
        else:
            self.hmm = hmm.load()
            self.val = valuation.load()
            self.ev = events.load()
            self.col = collective.load()
            states = pd.read_parquet(hmm.STATES_PATH)
            self.panel = valuation.load_training(states=states)
        self.panel = self.panel.merge(
            states[["player_id", "t", "tier", "cls"] + [f"f_{n}" for n in config.STATES]],
            on=["player_id", "t"], how="left")
        res = self.hmm["resilience"]
        self.resilience = dict(zip(res.player_id, res.resilience))


def _sample_rows(prob_rows, rng):
    # Tire un indice par ligne selon les probabilités de la ligne (vectorisé).
    cum = prob_rows.cumsum(axis=1)
    u = rng.random(len(prob_rows))[:, None]
    return np.minimum((u > cum).sum(axis=1), prob_rows.shape[1] - 1)


def adjust_for_resilience(trans, r):
    # Résilience r > 1 : les remontées sont plus probables, les descentes moins.
    # On multiplie les cases "vers le haut" par r et "vers le bas" par 1/r,
    # puis on renormalise chaque ligne pour qu'elle somme à 1.
    up = np.triu(np.ones((K, K)), 1)
    down = np.tril(np.ones((K, K)), -1)
    adj = trans * (1 + up * (r - 1)) * (1 + down * (1 / r - 1))
    return adj / adj.sum(axis=-1, keepdims=True)


def simulate(player_id, ctx=None, n=config.N_SIMULATIONS, steps=config.N_STEPS, origin_t=None,
             seed=config.RANDOM_SEED, resid_scale=RESID_SCALE, resilience_override=None, live=None):
    # live : mise à jour issue des matchs récents (collectors/live.py) :
    #   {"probs": probabilités d'état mises à jour, "obs": observations par trimestre}
    ctx = ctx or Context()
    rng = np.random.default_rng(seed)
    p = ctx.panel[ctx.panel.player_id == player_id].sort_values("t")
    if origin_t is not None:
        p = p[p.t <= origin_t]
    if p.empty:
        raise ValueError(f"Aucune donnée pour le joueur {player_id}")
    last = p.iloc[-1]
    params, ev, val = ctx.hmm["params"], ctx.ev, ctx.val
    model = val["model"]

    # --- Point de départ ---------------------------------------------------
    f0 = last[[f"f_{s}" for s in config.STATES]].to_numpy(dtype=float)
    f0 = f0 / f0.sum() if np.isfinite(f0).all() and f0.sum() > 0 else np.eye(K)[hmm.TITULAIRE]
    r = resilience_override or ctx.resilience.get(player_id, 1.0)
    family = FAMILY_INDEX[last.pos_group]
    tier = int(last.tier) if pd.notna(last.tier) else 0     # niveau de talent au départ (fixe)
    age0, t0 = float(last.age), int(last.t)
    live_obs = None
    if live is not None and len(live["obs"]):
        # Départ décalé : les trimestres 2026/27 déjà joués sont OBSERVÉS, pas simulés
        live_obs = live["obs"]
        f0 = np.asarray(live["probs"], dtype=float)
        t0 += len(live_obs)
        age0 += len(live_obs) / 4
    state = rng.choice(K, size=n, p=f0)
    elo = np.full(n, float(last.own_elo) if pd.notna(last.own_elo) else 1500.0)
    elo_home = elo.copy()                                  # niveau habituel du club
    years = float(last.contract_years) if pd.notna(last.contract_years) else 2.0
    contract = np.full(n, years)
    recent_perf = list(p.perf_adj.fillna(0).tail(3)) or [0.0]
    recent_share = list(p.minutes_share.fillna(0).tail(3)) or [0.0]
    if live_obs is not None:
        recent_perf = (recent_perf + list(live_obs.perf.fillna(0)))[-3:]
        recent_share = (recent_share + list(live_obs.share))[-3:]
    perf_hist = np.tile(np.array(recent_perf), (n, 1))
    share_hist = np.tile(np.array(recent_share), (n, 1))
    national_last = np.full(n, -99)                        # dernier trimestre joué en sélection
    if last.national:
        national_last[:] = t0

    share_edges = np.array(hmm.SHARE_BINS)
    centering = val["contract_centering"]

    def features(state, perf_recent, share_recent, elo, contract, national_flag, age, log_lag, quarter=4):
        # Valeur (log) prédite par le modèle DYNAMIQUE : part de la valeur d'il y a
        # un an (log_lag) et l'ajuste selon les caractéristiques actuelles.
        feat = pd.DataFrame({
            "log_lag": log_lag,
            **{f"p_{s}": (state == k).astype(float) for k, s in enumerate(config.STATES)},
            "perf_recent": perf_recent, "share_recent": share_recent,
            "family": last.family, "age_c": min(age - 26, 14), "elo_c": (elo - 1600) / 100,
            "league": last.league, "log_views": last.log_views, "has_views": last.has_views,
            "national": national_flag, "eu": last.eu, "left_foot": last.left_foot,
            "height_c": last.height_c, "season": str(val["last_season"]), "quarter": quarter,
        })
        for col, (lo_c, hi_c) in valuation.CONTRACT_BINS.items():
            feat[col] = ((contract > lo_c) & (contract <= hi_c)).astype(float) - centering[col]
        return val["dynamic"].predict(feat).to_numpy()

    # DYNAMIQUE DE LA VALEUR, trimestre par trimestre.
    # Le modèle annuel dit : log V(dans 1 an) = a + b x log V(aujourd'hui) + effets.
    # Sa "valeur d'équilibre" (vers laquelle il tend) est V* = (a + effets) / (1 - b).
    # Chaque trimestre, la valeur parcourt une fraction (1 - b^(1/4)) du chemin
    # vers V* : sur 4 trimestres, on retrouve exactement le comportement annuel.
    # Le bruit trimestriel est choisi pour que, cumulé sur un an, il égale
    # l'erreur annuelle mesurée.
    b_dyn = float(val["dynamic"].params["log_lag"])
    c = b_dyn ** 0.25
    quarterly_sd = val["annual_sd"] / np.sqrt(1 + c ** 2 + c ** 4 + c ** 6)
    log_value = np.full(n, float(np.log(last.market_value)) if last.market_value > 0 else np.log(1e5))
    infl = np.zeros(n)                                     # inflation cumulée depuis le départ

    out = {k: np.zeros((n, steps)) for k in ("state", "share", "perf", "elo", "value_real", "value_nominal",
                                             "points", "moved")}

    for h in range(steps):
        t = t0 + h + 1
        season, quarter = t // 4, t % 4 + 1
        age = age0 + (h + 1) / 4
        age_bin = int(np.digitize(age, hmm.AGE_BINS[1:-1]))
        cls = hmm.class_index(age_bin, family, tier)

        # 1. Nouvel état (chaîne de Markov ajustée par la résilience)
        A = adjust_for_resilience(params["trans"][cls], r)
        prev_state = state
        state = _sample_rows(A[state], rng)

        # 2. Temps de jeu et performance selon l'état
        b = _sample_rows(params["share"][state], rng)
        lo = np.clip(share_edges[b], 0, 1)
        hi = np.clip(share_edges[b + 1], 0, 1)
        share = lo + rng.random(n) * (hi - lo)
        perf = rng.normal(params["perf_mu"][state], params["perf_sd"][state])
        # Suspensions : cartons rouges ~ loi de Poisson selon les minutes jouées
        minutes = share * QUARTER_MATCHES * 90
        reds = rng.poisson(ev["red_rate_per_1000"] * minutes / 1000)
        share = np.clip(share - reds * events.SUSPENSION_MATCHES_PER_RED / QUARTER_MATCHES, 0, 1)

        # 3a. Transfert (probabilité selon état, âge, période du mercato)
        p_move = ev["move"][prev_state, age_bin, quarter - 1]
        moved = rng.random(n) < p_move
        # 3b. Contrat : on vieillit d'un trimestre ; prolongation possible à < 18 mois
        contract -= 0.25
        renew_year = np.vectorize(ev["contract"]["renewal_prob_per_year"].get)(np.array(config.STATES)[state])
        renew = (contract <= 1.5) & (contract > 0) & (rng.random(n) < 1 - (1 - renew_year) ** 0.25)
        extra = ev["contract"]["renewal_years_young"] if age < ev["contract"]["young_age"] else ev["contract"]["renewal_years"]
        contract = np.where(renew, contract + extra, contract)
        free_agent = contract <= 0                          # fin de contrat : départ libre
        moved = moved | free_agent
        contract = np.where(moved, ev["contract"]["new_contract_years_after_transfer"], contract)
        # 3c. Club : saut d'Elo en cas de transfert, sinon retour progressif vers son niveau
        jumps = np.array([rng.choice(ev["elo_jump"][s]) for s in prev_state[moved]]) if moved.any() else []
        elo_home = np.where(moved, elo_home, elo_home)
        if moved.any():
            elo_home[moved] = elo[moved] + jumps
        elo = elo + CLUB_MEAN_REVERSION * (elo_home - elo) + rng.normal(0, ev["elo_sd"] * 0.5, n)
        elo = np.where(moved, elo_home, elo)
        # 3d. Grand tournoi (Euro 2028, Coupe du monde 2030)
        if (season, quarter) in ev["tournament_quarters"]:
            plays = rng.random(n) < ev["national"][state, age_bin]
            national_last = np.where(plays, t, national_last)

        # 4. Valeur marchande
        perf_hist = np.column_stack([perf_hist, perf])[:, -4:]
        share_hist = np.column_stack([share_hist, share])[:, -4:]
        # a + effets = prédiction avec log_lag = 0 ; puis valeur d'équilibre
        effects = features(state, perf_hist.mean(axis=1), share_hist.mean(axis=1), elo, contract,
                           (t - national_last <= 8).astype(int), age, np.zeros(n), quarter)
        equilibrium = effects / (1 - b_dyn)
        log_value = (log_value + (1 - c) * (equilibrium - log_value)
                     + resid_scale * quarterly_sd * rng.normal(0, 1, n))
        infl = infl + val["inflation"]["drift"] / 4 + rng.normal(0, val["inflation"]["vol"] / 2, n)

        out["state"][:, h] = state
        out["share"][:, h] = share
        out["perf"][:, h] = perf
        out["elo"][:, h] = elo
        out["value_real"][:, h] = np.exp(log_value)
        out["value_nominal"][:, h] = np.exp(log_value + infl)
        out["points"][:, h] = collective.points_added(perf, share, ctx.col["beta"]) / 4
        out["moved"][:, h] = moved

    out["revenue"] = _revenues(out, t0, last.league)
    out["meta"] = {"player_id": player_id, "name": last.get("name"), "t0": t0, "age0": age0,
                   "value0": float(last.market_value), "club_elo0": float(elo_home.mean()),
                   "resilience": r, "start_probs": f0, "steps": steps, "n": n, "live_obs": live_obs,
                   "pos_group": last.pos_group, "tier": tier}
    return out


def _revenues(out, t0, league):
    # Regroupe les trimestres par saison (4 trimestres) et calcule les revenus.
    n, steps = out["state"].shape
    seasons = steps // 4
    result = {}
    for s in range(seasons):
        cols = slice(s * 4, s * 4 + 4)
        st = out["state"][:, cols].astype(int)
        dominant = np.array([np.bincount(row, minlength=K).argmax() for row in st])
        points = out["points"][:, cols].sum(axis=1)
        club_elo = out["elo"][:, cols].mean(axis=1)
        rev = revenue.season_revenue(dominant, points, club_elo, league)
        for k, v in rev.items():
            result.setdefault(k, np.zeros((n, seasons)))[:, s] = v
    return result


def summarize(values, bands=config.CONFIDENCE_BANDS):
    # Médiane et bandes de confiance par trimestre (pour les graphiques en éventail).
    out = {"median": np.median(values, axis=0)}
    for b in bands:
        out[f"low_{int(b * 100)}"] = np.quantile(values, (1 - b) / 2, axis=0)
        out[f"high_{int(b * 100)}"] = np.quantile(values, 1 - (1 - b) / 2, axis=0)
    return out
