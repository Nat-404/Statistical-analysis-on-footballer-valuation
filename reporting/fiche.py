# =============================================================================
# reporting/fiche.py — La fiche de résultats (en anglais) : 2 pages, style article.
#
#   1. Player profile + Table 1 (valeur à T0, statistiques de la saison passée)
#   2. Table 2 : hypothèses de la simulation
#   3. Results : Figure 1 (les 10 000 trajectoires de valeur), Figure 2
#      (histogramme des valeurs finales), Table 3, Figure 3 (temps de jeu),
#      Table 4 — chacun suivi d'une ou deux phrases d'analyse.
#
# T0 = date de la simulation : toutes les carrières partent de la valeur
# marchande actuelle du joueur (dernière valeur Transfermarkt connue).
# Graphiques : les 10 000 trajectoires en bleu pâle ; en bleu foncé, les deux
# carrières simulées les plus proches de la moyenne et de la médiane (de
# vraies trajectoires, qui gardent donc leur zigzag).
# Police : Latin Modern, la police de LaTeX (celle du papier de référence).
# =============================================================================

import datetime as dt
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

import config
from collectors import transfermarkt as tm
from models import hmm

# --- Police Latin Modern (fichiers OTF du CTAN, dans reporting/fonts) --------
for font in (Path(__file__).parent / "fonts").glob("*.otf"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({"font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

A4 = (8.27, 11.69)
LEFT, RIGHT = 0.1, 0.9
DARK, ACCENT = "#111111", "#b03a2e"
PALE_BLUE = "#8fb8e8"                   # les 10 000 trajectoires
MEAN_BLUE, MEDIAN_BLUE = "#0b2a66", "#1f4fa8"   # moyenne et médiane (bleu foncé)
STATES_EN = ["Injured", "Marginal", "Rotation", "Starter", "Strong", "Elite"]
POSITIONS = {"Goalkeeper": "goalkeeper", "Centre-Back": "centre-back", "Left-Back": "left-back",
             "Right-Back": "right-back", "Defensive Midfield": "defensive midfielder",
             "Central Midfield": "central midfielder", "Attacking Midfield": "attacking midfielder",
             "Left Midfield": "left midfielder", "Right Midfield": "right midfielder", "Left Winger": "left winger",
             "Right Winger": "right winger", "Centre-Forward": "centre-forward", "Second Striker": "second striker"}
NATIONALITIES = {"England": "English", "France": "French", "Spain": "Spanish", "Germany": "German",
                 "Italy": "Italian", "Brazil": "Brazilian", "Argentina": "Argentine", "Portugal": "Portuguese",
                 "Netherlands": "Dutch", "Belgium": "Belgian", "Morocco": "Moroccan", "Senegal": "Senegalese",
                 "Norway": "Norwegian", "Croatia": "Croatian", "Denmark": "Danish", "Scotland": "Scottish",
                 "Wales": "Welsh", "Switzerland": "Swiss", "United States": "American", "Uruguay": "Uruguayan",
                 "Colombia": "Colombian", "Japan": "Japanese", "Korea, South": "South Korean", "Poland": "Polish"}
TIERS = ["bottom 60%", "top 10–40%", "top 10%"]


# --- Petits outils de mise en forme -------------------------------------------
def eur(x):
    # 30 000 000 -> "€30m" ; 2 400 000 -> "€2.4m" ; 450 000 -> "€450k"
    if x >= 9.95e6:
        return f"€{x / 1e6:.0f}m"
    if x >= 1e6:
        return f"€{x / 1e6:.1f}m"
    return f"€{x / 1e3:.0f}k"


def pct(p):
    return f"{p * 100:.0f}%"


def date_en(d):
    return pd.Timestamp(d).strftime("%-d %B %Y")


class Page:
    # Une page A4 sur laquelle on "écrit" de haut en bas, comme dans un document.
    def __init__(self, pdf, number):
        self.pdf, self.fig, self.y = pdf, plt.figure(figsize=A4), 0.94
        self.fig.text(0.5, 0.02, str(number), ha="center", fontsize=8.5)

    def text(self, s, size=9.5, width=118, gap=0.004, **kw):
        lines = textwrap.fill(s, width)
        n = lines.count("\n") + 1
        self.fig.text(LEFT, self.y, lines, va="top", fontsize=size, linespacing=1.35, **kw)
        self.y -= n * size * 1.35 / 72 / A4[1] + gap

    def heading(self, s):
        self.y -= 0.008
        self.fig.text(LEFT, self.y, s, va="top", fontsize=10.5, weight="bold")
        self.y -= 0.024

    def rule(self, lw=0.8):
        self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [self.y, self.y], color=DARK, lw=lw))

    def table(self, caption, header, rows, widths, size=8.6, row_h=0.0175, aligns=None):
        # Tableau au style "booktabs" : 3 filets horizontaux, aucun trait vertical.
        self.fig.text(0.5, self.y, caption, ha="center", va="top", fontsize=8.8)
        self.y -= 0.02
        xs = LEFT + np.concatenate([[0], np.cumsum(widths[:-1])]) * (RIGHT - LEFT)
        aligns = aligns or ["left"] + ["right"] * (len(header) - 1)

        def line(y, lw):
            self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [y, y], color=DARK, lw=lw))

        def cells(values, y, weight="normal"):
            for x, w, v, a in zip(xs, widths, values, aligns):
                xx = x + 0.005 if a == "left" else x + w * (RIGHT - LEFT) - 0.005
                self.fig.text(xx, y, v, ha=a, va="center", fontsize=size, weight=weight)

        line(self.y, 1.0)
        if any(header):                                  # tableau sans en-tête : un seul filet en haut
            self.y -= row_h * 0.6
            cells(header, self.y, "bold")
            self.y -= row_h * 0.6
            line(self.y, 0.5)
        for r in rows:
            self.y -= row_h * 0.62
            cells(r, self.y)
            self.y -= row_h * 0.38
        self.y -= 0.004
        line(self.y, 1.0)
        self.y -= 0.014

    def axes(self, height):
        ax = self.fig.add_axes([LEFT + 0.05, self.y - height, RIGHT - LEFT - 0.05, height])
        self.y -= height + 0.045                         # place pour le titre de l'axe horizontal
        return ax

    def caption(self, s):
        self.fig.text(0.5, self.y, s, ha="center", va="top", fontsize=8.8, style="italic")
        self.y -= 0.022

    def save(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


# -----------------------------------------------------------------------------
# Graphiques
# -----------------------------------------------------------------------------
def representative(paths, target, log=False):
    # La carrière simulée la plus proche d'une courbe cible (moyenne ou médiane) :
    # une VRAIE trajectoire, qui garde donc son zigzag.
    a, b = (np.log(np.maximum(paths, 1e-9)), np.log(np.maximum(target, 1e-9))) if log else (paths, target)
    return paths[np.argmin(((a - b) ** 2).sum(axis=1))]


def monte_carlo_paths(ax, years, paths, ylabel, log=False, ylim=None):
    # Les 10 000 trajectoires en bleu pâle, puis en bleu foncé les carrières
    # les plus proches de la moyenne et de la médiane.
    segments = np.stack([np.broadcast_to(years, paths.shape), paths], axis=-1)
    ax.add_collection(LineCollection(segments, colors=PALE_BLUE, linewidths=0.25, alpha=0.05, rasterized=True))
    # 150 trajectoires tirées au hasard, un peu plus marquées, pour voir le zigzag de chaque carrière
    pick = np.random.default_rng(0).choice(len(paths), min(150, len(paths)), replace=False)
    ax.add_collection(LineCollection(segments[pick], colors=PALE_BLUE, linewidths=0.45, alpha=0.55, rasterized=True))
    ax.plot(years, representative(paths, paths.mean(axis=0), log), color=MEAN_BLUE, lw=1.5, label="mean path")
    ax.plot(years, representative(paths, np.median(paths, axis=0), log), color=MEDIAN_BLUE, lw=1.5, ls="--",
            label="median path")
    ax.set_xlim(0, years[-1])
    ax.set_ylim(*(ylim or (0, np.quantile(paths, 0.99))))
    ax.set_xticks(np.arange(0, years[-1] + 0.01, 1))
    ax.set_xlabel("years after T0 (one step = one quarter)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.tick_params(labelsize=8.5)


def histogram(ax, final, v0):
    # Distribution des valeurs à six ans, avec moyenne, médiane et V0
    cap = np.quantile(final, 0.99)
    ax.hist(np.minimum(final, cap), bins=60, color=PALE_BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(final.mean(), color=MEAN_BLUE, lw=1.5, label=f"mean {eur(final.mean() * 1e6)}")
    ax.axvline(np.median(final), color=MEDIAN_BLUE, lw=1.5, ls="--", label=f"median {eur(np.median(final) * 1e6)}")
    ax.axvline(v0, color=ACCENT, lw=1, ls=":", label=f"V0 {eur(v0 * 1e6)}")
    ax.set_xlim(0, cap)
    ax.set_xlabel("market value six years after T0 (€m, T0 prices; last bar = top 1%)", fontsize=9)
    ax.set_ylabel("number of careers", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.tick_params(labelsize=8.5)


BACKTEST_BANDS = [(5e6, "<5M", "below €5m"), (20e6, "5-20M", "€5–20m"), (50e6, "20-50M", "€20–50m"),
                  (float("inf"), ">50M", "above €50m")]


def reliability(v0):
    # Résultats de la validation sur le passé (2020 -> 2026), au niveau de bruit utilisé,
    # au total et pour la tranche de valeur du joueur analysé.
    path = config.PROCESSED_DIR / "backtest.parquet"
    if not path.exists():
        return None
    from evaluation import backtest as bt
    from simulation import monte_carlo as mc
    results = pd.read_parquet(path)
    results = results[results.scale == mc.RESID_SCALE] if mc.RESID_SCALE in set(results.scale) else results
    by_scale, by_band, _ = bt.report(results)
    row = by_scale.iloc[0]
    key, label = next((k, lab) for limit, k, lab in BACKTEST_BANDS if v0 < limit)
    band = by_band.loc[key] if key in by_band.index else None
    return {"coverage": row.couverture_80, "bias": row.biais_log, "players": int(row.joueurs),
            "band_label": label, "band_bias": None if band is None else band.biais_log,
            "band_coverage": None if band is None else band.couverture_80}


# -----------------------------------------------------------------------------
def last_season_stats(profile, panel_rows):
    # Statistiques de la dernière saison complète (Transfermarkt + FotMob si disponible)
    season = config.CURRENT_SEASON - 1
    q = panel_rows[panel_rows.season == season]
    stats = {"season": season, "matches": int(q.matches.sum()), "starts": int(q.starts.sum()),
             "minutes": float(q.minutes.sum()), "goals": int(q.goals.sum()), "assists": int(q.assists.sum()),
             "share": float(q.minutes_league.sum() / max((q.team_league_games.fillna(0) * 90).sum(), 1))}
    league = q.league.mode().iloc[0] if len(q) and q.league.notna().any() else None
    if league in config.LEAGUES:
        from collectors import live
        age = (pd.Timestamp(f"{season}-10-01") - pd.Timestamp(profile["date_of_birth"])).days / 365.25
        fm_stats = live.season_stats(profile["name"], age, league, season)
        if fm_stats:
            stats.update({f"fm_{k}": v for k, v in fm_stats.items()})
    return stats


def build_fiche(player_id, ctx, sim, live=None, path=None, sim_date=None, press=None):
    # press : contrôle presse GDELT (collectors/gdelt.py), affiché page 2 à titre d'information
    sim_date = pd.Timestamp(sim_date or dt.date.today())
    profile = tm.query(
        "SELECT name, date_of_birth, sub_position, current_club_name, country_of_citizenship, foot, height_in_cm, "
        "contract_expiration_date, international_caps, highest_market_value_in_eur FROM players WHERE player_id = ?",
        [player_id]).iloc[0].to_dict()
    rows = ctx.panel[ctx.panel.player_id == player_id].sort_values("t")
    last = rows.iloc[-1]
    meta = sim["meta"]
    v0, age0 = meta["value0"], (sim_date - pd.Timestamp(profile["date_of_birth"])).days / 365.25
    stats = last_season_stats(profile, rows)
    youth = tm.query("SELECT from_club_name FROM transfers WHERE player_id = ? AND "
                     "regexp_matches(from_club_name, 'Yth|Youth|U1[0-9]|U2[0-3]') ORDER BY transfer_date LIMIT 1",
                     [player_id])
    value_hist = tm.player_valuations(player_id)

    # Toutes les trajectoires partent de V0 à T0 : on ajoute T0 en tête
    n = len(sim["value_real"])
    years = np.arange(sim["value_real"].shape[1] + 1) / 4
    values = np.column_stack([np.full(n, v0), sim["value_real"]])
    nominal = np.column_stack([np.full(n, v0), sim["value_nominal"]])
    # Temps de jeu en moyenne glissante sur 4 trimestres, en incluant les 3 derniers trimestres RÉELS
    past = rows.minutes_share.fillna(0).tail(3).to_numpy()
    if meta.get("live_obs") is not None and len(meta["live_obs"]):
        past = np.concatenate([past, meta["live_obs"].share.to_numpy()])[-3:]
    padded = np.column_stack([np.tile(past, (n, 1)), sim["share"]])
    share = pd.DataFrame(padded).T.rolling(4, min_periods=1).mean().T.to_numpy()[:, len(past) - 1:]

    path = path or config.OUTPUTS_DIR / f"{profile['name'].replace(' ', '_')}_projection.pdf"
    with PdfPages(path) as pdf:
        _page_one(pdf, profile, stats, youth, value_hist, last, meta, v0, age0, sim_date, ctx, years, values)
        _page_two(pdf, sim, years, values, nominal, share, v0, press)
    return path


def _page_one(pdf, profile, stats, youth, value_hist, last, meta, v0, age0, sim_date, ctx, years, values):
    pg = Page(pdf, 1)
    f = pg.fig
    f.text(0.5, 0.955, f"Market Value Projection: {profile['name']}", ha="center", fontsize=15, weight="bold")
    f.text(0.5, 0.93, f"Hidden Markov model and Monte Carlo simulation  ·  T0 = {date_en(sim_date)}  ·  "
           f"{config.HORIZON_YEARS}-year horizon  ·  {meta['n']:,} simulated careers",
           ha="center", fontsize=9.5, style="italic")
    pg.y = 0.91
    pg.rule(0.6)
    pg.y -= 0.012

    # 1. Player profile
    pg.heading("1.  Player profile")
    nat = NATIONALITIES.get(profile["country_of_citizenship"], profile["country_of_citizenship"])
    pos = POSITIONS.get(profile["sub_position"], profile["sub_position"])
    foot = {"right": "right-footed", "left": "left-footed", "both": "two-footed"}.get(profile["foot"], "")
    contract = pd.Timestamp(profile["contract_expiration_date"]) if pd.notna(profile["contract_expiration_date"]) else None
    article = "an" if f"{age0:.0f}".startswith("8") or f"{age0:.0f}" in ("11", "18") else "a"
    s = (f"{profile['name']} is {article} {age0:.0f}-year-old {nat} {pos} ({foot}, {profile['height_in_cm']:.0f} cm) under "
         f"contract with {profile['current_club_name']}" + (f" until June {contract.year}" if contract is not None else "") + ". ")
    if len(youth):
        s += f"A product of the {youth.from_club_name.iloc[0].replace(' Yth.', '').replace(' Youth', '')} academy, "
        s += (f"he has {int(profile['international_caps'])} international caps at all levels. "
              if profile["international_caps"] else "")
    elif profile["international_caps"]:
        s += f"He has {int(profile['international_caps'])} international caps at all levels. "
    if len(value_hist):
        first = value_hist.iloc[0]
        s += (f"His market value rose from {eur(first.market_value_in_eur)} ({date_en(first.date)}) to {eur(v0)} at the "
              f"latest Transfermarkt valuation ({date_en(last.value_date)}); this is the starting value V0 of every "
              f"simulated career.")
    pg.text(s)
    pg.y -= 0.006

    # Table 1
    st = stats
    fm = "fm_xg" in st
    left = [("Market value at T0 (V0)", eur(v0)), ("Highest market value", eur(profile["highest_market_value_in_eur"])),
            ("Matches (starts)", f"{st['matches']} ({st['starts']})"), ("Minutes played", f"{st['minutes']:,.0f}")]
    right = [("Share of league minutes", pct(min(st["share"], 1))),
             ("Goals / assists", f"{st['goals']} / {st['assists']}"),
             ("xG / xA (FotMob)", f"{st['fm_xg']:.1f} / {st['fm_xa']:.1f}" if fm else "n/a"),
             ("Chances created / successful dribbles", f"{st['fm_chances']:.0f} / {st['fm_dribbles']:.0f}" if fm else "n/a")]
    season = st["season"]
    pg.table(f"Table 1 — Market value at T0 and {season}/{str(season + 1)[2:]} season statistics (all competitions)",
             ["", "", "", ""], [[a, b, c, d] for (a, b), (c, d) in zip(left, right)],
             [0.30, 0.15, 0.40, 0.15], aligns=["left", "right", "left", "right"])
    obs = meta.get("live_obs")
    if obs is not None and len(obs):
        absent = int(obs.missed_injury.sum())
        pg.text(f"Current season (FotMob, at T0): {int(obs.matches.sum())} matches, {obs.minutes.sum():.0f} minutes, "
                f"{obs.goals.sum():.0f} goal{'s' if obs.goals.sum() != 1 else ''}, {obs.xg.sum():.1f} xG"
                + (f"; {absent} match{'es' if absent > 1 else ''} missed through injury or absence." if absent else ".")
                + " These quarters are observed, not simulated, and update the starting state.", size=9)
    pg.y -= 0.004

    # 2. Assumptions
    pg.heading("2.  Simulation assumptions")
    ev, val = ctx.ev, ctx.val
    start = meta["start_probs"]
    order = np.argsort(start)[::-1]
    start_txt = ", ".join(f"{STATES_EN[k]} {pct(start[k])}" for k in order[:3] if start[k] >= 0.05)
    age_bin = int(np.digitize(meta["age0"], hmm.AGE_BINS[1:-1]))
    p_move_year = 1 - np.prod(1 - ev["move"][int(order[0]), age_bin, :])
    dest = ev["destination"][age_bin, meta["tier"]]
    from simulation import calibration
    drift = ctx.drift[age_bin, int(calibration.value_band(v0))] if ctx.drift is not None else 0.0
    renew = ev["contract"]["renewal_prob_per_year"]
    from simulation import monte_carlo as mc
    infl = val["inflation"]
    assumptions = [
        ["Starting value V0 and date T0", f"{eur(v0)} · {date_en(sim_date)}", "Transfermarkt"],
        ["Horizon, time step, simulations", f"6 years · 24 quarters · {meta['n']:,}", "—"],
        ["Starting level (hidden states)", start_txt, "filtered HMM + FotMob 26/27"],
        ["Talent tier (value rank for age)", TIERS[meta["tier"]], "value rank at T0"],
        ["State transitions", "84 matrices: age × position × talent", "Baum–Welch, 2014–2026"],
        ["Resilience (rebound after a decline)", f"{meta['resilience']:.2f} (1 = average)", "Bayesian estimate"],
        ["Transfer within a year / leaves top five if moved", f"{pct(p_move_year)} / {pct(dest[2])}", "Transfermarkt 2014–2026"],
        ["Contract renewal per year", f"{pct(renew['Rotation'])} (Rotation) to {pct(renew['Élite'])} (Elite)", "assumption"],
        ["Market inflation", f"+{(np.exp(infl['drift']) - 1) * 100:.1f}%/yr (s.d. {infl['vol'] * 100:.1f}%)", "season effects"],
        ["Market noise · drift correction at T0", f"{mc.RESID_SCALE:.2f} · {(np.exp(drift) - 1) * 100:+.1f}%/yr",
         "age × value band"],
    ]
    pg.table("Table 2 — Assumptions", ["Assumption", "Value", "Source"], assumptions,
             [0.42, 0.34, 0.24], aligns=["left", "left", "left"], size=8.4, row_h=0.0168)
    pg.text("Transition, transfer and inflation parameters are estimated on 9,391 players of the five major European "
            "leagues; only contract renewals rest on an assumption, since no public source records contract histories. "
            "Values are expressed at T0 market prices unless stated otherwise.", size=8.8)

    # 3. Results — Figure 1
    pg.heading("3.  Simulation results")
    ax = pg.axes(0.2)
    monte_carlo_paths(ax, years, values / 1e6, "market value (€m, T0 prices)", log=True)
    ax.axhline(v0 / 1e6, color=ACCENT, lw=0.8, ls=":")
    ax.text(years[-1], v0 / 1e6, "V0 ", color=ACCENT, fontsize=8.5, va="bottom", ha="right")
    pg.caption("Figure 1 — The 10,000 simulated market-value paths (pale blue); in dark blue, the simulated careers "
               "closest to the mean and to the median.")
    final = values[:, -1]
    mean6, med6, med1 = final.mean(), np.median(final), np.median(values[:, 4])
    pg.text(f"The median value moves from {eur(v0)} to {eur(med1)} after one year and to {eur(med6)} after six years. "
            f"Every path zigzags because each quarter brings new random events (form, injuries, transfers, market "
            f"noise); the dark paths are real simulated careers chosen for being closest to the centre of the "
            f"distribution.", size=9)
    pg.save()


def _page_two(pdf, sim, years, values, nominal, share, v0, press=None):
    pg = Page(pdf, 2)
    pg.y = 0.955
    final = values[:, -1]

    # Figure 2 — histogram
    ax = pg.axes(0.125)
    histogram(ax, final / 1e6, v0 / 1e6)
    pg.caption("Figure 2 — Distribution of the simulated market value six years after T0 (10,000 careers).")
    skew = "well above" if final.mean() > 1.15 * np.median(final) else "close to"
    pg.text(f"The distribution is skewed to the right: the mean ({eur(final.mean())}) is {skew} the median "
            f"({eur(np.median(final))}) because a minority of careers become exceptional. The value ends above V0 in "
            f"{pct((final > v0).mean())} of careers and below half of V0 in {pct((final <= v0 / 2).mean())}.", size=9)
    pg.y -= 0.006

    # Table 3
    rows = []
    for label, h in [("T0 + 1 year", 4), ("T0 + 2 years", 8), ("T0 + 3 years", 12), ("T0 + 6 years", 24)]:
        v = values[:, h]
        rows.append([label, eur(v.mean()), eur(np.median(v)), eur(np.quantile(v, 0.1)), eur(np.quantile(v, 0.9)),
                     pct((v > v0).mean()), eur(np.median(nominal[:, h]))])
    pg.table("Table 3 — Distribution of the simulated market value",
             ["Horizon", "Mean", "Median", "P10", "P90", "P(V > V0)", "Median incl. inflation"], rows,
             [0.16, 0.11, 0.11, 0.11, 0.11, 0.14, 0.26])
    w1 = np.quantile(values[:, 4], 0.9) / max(np.quantile(values[:, 4], 0.1), 1)
    w6 = np.quantile(final, 0.9) / max(np.quantile(final, 0.1), 1)
    pg.text(f"The P10–P90 range widens from a factor of {w1:.1f} after one year to {w6:.1f} after six: uncertainty "
            f"grows with the horizon, as expected for a career exposed to injuries, transfers and changes in level.",
            size=9)
    pg.y -= 0.014

    # Figure 3 — playing time
    ax = pg.axes(0.105)
    monte_carlo_paths(ax, years, share * 100, "playing time (%)", ylim=(0, 100))
    pg.caption("Figure 3 — The 10,000 simulated paths of the share of league minutes (4-quarter average), "
               "with the mean and median paths.")
    s0, s2, s6 = share[0, 0], np.median(share[:, 8]), np.median(share[:, -1])
    pg.text(f"Median playing time moves from {pct(s0)} (last four quarters) to {pct(s2)} in two years and {pct(s6)} "
            f"in six. Playing time is the main driver of value: a player who does not become a regular starter sees "
            f"his value converge towards that of rotation players.", size=9)
    pg.y -= 0.006

    # Table 4
    st = sim["state"]
    rev_total = sum(sim["revenue"].values())
    p_starter = (st[:, 7] >= 3).mean()
    p_move = sim["moved"].any(axis=1).mean()
    events = [
        ["At least one injured quarter within a year", pct((st[:, :4] == 0).any(axis=1).mean())],
        ["Starter or better two years after T0", pct(p_starter)],
        ["Star status (Strong or Elite half of the time)", pct(((st >= 4).mean(axis=1) >= 0.5).mean())],
        ["At least one transfer or loan within six years", pct(p_move)],
        ["Outside the five major leagues after six years", pct(sim["meta"].get("left_top5", np.nan))],
        ["Revenue generated for the club, median season 1 / season 6",
         f"{eur(np.median(rev_total[:, 0]))} / {eur(np.median(rev_total[:, -1]))}"],
    ]
    pg.table("Table 4 — Event probabilities over the simulation horizon", ["Event", "Probability"], events, [0.75, 0.25])
    pg.text(f"The central scenario is that of a player who {'becomes' if p_starter >= 0.5 else 'is not yet'} a regular "
            f"starter in the short term ({pct(p_starter)} chance of being one in two years), with a transfer or loan "
            f"{'very likely' if p_move > 0.8 else 'likely' if p_move > 0.5 else 'possible'} ({pct(p_move)}) within six "
            f"years. Revenue figures are orders of magnitude based on public data (UEFA, Premier League, Deloitte) "
            f"and explicit assumptions.", size=9)

    rel = reliability(v0)
    if rel:
        pg.y -= 0.004
        band = (f" For players valued {rel['band_label']} in 2020: coverage {pct(rel['band_coverage'])}, "
                f"bias {rel['band_bias']:+.2f}." if rel["band_bias"] is not None else "")
        pg.text(f"Reliability — Out-of-sample validation (models re-estimated in July 2020 and compared with true "
                f"values in June 2026, {rel['players']} players): the P10–P90 interval contains the true value in "
                f"{pct(rel['coverage'])} of cases (2020 prices); mean bias {rel['bias']:+.2f} (log of "
                f"forecast median over truth)."
                + band, size=8.2, style="italic", color="#333333")
    if press is not None:
        from collectors import gdelt
        pg.y -= 0.004
        pg.text(gdelt.summary_en(press), size=8.2, style="italic", color="#333333")
    pg.save()
