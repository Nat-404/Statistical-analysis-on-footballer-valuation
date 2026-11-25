# =============================================================================
# paper/make_figures.py — Génère les figures et la section "Results" du papier
# directement à partir des modèles entraînés et de la validation : les chiffres
# du papier sont ainsi toujours reproductibles.
#     python -m paper.make_figures
# =============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from evaluation import backtest as bt
from features import elo
from models import hmm
from simulation import monte_carlo as mc

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
STATES_EN = ["Injured", "Marginal", "Rotation", "Starter", "Strong", "Elite"]
from matplotlib import font_manager
for _font in (Path(__file__).resolve().parent.parent / "reporting" / "fonts").glob("*.otf"):
    font_manager.fontManager.addfont(str(_font))
plt.rcParams.update({"font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})

# Résultats des deux premières validations (avant corrections), documentés dans le projet
HISTORY = [
    ("Naive simulation", 0.81, 0.626, 0.995, 0.24),
    ("+ league changes, yearly talent tier", 0.70, 0.704, 0.896, 0.28),
    ("+ drift calibration", 0.39, 0.827, 0.22, 0.45),
    ("+ complete FotMob data", 0.30, 0.796, -0.03, 0.51),
]


def fig_transitions(params):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, tier, title in zip(axes, [0, 2], ["Bottom 60% of value", "Top 10% of value"]):
        m = params["trans"][hmm.class_index(2, 3, tier)]
        ax.imshow(m, cmap="Greys", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if m[i, j] > 0.55 else "black")
        ax.set_xticks(range(6), STATES_EN, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(6), STATES_EN, fontsize=7)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("state this quarter")
    fig.supxlabel("state next quarter", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "transitions.png")
    plt.close(fig)
    return params["trans"][hmm.class_index(2, 3, 2)]


def fig_elo():
    res = pd.read_parquet(elo.OUT_PATH)
    lg = res[res.competition_type == "domestic_league"].dropna(subset=["expected_home"])
    long = pd.concat([lg[["date", "competition_id", "elo_home_pre"]].set_axis(["date", "c", "e"], axis=1),
                      lg[["date", "competition_id", "elo_away_pre"]].set_axis(["date", "c", "e"], axis=1)])
    long["year"] = pd.to_datetime(long.date).dt.year
    names = {l["transfermarkt"]: l["name"] for l in config.LEAGUES.values()}
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    styles = ["-", "--", "-.", ":", (0, (5, 1))]
    for (code, name), ls in zip(names.items(), styles):
        s = long[long.c == code].groupby("year").e.mean()
        ax.plot(s.index, s.values, linestyle=ls, color="black", lw=1.2, label=name)
    ax.set_ylabel("mean Elo rating")
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(FIG / "elo.png")
    plt.close(fig)


def player_example(ctx, name):
    from collectors import live
    last = ctx.panel[ctx.panel.name == name].sort_values("t").iloc[-1]
    f = last[[f"f_{s}" for s in config.STATES]].to_numpy(float)
    probs, obs, added = live.update_player(name, last.age + 0.25, last.pos_group, f / f.sum(),
                                           ctx.hmm["params"], int(last.cls), refresh=False)
    sim = mc.simulate(int(last.player_id), ctx, live={"probs": probs, "obs": obs} if added else None)
    real, nominal = sim["value_real"] / 1e6, sim["value_nominal"] / 1e6
    years = np.arange(1, real.shape[1] + 1) / 4
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    for lo, hi, a in [(0.05, 0.95, 0.15), (0.10, 0.90, 0.25), (0.25, 0.75, 0.40)]:
        ax.fill_between(years, np.quantile(real, lo, axis=0), np.quantile(real, hi, axis=0), color="black", alpha=a, lw=0)
    ax.plot(years, np.median(real, axis=0), color="black", lw=1.5, label="median (real)")
    ax.plot(years, np.median(nominal, axis=0), "--", color="black", lw=1, label="median (nominal)")
    ax.axhline(sim["meta"]["value0"] / 1e6, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("years ahead"); ax.set_ylabel("market value (M€)")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    st = sim["state"].astype(int)
    shares = np.array([(st == k).mean(axis=0) for k in range(6)])
    greys = ["#000000", "#bbbbbb", "#888888", "#555555", "#333333", "#dddddd"]
    ax.stackplot(years, shares, colors=greys, labels=STATES_EN, edgecolor="white", lw=0.3)
    ax.set_xlabel("years ahead"); ax.set_ylabel("share of careers")
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "example.png")
    plt.close(fig)
    final = real[:, -1]
    return {"name": name, "age": sim["meta"]["age0"], "v0": sim["meta"]["value0"] / 1e6,
            "y1": np.median(real[:, 3]), "y3": np.median(real[:, 11]), "y6": np.median(final),
            "lo": np.quantile(final, 0.1), "hi": np.quantile(final, 0.9),
            "p100": (real[:, 15] >= 100).mean(), "obs": obs, "start": dict(zip(STATES_EN, sim["meta"]["start_probs"]))}


def fig_backtest(results, scale):
    r = results[results.scale == scale]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    axes[0].hist(r.pit, bins=10, range=(0, 1), color="grey", edgecolor="white")
    axes[0].axhline(len(r) / 10, color="black", ls="--", lw=0.8)
    axes[0].set_xlabel("rank of the true value in the forecast (PIT)"); axes[0].set_ylabel("players")
    axes[1].loglog(r.value_2020 / 1e6, r.truth / 1e6, "o", ms=2.5, color="black", alpha=0.5, label="true 2026")
    axes[1].loglog(r.value_2020 / 1e6, r["median"] / 1e6, "x", ms=2.5, color="grey", alpha=0.7, label="forecast median")
    lim = [0.5, 300]
    axes[1].plot(lim, lim, color="grey", lw=0.6, ls=":")
    axes[1].set_xlabel("value in July 2020 (M€)"); axes[1].set_ylabel("value in June 2026 (M€)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "backtest.png")
    plt.close(fig)


def ablation_delta(base, variant, scale):
    # Écart moyen d'erreur absolue (log) par joueur, variante - modèle retenu, et son erreur
    # type, sur le critère principal (prix du départ contre vraie valeur). Les deux fichiers
    # doivent venir du MÊME code : seule la variante testée diffère.
    m = base[base.scale == scale].merge(variant[variant.scale == scale], on="player_id", suffixes=("", "_v"))
    d = np.abs(np.log(m.median_v / m.truth_v)) - np.abs(np.log(m["median"] / m.truth))
    return d.mean(), d.std() / np.sqrt(len(d))


def second_validation(best):
    # Validation 2023 -> 2026 et test de l'impact on/off (s'ils ont été calculés).
    path23 = config.PROCESSED_DIR / "backtest_2023.parquet"
    abl = config.PROCESSED_DIR / "ablation"
    if not path23.exists():
        return ""
    r23 = pd.read_parquet(path23)
    s23, _, _ = bt.report(r23)
    row = s23.loc[best]
    text = (f"\n\nA second, shorter validation starts in July 2023 and ends in June 2026 (three years, "
            f"{int(row.joueurs)} players drawn in the same way). With the same noise scale and at 2023 prices, the 80\\% interval covers "
            f"the truth in {row.couverture_80 * 100:.0f}\\% of cases and the bias is {row.biais_log:+.2f}. This second "
            f"test matters for ingredients that exist only from 2020/21, which the 2020 validation cannot see.")
    files = [abl / "backtest_with_onoff.parquet", abl / "backtest_with_onoff_2023.parquet"]
    if all(f.exists() for f in files):
        d20, se20 = ablation_delta(pd.read_parquet(config.PROCESSED_DIR / "backtest.parquet"), pd.read_parquet(files[0]), best)
        d23, se23 = ablation_delta(r23, pd.read_parquet(files[1]), best)
        text += (f" The on/off criterion is such an ingredient. Adding it to the performance index changes the mean "
                 f"absolute error of the forecast medians by {d23:+.4f} (standard error {se23:.4f}) in logarithms over "
                 f"2023--2026: no measurable gain, so it is not retained. Over 2020--2026 the change is {d20:+.4f} "
                 f"({se20:.4f}); it cannot come from the criterion itself, which does not exist before 2020/21, but from "
                 f"the context model, re-estimated with the new index on all seasons (see Section~\\ref{{sec:flaws}}).")
    return text


def age_markets():
    # Hausse du marché selon l'âge, 2019/20 -> 2025/26 : valeur médiane des 24-28 ans
    # (fin de saison) et variation médiane, d'une saison à l'autre, d'un même joueur de 25-27 ans.
    q = pd.read_parquet(config.PROCESSED_DIR / "quarterly.parquet",
                        columns=["player_id", "season", "quarter", "age", "market_value", "league_id"])
    big5 = [l["transfermarkt"] for l in config.LEAGUES.values()]
    q = q[q.league_id.isin(big5) & (q.market_value > 0) & (q.quarter == 4)]
    start, end = config.BACKTEST_ORIGIN - 1, config.CURRENT_SEASON - 1
    med = q[q.age.between(24, 28)].groupby("season").market_value.median()
    young = float(np.log(med.loc[end] / med.loc[start]))
    w = q[["player_id", "season", "age", "market_value"]]
    pairs = w.merge(w.assign(season=w.season - 1), on=["player_id", "season"], suffixes=("", "_next"))
    pairs = pairs[pairs.age.between(25, 27) & pairs.season.between(start, end - 1)]
    prime = float(np.log(pairs.market_value_next / pairs.market_value).groupby(pairs.season).median().sum())
    return young, prime


def write_results(trans_top, example, results):
    by_scale, by_band, best = bt.report(results)
    b = by_scale.loc[best]
    r = results[results.scale == best]
    # Trois comparaisons (evaluation/backtest.py) : prix de 2020 contre vraie valeur
    # (principale), contre vraie valeur dégonflée par l'indice général, et euros courants.
    ratio_true = (r.truth / r.value_2020).median()
    ratio_pred = (r["median"] / r.value_2020).median()
    ratio_pred_nom = (r.median_nominal / r.value_2020).median()
    bias_nom = np.log(r.median_nominal / r.truth_nominal).mean()
    bias_defl = np.log(r.median_deflated / r.truth_deflated).mean()
    market = r.market_change.iloc[0]
    from models import valuation
    infl_2020 = valuation.fit(bt.ORIGIN_T)["inflation"]["drift"]
    young, prime = age_markets()
    rows = "\n".join(f"{n} & {rp:.2f} & {cov * 100:.0f}\\% & {bias:+.2f} & {'--' if pit is None else f'{pit:.2f}'} \\\\"
                     for n, rp, cov, bias, pit in HISTORY)
    rows += (f"\n\\midrule\nFinal model, current euros & {ratio_pred_nom:.2f} & {r.in_80_nominal.mean() * 100:.0f}\\%"
             f" & {bias_nom:+.2f} & {r.pit_nominal.mean():.2f} \\\\")
    rows += (f"\nFinal model, 2020 prices, truth deflated by the market index & {ratio_pred:.2f} & "
             f"{r.in_80_deflated.mean() * 100:.0f}\\% & {bias_defl:+.2f} & {r.pit_deflated.mean():.2f} \\\\")
    rows += (f"\n\\textbf{{Final model, 2020 prices}} & \\textbf{{{ratio_pred:.2f}}} & \\textbf{{{b.couverture_80 * 100:.0f}\\%}}"
             f" & \\textbf{{{b.biais_log:+.2f}}} & \\textbf{{{r.pit.mean():.2f}}} \\\\")
    by_band = by_band.reindex([b for b in ["<5M", "5-20M", "20-50M", ">50M"] if b in by_band.index])
    band_rows = "\n".join(f"{band} & {int(x.joueurs)} & {x.couverture_80 * 100:.0f}\\% & {x.biais_log:+.2f} \\\\"
                          for band, x in by_band.iterrows()).replace("M ", "\\,M ").replace("<", "$<$").replace(">", "$>$")
    scale_rows = "\n".join(f"{s:.2f} & {x.couverture_50 * 100:.0f}\\% & {x.couverture_80 * 100:.0f}\\% & "
                           f"{x.couverture_90 * 100:.0f}\\% & {np.exp(x.largeur_80):.1f} \\\\" for s, x in by_scale.iterrows())
    e = example
    obs_text = ""
    if len(e["obs"]):
        q1 = e["obs"].iloc[0]
        obs_text = (f" The on-demand update downloads his {int(e['obs'].matches.sum())} league matches of 2026/27: "
                    f"he played {q1.share * 100:.0f}\\% of available minutes in August--October, then disappeared from every "
                    f"team sheet, which the model reads as an injury. His starting distribution is therefore "
                    f"{e['start']['Injured'] * 100:.0f}\\% Injured.")
    text = f"""
\\subsection{{Estimated transitions}}
Figure~\\ref{{fig:transitions}} shows two of the 84 estimated transition matrices, for attacking players aged 23 to 25. The diagonal dominates: careers are persistent from one quarter to the next. The talent tier changes the picture in the upper states. A Strong player of the top tier stays Strong with probability {trans_top[4, 4]:.2f}, whereas in the bottom tier the Strong state is fragile and players fall back to Starter or Rotation. Injured players mostly return through Rotation, as one would expect of a player regaining fitness.

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=\\textwidth]{{figures/transitions.png}}
\\caption{{Estimated quarterly transition matrices for attacking players aged 23--25.}}
\\label{{fig:transitions}}
\\end{{figure}}

The Elo ratings computed with the conservation rule of Proposition~\\ref{{prop:elo}} order the five leagues consistently over time (Figure~\\ref{{fig:elo}}).

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.92\\textwidth]{{figures/elo.png}}
\\caption{{Mean Elo rating of clubs in the five major leagues.}}
\\label{{fig:elo}}
\\end{{figure}}

\\subsection{{An example: {e['name']}}}
{e['name']} was {e['age']:.1f} years old at the start of the forecast and valued at {e['v0']:.0f} million euros.{obs_text} Figure~\\ref{{fig:example}} shows the result of $10\\,000$ simulated careers. The median real value is {e['y1']:.0f} million euros after one year, {e['y3']:.0f} after three and {e['y6']:.0f} after six, with an 80\\% interval of {e['lo']:.0f} to {e['hi']:.0f} million euros in 2032. The probability that he is worth at least 100 million euros four years from now is {e['p100'] * 100:.0f}\\%. The right panel shows the mechanism: most careers leave the Injured state within two quarters, and the share of Strong and Elite quarters slowly declines as he moves past 28.

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=\\textwidth]{{figures/example.png}}
\\caption{{{e['name']}: distribution of simulated market values (left; bands contain 50\\%, 80\\% and 90\\% of careers) and of latent states (right).}}
\\label{{fig:example}}
\\end{{figure}}

\\subsection{{Out-of-sample validation}}
A model of this kind can always be made to look plausible on a few famous names. The only honest test is to forecast the past. We therefore placed ourselves in July 2020: every component (the hidden Markov model, the valuation equations, the event probabilities, the team-contribution model and the drift calibration) was re-estimated with data available at that date only. We then simulated six years ahead for {len(r)} players of the top five leagues, aged 18 to 32 and worth at least one million euros, stratified by value, and compared the forecasts with their true Transfermarkt values in June 2026.

Three numbers summarise a probabilistic forecast. The \\emph{{coverage}} of the 80\\% interval should be close to 80\\%. The \\emph{{bias}} is the mean of $\\log(\\text{{forecast median}}/\\text{{truth}})$, ideally zero. The \\emph{{PIT}} is the fraction of simulated careers below the true value; if the forecasts are well calibrated it is uniformly distributed, with mean $0.5$. Table~\\ref{{tab:backtest}} shows how the model improved as its flaws were found and corrected. In reality the median player kept only {ratio_true:.2f} of his 2020 value six years later; the naive simulation predicted 0.81. The earlier versions were scored in current euros, so their forecasts also contained a forecast of market inflation, and that forecast turned out to dominate the error. With data up to 2020 the season effects grow by {(np.exp(infl_2020) - 1) * 100:.0f}\\% a year (the 2015--2019 boom); the index estimated on all the data grows by {(np.exp(market) - 1) * 100:.0f}\\% in total between 2020 and 2026. Moreover the growth was uneven across ages: the median value of players aged 24--28 rose by {(np.exp(young) - 1) * 100:.0f}\\% over the period, whereas the value of a given player aged 25--27 changed by {(np.exp(prime) - 1) * 100:+.0f}\\% in total over the same seasons (sum of median yearly changes). The tested players age by six years during the test, so the general index overstates the inflation of their own market. Market inflation is a forecast of its own, which the career model cannot be expected to make. The final model is therefore judged at 2020 prices against the true 2026 value, and Table~\\ref{{tab:backtest}} also reports the comparison with the true value deflated by the general index and the forecast in current euros.

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lcccc}}
\\toprule
Model & value ratio 2026/2020 & 80\\% coverage & bias (log) & mean PIT \\\\
\\midrule
{rows}
\\midrule
True outcome & {ratio_true:.2f} & 80\\% & 0 & 0.50 \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Out-of-sample forecasts from July 2020 to June 2026 ({len(r)} players). Earlier versions and the first final line are scored in current euros, the last three lines differ only in how market inflation is treated (see text).}}
\\label{{tab:backtest}}
\\end{{table}}

The amount of market noise added in the simulation is the one free parameter of the forecast intervals. Table~\\ref{{tab:scale}} shows the coverage for several values; we retain the value {best:.2f}, whose 80\\% interval covers the truth in {b.couverture_80 * 100:.0f}\\% of cases. The intervals are wide (the upper bound of the 80\\% interval is on average {np.exp(b.largeur_80):.0f} times the lower bound), which is the honest price of a six-year horizon: Raheem Sterling went from 128 to 3 million euros and Sadio Man\\'e from 120 to 6 million between the two dates.

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{ccccc}}
\\toprule
noise scale & 50\\% coverage & 80\\% coverage & 90\\% coverage & width of 80\\% interval (ratio) \\\\
\\midrule
{scale_rows}
\\bottomrule
\\end{{tabular}}
\\caption{{Coverage of the forecast intervals as a function of the noise scale.}}
\\label{{tab:scale}}
\\end{{table}}

{second_validation(best)}

Coverage is similar across value bands (Table~\\ref{{tab:bands}}), but the bias is not: it ranges from {by_band.biais_log.min():+.2f} to {by_band.biais_log.max():+.2f}, and the forecasts remain too high for players already worth more than 20 million euros in 2020. The decline of stars is the part of a career that the model captures least well.

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lccc}}
\\toprule
value in 2020 & players & 80\\% coverage & bias (log) \\\\
\\midrule
{band_rows}
\\bottomrule
\\end{{tabular}}
\\caption{{Validation by value band.}}
\\label{{tab:bands}}
\\end{{table}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=\\textwidth]{{figures/backtest.png}}
\\caption{{Out-of-sample validation. Left: the rank of the true value within the simulated distribution (a flat histogram means calibrated forecasts). Right: forecast medians and true values against the 2020 value.}}
\\label{{fig:backtest}}
\\end{{figure}}
"""
    (OUT / "results.tex").write_text(text, encoding="utf-8")


def write_numbers(ctx):
    # Tous les chiffres cités dans le texte du papier (hors section Results), sous
    # forme de macros LaTeX : \num... Le papier ne contient AUCUN chiffre de modèle
    # écrit à la main, il ne peut donc pas diverger du code.
    import json
    from collectors import tournaments as tn
    from features import quarterly as qt
    from models import performance as pf
    from simulation import calibration

    def fmt_int(x):
        return f"{int(round(x)):,}".replace(",", "\\,")

    q = pd.read_parquet(qt.OUT_PATH, columns=["player_id", "t"])
    perf = pd.read_parquet(pf.OUT_PATH, columns=["perf", "perf_source", "t"])
    fm_cols = pd.read_parquet(qt.OUT_PATH, columns=["xg_minutes", "fm_minutes"])
    context = json.loads(pf.CONTEXT_PATH.read_text())
    diag = ctx.val["diagnostics"]
    params, ev = ctx.hmm["params"], ctx.ev
    # Attaquants de 23-25 ans (comme la figure des transitions) : rester "Fort" d'un trimestre à l'autre
    stay_top = params["trans"][hmm.class_index(2, 3, 2)][hmm.FORT, hmm.FORT]
    stay_bottom = params["trans"][hmm.class_index(2, 3, 0)][hmm.FORT, hmm.FORT]
    age_bin = lambda a: int(np.digitize(a, hmm.AGE_BINS[1:-1]))
    calib = calibration.load(full=True)
    tour = tn.load()
    played = tour[tour.minutes > 0]
    reliability = pd.read_parquet(qt.ABSENCE_RELIABILITY_PATH)
    rated = perf[perf.perf.notna()]
    fm_rows = fm_cols[fm_cols.fm_minutes > 0]
    numbers = {
        "Players": fmt_int(q.player_id.nunique()), "PlayerQuarters": fmt_int(len(q)),
        "BetaOwn": f"{context['own']:+.3f}", "BetaOpp": f"{context['opp']:+.3f}",
        "StayStrongTop": f"{stay_top:.2f}", "StayStrongBottom": f"{stay_bottom:.2f}",
        "LeaveTopOld": f"{ev['destination'][age_bin(32), 1, 2] * 100:.0f}",
        "LeaveTopYoungStar": f"{ev['destination'][age_bin(23), 2, 2] * 100:.0f}",
        "RsqLevel": f"{diag['r2_level'] * 100:.0f}", "RsqDyn": f"{diag['r2_dynamic'] * 100:.0f}",
        "Bdyn": f"{diag['b']:.2f}", "SigmaA": f"{diag['annual_sd']:.2f}",
        "NBig": fmt_int(diag["n_big"]), "BigBias": f"{(np.exp(diag['big_bias']) - 1) * 100:.0f}",
        "Inflation": f"{(np.exp(ctx.val['inflation']['drift']) - 1) * 100:.1f}",
        "CalibCases": fmt_int(len(calib["gaps"])),
        "CalibGap": f"{-calib['gaps'].gap.median() * 100:.0f}",
        "GapBandMin": f"{calib['gaps'].groupby('band').gap.median().min():+.2f}",
        "GapBandMax": f"{calib['gaps'].groupby('band').gap.median().max():+.2f}",
        "TournamentMatches": fmt_int(played.match_id.nunique()),
        "TournamentLinked": f"{played.tm_player_id.notna().mean() * 100:.0f}",
        "AbsenceKept": str(int((reliability.consistent_share >= qt.ABSENCE_RELIABILITY).sum())),
        "AbsenceTotal": str(len(reliability)),
        "ShareGoalsAssists": f"{(rated.perf_source == 'transfermarkt').mean() * 100:.0f}",
        "ShareXg": f"{(fm_rows.xg_minutes > 0).mean() * 100:.0f}",
    }
    # Validation 2020 -> 2026 (niveau de bruit retenu)
    results = pd.read_parquet(config.PROCESSED_DIR / "backtest.parquet")
    by_scale, by_band, best = bt.report(results)
    final = results[results.scale == best]
    numbers.update({"FinalCoverage": f"{by_scale.loc[best, 'couverture_80'] * 100:.0f}",
                    "FinalBias": f"{by_scale.loc[best, 'biais_log']:+.2f}", "FinalPit": f"{final.pit.mean():.2f}",
                    "BandBiasMin": f"{by_band.biais_log.min():+.2f}", "BandBiasMax": f"{by_band.biais_log.max():+.2f}"})
    lines = [f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in numbers.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return numbers


def main():
    FIG.mkdir(exist_ok=True)
    ctx = mc.Context()
    write_numbers(ctx)
    trans_top = fig_transitions(ctx.hmm["params"])
    fig_elo()
    example = player_example(ctx, "Bukayo Saka")
    results = pd.read_parquet(config.PROCESSED_DIR / "backtest.parquet")
    _, _, best = bt.report(results)
    fig_backtest(results, best)
    write_results(trans_top, example, results)
    print("✓ figures et results.tex écrits")


if __name__ == "__main__":
    main()
