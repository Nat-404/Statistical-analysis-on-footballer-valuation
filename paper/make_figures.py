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


def write_results(trans_top, example, results):
    by_scale, by_band, best = bt.report(results)
    b = by_scale.loc[best]
    r = results[results.scale == best]
    ratio_true = (r.truth / r.value_2020).median()
    ratio_pred = (r["median"] / r.value_2020).median()
    rows = "\n".join(f"{n} & {rp:.2f} & {cov * 100:.0f}\\% & {bias:+.2f} & {'--' if pit is None else f'{pit:.2f}'} \\\\"
                     for n, rp, cov, bias, pit in HISTORY)
    rows += (f"\n\\textbf{{+ FotMob xG, defence and injuries (final)}} & \\textbf{{{ratio_pred:.2f}}} & \\textbf{{{b.couverture_80 * 100:.0f}\\%}}"
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

Three numbers summarise a probabilistic forecast. The \\emph{{coverage}} of the 80\\% interval should be close to 80\\%. The \\emph{{bias}} is the mean of $\\log(\\text{{forecast median}}/\\text{{truth}})$, ideally zero. The \\emph{{PIT}} is the fraction of simulated careers below the true value; if the forecasts are well calibrated it is uniformly distributed, with mean $0.5$. Table~\\ref{{tab:backtest}} shows how the model improved as its flaws were found and corrected. In reality the median player kept only {ratio_true:.2f} of his 2020 value six years later; the naive simulation predicted 0.81.

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
\\caption{{Out-of-sample forecasts from July 2020 to June 2026 ({len(r)} players).}}
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

Coverage is similar across value bands (Table~\\ref{{tab:bands}}), but the bias changes sign: the model slightly overestimates the future value of players already worth more than 20 million euros and underestimates that of cheaper players. The overall bias is small only because these two errors largely cancel.

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


def main():
    FIG.mkdir(exist_ok=True)
    ctx = mc.Context()
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
