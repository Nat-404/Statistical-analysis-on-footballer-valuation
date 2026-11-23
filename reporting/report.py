# =============================================================================
# reporting/report.py — Étape 14 : le rapport interactif d'un joueur.
#
# Produit une page HTML (outputs/<joueur>.html) avec les graphiques :
#   A. Performance individuelle : éventail de l'indice, probabilités d'état, radar
#   B. Performance collective   : points apportés par saison, temps de jeu
#   C. Valeur marchande         : éventail avec/sans inflation, distribution à
#                                 6 ans, probabilité de dépasser des seuils,
#                                 trimestre où la valeur est au plus haut
#   D. Revenus générés          : par source, scénario médian vs "devient une star"
#   E. Comprendre le modèle     : sensibilité (tornade), carrières simulées
# Un "éventail" montre la médiane (ligne) et les bandes 50 %, 80 % et 90 %.
# =============================================================================

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from simulation import monte_carlo as mc

# Couleurs : une teinte principale (bleu) déclinée pour les bandes, + des couleurs d'états
BLUE = "#2a5bd7"
BAND_ALPHA = {50: 0.35, 80: 0.20, 90: 0.10}
STATE_COLORS = ["#c0392b", "#95a5a6", "#f1c40f", "#3498db", "#8e44ad", "#16a085"]
SOURCE_COLORS = ["#2a5bd7", "#e67e22", "#16a085", "#8e44ad"]


def quarter_labels(t0, steps):
    # t = saison x 4 + trimestre - 1 -> "T1 26/27"
    labels = []
    for h in range(1, steps + 1):
        t = t0 + h
        s, q = t // 4, t % 4 + 1
        labels.append(f"T{q} {str(s)[2:]}/{str(s + 1)[2:]}")
    return labels


def season_labels(t0, n):
    first = (t0 + 1) // 4
    return [f"{str(s)[2:]}/{str(s + 1)[2:]}" for s in range(first, first + n)]


def fan(fig, x, values, name, color=BLUE, row=None, col=None, unit=""):
    # Ajoute un éventail (bandes de confiance + médiane) à une figure.
    summary = mc.summarize(values)
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    kw = {"row": row, "col": col} if row else {}
    for b in sorted(config.CONFIDENCE_BANDS, reverse=True):
        pct = int(b * 100)
        fig.add_trace(go.Scatter(x=x, y=summary[f"high_{pct}"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), **kw)
        fig.add_trace(go.Scatter(x=x, y=summary[f"low_{pct}"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=f"rgba{rgb + (BAND_ALPHA[pct],)}",
                                 name=f"{name} – {pct} %", hovertemplate=f"{pct} % : %{{y:.1f}}{unit}<extra></extra>"), **kw)
    fig.add_trace(go.Scatter(x=x, y=summary["median"], mode="lines", line=dict(color=color, width=2.5),
                             name=f"{name} – médiane", hovertemplate=f"médiane : %{{y:.1f}}{unit}<extra></extra>"), **kw)


def _layout(fig, title, ytitle, height=420):
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)), height=height,
                      margin=dict(l=60, r=20, t=60, b=50), template="plotly_white",
                      legend=dict(orientation="h", y=-0.18), hovermode="x unified",
                      font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12))
    fig.update_yaxes(title_text=ytitle)
    return fig


# -----------------------------------------------------------------------------
# Graphiques
# -----------------------------------------------------------------------------
def chart_performance(sim, x):
    perf = pd.DataFrame(sim["perf"]).T.rolling(4, min_periods=1).mean().T.to_numpy()
    fig = go.Figure()
    fan(fig, x, perf, "Indice de performance (moyenne 4 trim.)")
    fig.add_hline(y=0, line_dash="dot", line_color="#888", annotation_text="joueur moyen du top 5")
    return _layout(fig, "A1 · Performance individuelle (indice ajusté du contexte)", "écarts-types")


def chart_states(sim, x):
    st = sim["state"].astype(int)
    fig = go.Figure()
    for k, name in enumerate(config.STATES):
        fig.add_trace(go.Scatter(x=x, y=(st == k).mean(axis=0) * 100, stackgroup="one", name=name,
                                 line=dict(width=0.5, color=STATE_COLORS[k]),
                                 hovertemplate=f"{name} : %{{y:.0f}} %<extra></extra>"))
    return _layout(fig, "A2 · Probabilité de chaque état, trimestre par trimestre", "% des carrières simulées")


def chart_radar(sim):
    # 5 dimensions ramenées sur 0-100 : aujourd'hui (1er trimestre), dans 3 ans, dans 6 ans
    def snapshot(h):
        st = sim["state"][:, h]
        return [
            np.clip(50 + 25 * np.median(sim["perf"][:, max(0, h - 3):h + 1]), 0, 100),
            np.median(sim["share"][:, max(0, h - 3):h + 1]) * 100,
            np.clip(50 + 10 * np.median(sim["points"][:, max(0, h - 3):h + 1].sum(axis=1)), 0, 100),
            np.clip(np.log10(np.median(sim["value_real"][:, h]) / 1e5) / np.log10(3000) * 100, 0, 100),
            np.mean(st >= 4) * 100,
        ]
    axes = ["Performance", "Temps de jeu", "Apport collectif", "Valeur (échelle log)", "Prob. Fort/Élite"]
    fig = go.Figure()
    for h, label, color in [(0, "Aujourd'hui", "#95a5a6"), (11, "Dans 3 ans", "#3498db"), (23, "Dans 6 ans", BLUE)]:
        vals = snapshot(min(h, sim["state"].shape[1] - 1))
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], name=label,
                                      line=dict(color=color), fill="toself", opacity=0.45))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)))
    return _layout(fig, "A3 · Profil aujourd'hui, dans 3 ans et dans 6 ans", "", height=460)


def chart_collective(sim, seasons):
    n_seasons = len(seasons)
    pts = sim["points"][:, :n_seasons * 4].reshape(sim["points"].shape[0], n_seasons, 4).sum(axis=2)
    share = sim["share"][:, :n_seasons * 4].reshape(sim["share"].shape[0], n_seasons, 4).mean(axis=2) * 100
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Points apportés à l'équipe par saison",
                                                        "Part du temps de jeu par saison (%)"))
    fan(fig, seasons, pts, "Points", row=1, col=1, unit=" pts")
    fan(fig, seasons, share, "Temps de jeu", color="#16a085", row=1, col=2, unit=" %")
    fig.add_hline(y=0, line_dash="dot", line_color="#888", row=1, col=1)
    return _layout(fig, "B · Performance collective (par rapport à un joueur moyen du top 5)", "")


def chart_value(sim, x):
    fig = go.Figure()
    fan(fig, x, sim["value_nominal"] / 1e6, "Avec inflation", color="#e67e22", unit=" M€")
    fan(fig, x, sim["value_real"] / 1e6, "Sans inflation (€ 2026)", color=BLUE, unit=" M€")
    fig.add_hline(y=sim["meta"]["value0"] / 1e6, line_dash="dot", line_color="#888",
                  annotation_text="valeur actuelle")
    return _layout(fig, "C1 · Valeur marchande, avec et sans inflation du marché", "millions d'euros", height=480)


def chart_distribution(sim):
    final = sim["value_real"][:, -1] / 1e6
    width = max(np.quantile(final, 0.99) / 40, 0.1)            # ~40 barres sur la zone affichée
    fig = go.Figure(go.Histogram(x=final, xbins=dict(start=0, size=width), marker_color=BLUE, opacity=0.8,
                                 hovertemplate="%{x:.0f} M€ : %{y} carrières<extra></extra>"))
    for q, label in [(0.1, "10 %"), (0.5, "médiane"), (0.9, "90 %")]:
        fig.add_vline(x=np.quantile(final, q), line_dash="dash", line_color="#333",
                      annotation_text=f"{label} : {np.quantile(final, q):.0f} M€")
    # On coupe l'axe au 99e centile : quelques carrières extrêmes écraseraient le graphique
    fig.update_xaxes(title_text="valeur dans 6 ans (M€, sans inflation)", range=[0, np.quantile(final, 0.99)])
    return _layout(fig, "C2 · Distribution de la valeur dans 6 ans", "nombre de carrières")


def chart_thresholds(sim, x):
    v0 = sim["meta"]["value0"]
    candidates = [5e6, 10e6, 20e6, 50e6, 100e6, 150e6, 200e6]
    thresholds = [c for c in candidates if 0.25 * v0 <= c <= 3 * v0][:4] or [v0]
    fig = go.Figure()
    for k, th in enumerate(thresholds):
        fig.add_trace(go.Scatter(x=x, y=(sim["value_real"] >= th).mean(axis=0) * 100, mode="lines",
                                 name=f"> {th / 1e6:.0f} M€", line=dict(width=2.5, color=SOURCE_COLORS[k % 4]),
                                 hovertemplate=f"> {th / 1e6:.0f} M€ : %{{y:.0f}} %<extra></extra>"))
    return _layout(fig, "C3 · Probabilité de dépasser une valeur donnée", "% des carrières")


def chart_peak(sim, x):
    peak = sim["value_real"].argmax(axis=1)
    counts = np.bincount(peak, minlength=len(x)) / len(peak) * 100
    fig = go.Figure(go.Bar(x=x, y=counts, marker_color=BLUE,
                           hovertemplate="%{x} : %{y:.1f} %<extra></extra>"))
    return _layout(fig, "C4 · À quel moment la valeur atteint-elle son maximum ? (quand vendre)", "% des carrières")


def chart_revenue(sim, seasons):
    rev = sim["revenue"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Scénario médian (M€ par saison)",
                                                        "Scénario « devient une star » (M€ par saison)"))
    # "Star" : carrières où il est Fort ou Élite au moins la moitié des trimestres
    star = (sim["state"] >= 4).mean(axis=1) >= 0.5
    for k, (source, values) in enumerate(rev.items()):
        fig.add_trace(go.Bar(x=seasons, y=np.median(values, axis=0) / 1e6, name=source,
                             marker_color=SOURCE_COLORS[k], legendgroup=source), row=1, col=1)
        star_values = values[star] if star.sum() >= 50 else np.quantile(values, 0.9, axis=0, keepdims=True)
        fig.add_trace(go.Bar(x=seasons, y=np.median(star_values, axis=0) / 1e6, name=source,
                             marker_color=SOURCE_COLORS[k], legendgroup=source, showlegend=False), row=1, col=2)
    fig.update_layout(barmode="stack")
    title = (f"D · Revenus générés pour le club (scénarios — hypothèses, voir notes) · "
             f"probabilité du scénario star : {star.mean() * 100:.0f} %")
    return _layout(fig, title, "millions d'euros")


def chart_tornado(sensitivity, base):
    s = sensitivity.sort_values("range")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=s.factor, x=s.low - base, base=base, orientation="h", name="scénario bas",
                         marker_color="#c0392b", hovertemplate="%{x:.0f} M€<extra></extra>"))
    fig.add_trace(go.Bar(y=s.factor, x=s.high - base, base=base, orientation="h", name="scénario haut",
                         marker_color="#16a085", hovertemplate="%{x:.0f} M€<extra></extra>"))
    fig.add_vline(x=base, line_color="#333")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="valeur médiane dans 6 ans (M€)")
    return _layout(fig, "E1 · Sensibilité : quels facteurs font le plus bouger la valeur ?", "", height=380)


def chart_paths(sim, x, k=25, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.choice(sim["value_real"].shape[0], k, replace=False)
    fig = go.Figure()
    for i in idx:
        fig.add_trace(go.Scatter(x=x, y=sim["value_real"][i] / 1e6, mode="lines", line=dict(width=1),
                                 opacity=0.6, showlegend=False, hovertemplate="%{y:.0f} M€<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=np.median(sim["value_real"], axis=0) / 1e6, mode="lines",
                             line=dict(color="#111", width=3), name="médiane"))
    return _layout(fig, f"E2 · {k} carrières simulées parmi {sim['meta']['n']:,}", "M€ (sans inflation)")


# -----------------------------------------------------------------------------
# Sensibilité (tornade)
# -----------------------------------------------------------------------------
def sensitivity(player_id, ctx, n=2000, live=None):
    # On relance la simulation en changeant UN facteur à la fois, et on regarde
    # la valeur médiane dans 6 ans dans le scénario bas et haut.
    def median_final(**kw):
        return float(np.median(mc.simulate(player_id, ctx, n=n, live=live, **kw)["value_real"][:, -1]) / 1e6)
    base = median_final()
    rows = [
        ("Résilience (0,8 / 1,2)", median_final(resilience_override=0.8), median_final(resilience_override=1.2)),
        ("Bruit de marché (x0,5 / x1,5)", median_final(resid_scale=mc.RESID_SCALE * 1.5),
         median_final(resid_scale=mc.RESID_SCALE * 0.5)),
        ("Hasard (2 autres tirages)", median_final(seed=7), median_final(seed=11)),
    ]
    df = pd.DataFrame(rows, columns=["factor", "a", "b"])
    df["low"], df["high"] = df[["a", "b"]].min(axis=1), df[["a", "b"]].max(axis=1)
    df["range"] = df.high - df.low
    return df, base


# -----------------------------------------------------------------------------
# Page HTML
# -----------------------------------------------------------------------------
CSS = """
:root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5f6b7a; --accent:#2a5bd7; --line:#e3e7ee; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#12151b; --card:#1b2029; --ink:#e8ecf2; --muted:#9aa6b6; --accent:#7aa2ff; --line:#2a313d; } }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,Helvetica,Arial,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:28px; margin:0 0 4px; } h2 { font-size:19px; margin:36px 0 12px; }
.sub { color:var(--muted); margin:0 0 20px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.kpi b { display:block; font-size:22px; margin-top:4px; } .kpi span { color:var(--muted); font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px; margin-bottom:14px; }
.note { color:var(--muted); font-size:13px; line-height:1.5; }
table.live { width:100%; border-collapse:collapse; font-size:14px; }
table.live th, table.live td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); }
table.live th:first-child, table.live td:first-child { text-align:left; }
.card { overflow-x:auto; }
"""


def live_block(obs):
    # Encadré "forme de la saison en cours" (données FotMob récupérées à la demande)
    if obs is None or len(obs) == 0:
        return '<p class="note">Aucune donnée 2026/27 trouvée sur FotMob pour ce joueur : départ en juillet 2026.</p>'
    names = {1: "T1 (août-oct.)", 2: "T2 (nov.-janv.)", 3: "T3 (févr.-avr.)", 4: "T4 (mai-juil.)"}
    rows = "".join(
        f"<tr><td>{names[r.quarter]}</td><td>{r.matches}</td><td>{r.minutes:.0f}</td><td>{r.share * 100:.0f} %</td>"
        f"<td>{r.goals:.0f}</td><td>{r.xg:.1f}</td><td>{r.missed_injury}</td>"
        f"<td>{'' if np.isnan(r.perf) else f'{r.perf:+.2f}'}</td></tr>" for r in obs.itertuples())
    return ('<h2>Forme 2026/27 (FotMob, récupérée à la demande)</h2><div class="card"><table class="live">'
            "<tr><th>Trimestre</th><th>Matchs</th><th>Minutes</th><th>Temps de jeu</th><th>Buts</th>"
            "<th>xG</th><th>Matchs manqués (blessure)</th><th>Indice perf. (saison)</th></tr>"
            f"{rows}</table><p class='note'>Ces trimestres sont observés, pas simulés : ils mettent à jour "
            "l'état de départ du joueur dans la chaîne de Markov avant la simulation.</p></div>")


def build_report(player_id, ctx=None, path=None, live=None):
    ctx = ctx or mc.Context()
    sim = mc.simulate(player_id, ctx, live=live)
    meta = sim["meta"]
    x = quarter_labels(meta["t0"], meta["steps"])
    seasons = season_labels(meta["t0"], meta["steps"] // 4)
    sens, base = sensitivity(player_id, ctx, live=live)
    final_real = sim["value_real"][:, -1] / 1e6
    final_nom = sim["value_nominal"][:, -1] / 1e6
    p_star = ((sim["state"] >= 4).mean(axis=1) >= 0.5).mean() * 100

    figures = [chart_performance(sim, x), chart_states(sim, x), chart_radar(sim),
               chart_collective(sim, seasons),
               chart_value(sim, x), chart_distribution(sim), chart_thresholds(sim, x), chart_peak(sim, x),
               chart_revenue(sim, seasons),
               chart_tornado(sens, base), chart_paths(sim, x)]
    sections = {0: "A. Performance individuelle", 3: "B. Performance collective", 4: "C. Valeur marchande",
                8: "D. Revenus générés", 9: "E. Comprendre le modèle"}
    parts = []
    for i, fig in enumerate(figures):
        if i in sections:
            parts.append(f"<h2>{sections[i]}</h2>")
        parts.append(f'<div class="card">{fig.to_html(full_html=False, include_plotlyjs=(i == 0) and "cdn")}</div>')

    name = html.escape(str(meta["name"]))
    kpis = [
        ("Valeur actuelle", f"{meta['value0'] / 1e6:.0f} M€"),
        ("Dans 6 ans (médiane, € 2026)", f"{np.median(final_real):.0f} M€"),
        ("Fourchette 80 %", f"{np.quantile(final_real, .1):.0f} – {np.quantile(final_real, .9):.0f} M€"),
        ("Dans 6 ans avec inflation", f"{np.median(final_nom):.0f} M€"),
        ("Probabilité « star »", f"{p_star:.0f} %"),
        ("Résilience", f"{meta['resilience']:.2f}"),
    ]
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis)
    notes = """
    <h2>Notes de lecture</h2>
    <p class="note">Chaque graphique en éventail montre la médiane (ligne) et les bandes contenant 50 %, 80 % et
    90 % des 10 000 carrières simulées. « Sans inflation » : prix du marché de 2026 ; « avec inflation » :
    le marché des transferts continue d'évoluer comme sur 2014-2026 (trajectoire aléatoire).
    Les revenus (D) sont des <b>scénarios</b> construits à partir de chiffres publics (UEFA 2024-27,
    Premier League 2024/25, Deloitte Football Money League 2026) et d'hypothèses explicites
    (maillots vendus, part du sponsoring attribuable au joueur) : ils donnent un ordre de grandeur, pas une prévision.
    La prolongation de contrat repose aussi sur des hypothèses (aucune source ne donne l'historique des contrats).</p>
    """
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>{name} · projection</title>
    <style>{CSS}</style></head><body><main>
    <h1>{name}</h1>
    <p class="sub">Projection sur {config.HORIZON_YEARS} ans ({meta['steps']} trimestres) ·
    {meta['n']:,} carrières simulées · âge au départ {meta['age0']:.1f} ans · départ : {x[0]}</p>
    <div class="kpis">{kpi_html}</div>
    {live_block(meta.get("live_obs"))}
    {''.join(parts)}
    {notes}
    </main></body></html>"""
    path = path or config.OUTPUTS_DIR / f"{str(meta['name']).replace(' ', '_')}.html"
    path.write_text(page, encoding="utf-8")
    return path, sim
