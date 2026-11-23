# =============================================================================
# reporting/fiche.py — La fiche de résultats : 2 pages, style article académique.
#
#   1. Présentation du joueur + Tableau 1 (valeur à T0, statistiques 2025/26)
#   2. Tableau 2 : hypothèses de la simulation
#   3. Résultats : Figure 1 (les 10 000 trajectoires de valeur), Tableau 3,
#      Figure 2 (les 10 000 trajectoires de temps de jeu), Tableau 4 — chacun
#      suivi d'une ou deux phrases d'analyse générées à partir des chiffres.
#
# T0 = date de la simulation. Toutes les carrières partent de la valeur
# marchande actuelle du joueur (dernière valeur Transfermarkt connue).
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
GREY, DARK, NAVY, ACCENT = "#9aa3ad", "#111111", "#1f3a93", "#b03a2e"
POSITIONS = {"Goalkeeper": "gardien", "Centre-Back": "défenseur central", "Left-Back": "latéral gauche",
             "Right-Back": "latéral droit", "Defensive Midfield": "milieu défensif", "Central Midfield": "milieu central",
             "Attacking Midfield": "milieu offensif", "Left Midfield": "milieu gauche", "Right Midfield": "milieu droit",
             "Left Winger": "ailier gauche", "Right Winger": "ailier droit", "Centre-Forward": "avant-centre",
             "Second Striker": "second attaquant"}
COUNTRIES = {"England": "anglais", "France": "français", "Spain": "espagnol", "Germany": "allemand",
             "Italy": "italien", "Brazil": "brésilien", "Argentina": "argentin", "Portugal": "portugais",
             "Netherlands": "néerlandais", "Belgium": "belge", "Morocco": "marocain", "Senegal": "sénégalais",
             "Norway": "norvégien", "Croatia": "croate", "Denmark": "danois", "Scotland": "écossais"}
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]
TIERS = ["60 % inférieurs", "top 10–40 %", "top 10 %"]


# --- Petits outils de mise en forme -------------------------------------------
def eur(x):
    if x >= 9.95e6:
        return f"{x / 1e6:.0f} M€"
    if x >= 1e6:
        return f"{x / 1e6:.1f} M€".replace(".", ",")
    return f"{x / 1e3:.0f} k€"


def pct(p):
    return f"{p * 100:.0f} %"


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def date_fr(d):
    d = pd.Timestamp(d)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


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


def spaghetti(ax, years, paths, label_unit, scale=1.0):
    # Les 10 000 trajectoires en gris très clair, puis moyenne et médiane en foncé.
    paths = paths * scale
    segments = np.stack([np.broadcast_to(years, paths.shape), paths], axis=-1)
    lc = LineCollection(segments, colors=GREY, linewidths=0.25, alpha=0.035, rasterized=True)
    ax.add_collection(lc)
    ax.plot(years, paths.mean(axis=0), color=NAVY, lw=1.8, label="moyenne")
    ax.plot(years, np.median(paths, axis=0), color=DARK, lw=1.8, ls="--", label="médiane")
    ax.set_xlim(0, years[-1])
    ax.set_ylim(0, np.quantile(paths, 0.99))
    ax.set_xlabel("années après T0", fontsize=9)
    ax.set_ylabel(label_unit, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.tick_params(labelsize=8.5)


def reliability():
    # Résultats de la validation sur le passé (2020 -> 2026), s'ils existent
    path = config.PROCESSED_DIR / "backtest.parquet"
    if not path.exists():
        return None
    from evaluation import backtest as bt
    from simulation import monte_carlo as mc
    by_scale, _, _ = bt.report(pd.read_parquet(path))
    row = by_scale.loc[mc.RESID_SCALE] if mc.RESID_SCALE in by_scale.index else by_scale.iloc[-1]
    return {"coverage": row.couverture_80, "bias": row.biais_log, "players": int(row.joueurs)}


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


def build_fiche(player_id, ctx, sim, live=None, path=None, sim_date=None):
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
    youth = tm.query("SELECT from_club_name FROM transfers WHERE player_id = ? AND regexp_matches(from_club_name, 'Yth|Youth|U1[0-9]|U2[0-3]') "
                     "ORDER BY transfer_date LIMIT 1", [player_id])
    value_hist = tm.player_valuations(player_id)

    # Toutes les trajectoires partent de V0 à T0 : on ajoute T0 en tête
    steps = sim["value_real"].shape[1]
    years = np.arange(steps + 1) / 4
    values = np.column_stack([np.full(len(sim["value_real"]), v0), sim["value_real"]])
    nominal = np.column_stack([np.full(len(sim["value_real"]), v0), sim["value_nominal"]])
    # Temps de jeu en moyenne glissante sur 4 trimestres, en incluant les 3 derniers trimestres RÉELS
    # (sinon le premier point simulé ne reposerait que sur un trimestre et créerait un pic artificiel)
    past = rows.minutes_share.fillna(0).tail(3).to_numpy()
    if meta.get("live_obs") is not None and len(meta["live_obs"]):
        past = np.concatenate([past, meta["live_obs"].share.to_numpy()])[-3:]
    padded = np.column_stack([np.tile(past, (len(values), 1)), sim["share"]])
    rolled = pd.DataFrame(padded).T.rolling(4, min_periods=1).mean().T.to_numpy()
    share = rolled[:, len(past) - 1:]                    # colonne 0 = T0 (réel), puis les 24 trimestres simulés

    path = path or config.OUTPUTS_DIR / f"{profile['name'].replace(' ', '_')}_projection.pdf"
    with PdfPages(path) as pdf:
        _page_one(pdf, profile, stats, youth, value_hist, last, meta, v0, age0, sim_date, ctx, sim, years, values)
        _page_two(pdf, profile, sim, years, values, nominal, share, v0, ctx)
    return path


def _page_one(pdf, profile, stats, youth, value_hist, last, meta, v0, age0, sim_date, ctx, sim, years, values):
    pg = Page(pdf, 1)
    f = pg.fig
    f.text(0.5, 0.955, f"Projection de la valeur marchande de {profile['name']}", ha="center", fontsize=15, weight="bold")
    f.text(0.5, 0.93, f"Chaîne de Markov cachée et simulation Monte Carlo  ·  T0 = {date_fr(sim_date)}  ·  "
           f"horizon {config.HORIZON_YEARS} ans  ·  {meta['n']:,} carrières simulées".replace(",", " "),
           ha="center", fontsize=9.5, style="italic")
    pg.y = 0.91
    pg.rule(0.6)
    pg.y -= 0.012

    # 1. Présentation
    pg.heading("1.  Présentation du joueur")
    nat = COUNTRIES.get(profile["country_of_citizenship"], profile["country_of_citizenship"])
    pos = POSITIONS.get(profile["sub_position"], profile["sub_position"])
    first_val = value_hist.iloc[0] if len(value_hist) else None
    contract = pd.Timestamp(profile["contract_expiration_date"]) if pd.notna(profile["contract_expiration_date"]) else None
    s = (f"{profile['name']} est un {pos} {nat} de {age0:.0f} ans ({'droitier' if profile['foot'] == 'right' else 'gaucher' if profile['foot'] == 'left' else 'ambidextre'}, "
         f"{profile['height_in_cm']:.0f} cm), sous contrat avec {profile['current_club_name']}"
         + (f" jusqu'en juin {contract.year}" if contract is not None else "") + ". ")
    if len(youth):
        s += f"Formé à {youth.from_club_name.iloc[0].replace(' Yth.', '').replace(' Youth', '')}, "
    s += (f"il compte {int(profile['international_caps'] or 0)} sélections, toutes catégories confondues. "
          if profile["international_caps"] else "")
    if first_val is not None:
        s += (f"Sa valeur marchande est passée de {eur(first_val.market_value_in_eur)} ({date_fr(first_val.date)}) "
              f"à {eur(v0)} lors de la dernière estimation Transfermarkt ({date_fr(last.value_date)}) : "
              f"c'est la valeur de départ V0 de toutes les simulations.")
    pg.text(s)
    pg.y -= 0.006

    # Tableau 1
    st = stats
    fm = "fm_xg" in st
    left = [("Valeur marchande à T0 (V0)", eur(v0)), ("Valeur record", eur(profile["highest_market_value_in_eur"])),
            ("Matchs (titularisations)", f"{st['matches']} ({st['starts']})"),
            ("Minutes jouées", f"{st['minutes']:.0f}")]
    right = [("Part du temps de jeu en championnat", pct(min(st["share"], 1))),
             ("Buts / passes décisives", f"{st['goals']} / {st['assists']}"),
             ("xG / xA (FotMob)", f"{num(st['fm_xg'])} / {num(st['fm_xa'])}" if fm else "n.d."),
             ("Occasions créées / dribbles réussis", f"{st['fm_chances']:.0f} / {st['fm_dribbles']:.0f}" if fm else "n.d.")]
    rows_t1 = [[a, b, c, d] for (a, b), (c, d) in zip(left, right)]
    season = st["season"]
    pg.table(f"Tableau 1 — Valeur à T0 et statistiques de la saison {season}/{str(season + 1)[2:]} (toutes compétitions)",
             ["", "", "", ""], rows_t1, [0.30, 0.15, 0.40, 0.15], aligns=["left", "right", "left", "right"])
    obs = meta.get("live_obs")
    if obs is not None and len(obs):
        absent = int(obs.missed_injury.sum())
        pg.text(f"Saison en cours (FotMob, à la date T0) : {int(obs.matches.sum())} matchs, {obs.minutes.sum():.0f} minutes, "
                f"{obs.goals.sum():.0f} but(s), {num(obs.xg.sum())} xG"
                + (f" ; {absent} match(s) manqué(s) sur blessure ou absence." if absent else ".")
                + " Ces trimestres sont observés et mettent à jour l'état de départ.", size=9)
    pg.y -= 0.004

    # 2. Hypothèses
    pg.heading("2.  Hypothèses de la simulation")
    ev, val = ctx.ev, ctx.val
    start = meta["start_probs"]
    order = np.argsort(start)[::-1]
    start_txt = ", ".join(f"{config.STATES[k]} {pct(start[k])}" for k in order[:3] if start[k] >= 0.05)
    age_bin = int(np.digitize(meta["age0"], hmm.AGE_BINS[1:-1]))
    main_state = int(order[0])
    p_move_year = 1 - np.prod(1 - ev["move"][main_state, age_bin, :])
    dest = ev["destination"][age_bin, meta["tier"]]
    drift = ctx.drift[age_bin, meta["tier"]] if ctx.drift is not None else 0.0
    renew = ev["contract"]["renewal_prob_per_year"]
    from simulation import monte_carlo as mc
    infl = val["inflation"]
    assumptions = [
        ["Valeur initiale V0 et date T0", f"{eur(v0)} · {date_fr(sim_date)}", "Transfermarkt"],
        ["Horizon, pas de temps, simulations", f"6 ans · 24 trimestres · {meta['n']:,}".replace(",", " "), "—"],
        ["Niveau de départ (états cachés)", start_txt, "HMM filtré + FotMob 26/27"],
        ["Classe de talent (valeur / âge)", TIERS[meta["tier"]], "rang de valeur à T0"],
        ["Transitions entre états", "84 matrices âge × poste × talent", "Baum–Welch, 2014–2026"],
        ["Résilience (rebond après une baisse)", num(meta["resilience"], 2) + " (1 = moyenne)", "estimateur bayésien"],
        ["Transfert dans l'année / hors top 5 si départ", f"{pct(p_move_year)} / {pct(dest[2])}", "Transfermarkt 2014–2026"],
        ["Contrat : prolongation par an", f"{pct(renew['Rotation'])} (Rotation) à {pct(renew['Élite'])} (Élite)", "hypothèse"],
        ["Inflation du marché", f"+{num((np.exp(infl['drift']) - 1) * 100)} %/an (écart-type {num(infl['vol'] * 100)} %)", "indices saisonniers"],
        ["Bruit de marché · correction de dérive", f"{num(mc.RESID_SCALE, 2)} · {num((np.exp(drift) - 1) * 100)} %/an", "validation 2020→2026"],
    ]
    pg.table("Tableau 2 — Hypothèses retenues", ["Hypothèse", "Valeur retenue", "Source"], assumptions,
             [0.40, 0.36, 0.24], aligns=["left", "left", "left"], size=8.4, row_h=0.0168)
    pg.text("Les probabilités de transition, de transfert et l'inflation sont estimées sur 9 391 joueurs des cinq grands "
            "championnats ; seules les prolongations de contrat reposent sur une hypothèse (aucune source publique "
            "ne donne l'historique des contrats). Les valeurs sont exprimées aux prix du marché de T0, sauf mention "
            "« avec inflation ».", size=8.8)

    # 3. Résultats — Figure 1 (en bas de la page 1)
    pg.heading("3.  Résultats de la simulation")
    ax = pg.axes(0.2)
    spaghetti(ax, years, values / 1e6, "valeur marchande (M€, prix de T0)")
    ax.axhline(v0 / 1e6, color=ACCENT, lw=0.8, ls=":")
    ax.text(years[-1], v0 / 1e6, "V0 ", color=ACCENT, fontsize=8.5, va="bottom", ha="right")
    pg.caption("Figure 1 — Les 10 000 trajectoires simulées de la valeur marchande (gris), leur moyenne et leur médiane.")
    final = values[:, -1]
    mean6, med6, med1 = final.mean(), np.median(final), np.median(values[:, 4])
    pg.text(f"En médiane, la valeur passe de {eur(v0)} à {eur(med1)} après un an et à {eur(med6)} après six ans. "
            f"La moyenne ({eur(mean6)}) {'reste au-dessus de' if mean6 > med6 else 'est proche de'} la médiane : "
            f"quelques carrières exceptionnelles tirent la distribution vers le haut. La valeur à six ans dépasse V0 "
            f"dans {pct((final > v0).mean())} des carrières.", size=9)
    pg.save()


def _page_two(pdf, profile, sim, years, values, nominal, share, v0, ctx):
    pg = Page(pdf, 2)
    pg.y = 0.955
    final = values[:, -1]

    # Tableau 3
    rows = []
    for label, h in [("T0 + 1 an", 4), ("T0 + 2 ans", 8), ("T0 + 3 ans", 12), ("T0 + 6 ans", 24)]:
        v = values[:, h]
        rows.append([label, eur(v.mean()), eur(np.median(v)), eur(np.quantile(v, 0.1)), eur(np.quantile(v, 0.9)),
                     pct((v > v0).mean()), eur(np.median(nominal[:, h]))])
    pg.table("Tableau 3 — Distribution de la valeur marchande simulée",
             ["Horizon", "Moyenne", "Médiane", "P10", "P90", "P(V > V0)", "Médiane avec inflation"], rows,
             [0.15, 0.12, 0.12, 0.12, 0.12, 0.13, 0.24])
    w1 = np.quantile(values[:, 4], 0.9) / max(np.quantile(values[:, 4], 0.1), 1)
    w6 = np.quantile(final, 0.9) / max(np.quantile(final, 0.1), 1)
    pg.text(f"L'intervalle P10–P90 passe d'un facteur {num(w1)} à un an à un facteur {num(w6)} à six ans : "
            f"l'incertitude grandit avec l'horizon, ce qui est attendu pour une carrière exposée aux blessures, "
            f"aux transferts et à l'évolution du niveau.", size=9)
    pg.y -= 0.008

    # Figure 2
    ax = pg.axes(0.17)
    spaghetti(ax, years, share * 100, "temps de jeu (%, moyenne 4 trim.)")
    ax.set_ylim(0, 100)
    pg.caption("Figure 2 — Les 10 000 trajectoires de la part du temps de jeu en championnat, moyenne et médiane.")
    s0, s2, s6 = share[0, 0], np.median(share[:, 8]), np.median(share[:, -1])
    pg.text(f"Le temps de jeu médian passe de {pct(s0)} (quatre derniers trimestres) à {pct(s2)} dans deux ans, puis "
            f"{pct(s6)} à six ans. C'est le principal moteur de la valeur : un joueur qui ne s'impose pas comme "
            f"titulaire voit sa valeur se rapprocher de celle des joueurs de rotation.", size=9)
    pg.y -= 0.008

    # Tableau 4
    st = sim["state"]
    rev_total = sum(sim["revenue"].values())
    events = [
        ["Au moins un trimestre blessé dans l'année", pct((st[:, :4] == 0).any(axis=1).mean())],
        ["Titulaire ou mieux dans deux ans (T0 + 2 ans)", pct((st[:, 7] >= 3).mean())],
        ["Statut de star (Fort ou Élite la moitié du temps)", pct(((st >= 4).mean(axis=1) >= 0.5).mean())],
        ["Au moins un transfert ou prêt en six ans", pct(sim["moved"].any(axis=1).mean())],
        ["Hors des cinq grands championnats à six ans", pct(sim["meta"].get("left_top5", np.nan))],
        ["Valeur à six ans : au moins 2 × V0  /  au plus V0 / 2",
         f"{pct((final >= 2 * v0).mean())}  /  {pct((final <= v0 / 2).mean())}"],
        ["Revenus générés pour le club, médiane saison 1 / saison 6",
         f"{eur(np.median(rev_total[:, 0]))} / {eur(np.median(rev_total[:, -1]))}"],
    ]
    pg.table("Tableau 4 — Probabilités d'événements sur l'horizon de simulation", ["Événement", "Probabilité"], events,
             [0.75, 0.25])
    p_starter = (st[:, 7] >= 3).mean()
    p_move = sim["moved"].any(axis=1).mean()
    pg.text(f"Le scénario central est celui d'un joueur qui {'devient titulaire' if p_starter >= 0.5 else 'ne devient pas encore titulaire'} "
            f"à court terme ({pct(p_starter)} de chances de l'être dans deux ans) ; un transfert est probable "
            f"({pct(p_move)}) sur six ans. Les revenus sont des ordres de grandeur fondés sur des chiffres publics "
            f"(UEFA, Premier League, Deloitte) et des hypothèses explicites.", size=9)

    # Fiabilité
    rel = reliability()
    if rel:
        pg.y -= 0.006
        pg.text(f"Fiabilité — Validation hors échantillon (modèles ré-estimés en juillet 2020, comparés aux valeurs "
                f"réelles de juin 2026, {rel['players']} joueurs) : l'intervalle P10–P90 contient la vraie valeur dans "
                f"{pct(rel['coverage'])} des cas ; biais résiduel de {num(rel['bias'], 2)} (log), le modèle restant "
                f"légèrement optimiste pour les joueurs les plus chers.", size=8.2, style="italic", color="#333333")
    pg.save()
