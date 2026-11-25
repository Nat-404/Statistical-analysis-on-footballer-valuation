# =============================================================================
# app/simulation_app.py — Fenêtre de simulation (sans Terminal).
#
# On tape le nom d'un joueur (des suggestions s'affichent), on clique sur
# "Simulation" : la barre d'avancement suit les étapes de Main.analyse(), puis
# la fiche PDF de 2 pages (la même que python Main.py "Nom") est enregistrée
# dans le dossier Téléchargements.
#
# Lancement : double-clic sur "Football Valuation.app" (créée par
# app/make_app.sh), ou  .venv/bin/python -m app.simulation_app
#
# La fenêtre utilise la police des deux documents (Latin Modern, fichiers de
# reporting/fonts) : elle est déclarée pour CE programme uniquement, rien
# n'est installé sur le Mac.
# =============================================================================

import ctypes
import math
import queue
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tkinter as tk                                   # noqa: E402
from tkinter import ttk                                # noqa: E402

DOWNLOADS = Path.home() / "Downloads"
FONTS = ROOT / "reporting" / "fonts"

# Couleurs des documents (reporting/fiche.py)
INK, ACCENT = "#111111", "#0b2a66"                     # texte, bleu foncé des courbes
PALE, MUTED, PAPER = "#8fb8e8", "#666666", "#ffffff"
SERIF, CAPS = "Latin Modern Roman", "Latin Modern Roman Caps"

# Étapes de Main.analyse() : (début, fin) sur la barre, durée typique en secondes.
# Pour les étapes sans avancement mesurable (chargement, requêtes en ligne), la
# barre progresse selon le temps écoulé sans jamais atteindre la fin de l'étape.
STEPS_ONLINE = {"models": (0.00, 0.15, 25), "fotmob": (0.15, 0.30, 20), "press": (0.30, 0.55, 90),
                "simulation": (0.55, 0.93, None), "pdf": (0.93, 1.00, 8)}
STEPS_OFFLINE = {"models": (0.00, 0.25, 25), "simulation": (0.25, 0.90, None), "pdf": (0.90, 1.00, 8)}
LABELS = {"models": "Chargement des modèles…",
          "fotmob": "Matchs de la saison en cours (FotMob)…",
          "press": "Contrôle presse (GDELT, jusqu'à 3 minutes)…",
          "simulation": "Simulation de 10 000 carrières… {pct} %",
          "pdf": "Mise en page de la fiche PDF…"}


def register_fonts():
    # Rend les polices Latin Modern disponibles pour ce programme seulement
    # (CoreText, portée "processus"). Sans effet hors macOS : police de repli.
    try:
        ct = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreText.framework/CoreText")
        cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    except OSError:
        return
    cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
    cf.CFURLCreateFromFileSystemRepresentation.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long,
                                                           ctypes.c_bool]
    ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
    ct.CTFontManagerRegisterFontsForURL.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    for font in FONTS.glob("*.otf"):
        raw = str(font).encode()
        url = cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), False)
        ct.CTFontManagerRegisterFontsForURL(url, 1, None)          # 1 = ce processus uniquement


def normalize(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def load_players():
    # Joueurs analysables (ceux de la chaîne de Markov), avec leur dernier âge et
    # leur dernière valeur, pour les suggestions. Lecture rapide (< 1 seconde).
    import pandas as pd
    import config
    known = pd.read_parquet(config.PROCESSED_DIR / "hmm_states.parquet", columns=["player_id"]).player_id.unique()
    q = pd.read_parquet(config.PROCESSED_DIR / "quarterly.parquet",
                        columns=["player_id", "name", "t", "age", "market_value"])
    q = q[q.player_id.isin(known) & q.name.notna()].sort_values("t").groupby("player_id").tail(1)
    q = q.sort_values("market_value", ascending=False)
    q["key"] = q.name.map(normalize)
    return q[["player_id", "name", "age", "market_value", "key"]].reset_index(drop=True)


def describe(row):
    value = f"{row.market_value / 1e6:.1f} M€" if row.market_value >= 1e6 else f"{row.market_value / 1e3:.0f} k€"
    return f"{row.name}  ·  {row.age:.0f} ans  ·  {value}"


class App:
    def __init__(self, root):
        self.root, self.players, self.matches = root, None, []
        self.events, self.running, self.result = queue.Queue(), False, None
        self.steps, self.step, self.step_started, self.sim_fraction = STEPS_ONLINE, None, 0.0, 0.0
        root.title("Football Player Valuation")
        root.configure(bg=PAPER)
        root.resizable(False, False)
        self._style()
        self._layout()
        threading.Thread(target=self._load_players, daemon=True).start()
        root.after(100, self._tick)

    # ---------------------------------------------------------------- interface
    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=PAPER)
        style.configure("TCheckbutton", background=PAPER, foreground=MUTED, font=(SERIF, 12))
        style.map("TCheckbutton", background=[("active", PAPER)])
        style.configure("Run.TButton", font=(SERIF, 15), foreground=PAPER, background=ACCENT,
                        borderwidth=0, focusthickness=0, padding=(22, 8))
        style.map("Run.TButton", background=[("disabled", PALE), ("active", "#1f4fa8")],
                  foreground=[("disabled", PAPER)])
        style.configure("Link.TButton", font=(SERIF, 12), foreground=ACCENT, background=PAPER,
                        borderwidth=1, bordercolor=PALE, padding=(10, 4))
        style.map("Link.TButton", background=[("active", "#eef4fc")])
        style.configure("Bar.Horizontal.TProgressbar", troughcolor="#eef4fc", background=ACCENT,
                        bordercolor="#eef4fc", lightcolor=ACCENT, darkcolor=ACCENT, thickness=10)

    def _layout(self):
        frame = ttk.Frame(self.root, padding=(36, 28, 36, 26))
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Football Player Valuation", font=(CAPS, 24), fg=INK, bg=PAPER).pack(anchor="w")
        tk.Label(frame, text="Hidden Markov model and Monte Carlo simulation  —  six-year market-value projection",
                 font=(SERIF, 12, "italic"), fg=MUTED, bg=PAPER).pack(anchor="w", pady=(2, 18))
        tk.Frame(frame, height=1, bg=INK).pack(fill="x", pady=(0, 18))

        tk.Label(frame, text="Joueur", font=(SERIF, 13), fg=INK, bg=PAPER).pack(anchor="w")
        self.query = tk.StringVar()
        self.entry = tk.Entry(frame, textvariable=self.query, font=(SERIF, 16), fg=INK, bg=PAPER, width=44,
                              relief="flat", highlightthickness=1, highlightbackground=PALE,
                              highlightcolor=ACCENT, insertbackground=INK)
        self.entry.pack(anchor="w", fill="x", ipady=6, pady=(4, 0))
        self.entry.focus_set()
        self.query.trace_add("write", lambda *_: self._suggest())
        self.entry.bind("<Down>", lambda _: self._focus_list())
        self.entry.bind("<Return>", lambda _: self.start())

        # Suggestions (affichées seulement pendant la saisie)
        self.list_frame = tk.Frame(frame, bg=PAPER)
        self.listbox = tk.Listbox(self.list_frame, font=(SERIF, 13), height=5, fg=INK, bg=PAPER, relief="flat",
                                  highlightthickness=1, highlightbackground=PALE, selectbackground="#eef4fc",
                                  selectforeground=ACCENT, activestyle="none")
        self.listbox.pack(fill="x")
        self.listbox.bind("<<ListboxSelect>>", lambda _: self._pick())
        self.listbox.bind("<Return>", lambda _: (self._pick(), self.start()))
        self.listbox.bind("<Double-Button-1>", lambda _: (self._pick(), self.start()))
        self.list_anchor = tk.Frame(frame, height=0, bg=PAPER)
        self.list_anchor.pack(fill="x")

        self.offline = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Sans connexion (plus rapide, sans matchs récents ni contrôle presse)",
                        variable=self.offline).pack(anchor="w", pady=(12, 16))

        self.button = ttk.Button(frame, text="Simulation", style="Run.TButton", command=self.start)
        self.button.pack(anchor="w")

        self.bar = ttk.Progressbar(frame, style="Bar.Horizontal.TProgressbar", maximum=1000, length=520)
        self.bar.pack(fill="x", pady=(22, 6))
        self.status = tk.Label(frame, text="Chargement de la liste des joueurs…", font=(SERIF, 12),
                               fg=MUTED, bg=PAPER, anchor="w", justify="left", wraplength=560)
        self.status.pack(anchor="w", fill="x")

        self.after_row = ttk.Frame(frame)
        self.after_row.pack(anchor="w", pady=(10, 0))
        self.open_btn = ttk.Button(self.after_row, text="Ouvrir le PDF", style="Link.TButton",
                                   command=lambda: subprocess.run(["open", str(self.result)]))
        self.reveal_btn = ttk.Button(self.after_row, text="Afficher dans le Finder", style="Link.TButton",
                                     command=lambda: subprocess.run(["open", "-R", str(self.result)]))
        tk.Label(frame, text="Statistical scenarios, not financial advice or official valuations.",
                 font=(SERIF, 10, "italic"), fg=MUTED, bg=PAPER).pack(anchor="w", pady=(16, 0))

    # ------------------------------------------------------------ suggestions
    def _load_players(self):
        try:
            self.players = load_players()
            self.events.put(("ready", len(self.players)))
        except Exception as error:                     # noqa: BLE001 : on affiche l'erreur
            self.events.put(("error", f"Données introuvables ({error}). Lancer d'abord python pipeline.py."))

    def _suggest(self):
        self.picked = None
        text = normalize(self.query.get().strip())
        if self.players is None or len(text) < 2 or self.running:
            self.list_frame.pack_forget()
            return
        hits = self.players[self.players.key.str.contains(text, regex=False)].head(5)
        self.matches = list(hits.itertuples())
        self.listbox.delete(0, "end")
        for row in self.matches:
            self.listbox.insert("end", describe(row))
        if self.matches and not (len(self.matches) == 1 and self.matches[0].key == text):
            self.listbox.configure(height=len(self.matches))
            self.list_frame.pack(in_=self.list_anchor, fill="x")
        else:
            self.list_frame.pack_forget()

    def _focus_list(self):
        if self.matches:
            self.listbox.focus_set()
            self.listbox.selection_set(0)

    def _pick(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        row = self.matches[sel[0]]
        self.query.set(row.name)                        # (efface self.picked via _suggest)
        self.picked = row
        self.list_frame.pack_forget()
        self.entry.focus_set()
        self.entry.icursor("end")

    def _resolve(self):
        # Joueur choisi dans la liste, sinon le plus cher de ceux qui correspondent
        if getattr(self, "picked", None) is not None:
            return self.picked
        text = normalize(self.query.get().strip())
        if self.players is None or not text:
            return None
        exact = self.players[self.players.key == text]
        hits = exact if len(exact) else self.players[self.players.key.str.contains(text, regex=False)]
        return next(hits.itertuples(), None)

    # ------------------------------------------------------------- simulation
    def start(self):
        if self.running:
            return
        row = self._resolve()
        if row is None:
            self._set_status("Aucun joueur ne correspond à ce nom." if self.players is not None
                             else "La liste des joueurs n'est pas encore chargée.", error=True)
            return
        self.running, self.result = True, None
        self.list_frame.pack_forget()
        self.open_btn.pack_forget()
        self.reveal_btn.pack_forget()
        self.button.state(["disabled"])
        self.entry.configure(state="disabled")
        self.query.set(row.name)
        self.steps = STEPS_OFFLINE if self.offline.get() else STEPS_ONLINE
        self.bar["value"] = 0
        self.started = time.time()
        threading.Thread(target=self._work, args=(int(row.player_id), self.offline.get()), daemon=True).start()

    def _work(self, player_id, offline):
        # Fil de calcul : n'agit JAMAIS sur la fenêtre directement (Tk n'est pas
        # sûr entre fils) ; il envoie des messages que _tick() lit.
        try:
            import Main
            pdf = Main.analyse(player_id=player_id, offline=offline, out_dir=DOWNLOADS,
                               progress=lambda step, f: self.events.put(("step", step, f)))
            self.events.put(("done", pdf))
        except Exception as error:                     # noqa: BLE001
            traceback.print_exc()
            self.events.put(("error", f"La simulation a échoué : {error}"))

    def _tick(self):
        while not self.events.empty():
            event = self.events.get()
            kind = event[0]
            if kind == "ready":
                self._set_status(f"{event[1]:,} joueurs disponibles. Tapez un nom puis cliquez sur « Simulation ».".replace(",", " "))
                self._suggest()
            elif kind == "step":
                _, step, fraction = event
                if step != self.step:
                    self.step, self.step_started = step, time.time()
                self.sim_fraction = fraction
            elif kind == "done":
                self._finish(event[1])
            elif kind == "error":
                self._finish(None, event[1])
        if self.running and self.step in self.steps:
            lo, hi, expected = self.steps[self.step]
            if expected is None:
                f = self.sim_fraction
            else:
                f = 0.97 * (1 - math.exp(-(time.time() - self.step_started) / expected))
            self.bar["value"] = max(self.bar["value"], 1000 * (lo + (hi - lo) * f))
            self._set_status(LABELS[self.step].format(pct=int(self.sim_fraction * 100)))
        self.root.after(100, self._tick)

    def _finish(self, pdf, error=None):
        self.running, self.step = False, None
        self.button.state(["!disabled"])
        self.entry.configure(state="normal")
        if error:
            self.bar["value"] = 0
            self._set_status(error, error=True)
            return
        self.result = Path(pdf)
        self.bar["value"] = 1000
        minutes, seconds = divmod(int(time.time() - self.started), 60)
        self._set_status(f"✓  Fiche enregistrée dans Téléchargements : {self.result.name}   "
                         f"({minutes} min {seconds:02d} s)", done=True)
        self.open_btn.pack(side="left")
        self.reveal_btn.pack(side="left", padx=(8, 0))

    def _set_status(self, text, error=False, done=False):
        self.status.configure(text=text, fg="#b03a2e" if error else (ACCENT if done else MUTED))


def main():
    import os
    os.chdir(ROOT)                      # certains modules lisent data/ en chemin relatif
    register_fonts()
    root = tk.Tk()
    App(root)
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    main()
