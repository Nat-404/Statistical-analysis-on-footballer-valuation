# =============================================================================
# pipeline.py — Reconstruit TOUT le projet, dans l'ordre, à partir des données.
#
#     python pipeline.py            -> tout reconstruire
#     python pipeline.py --from hmm -> reprendre à partir d'une étape
#
# À lancer quand de nouvelles données sont arrivées (fin de la collecte FotMob,
# nouvelles pages vues Wikipédia...). Chaque étape lit le résultat de la
# précédente : l'ordre compte.
# =============================================================================

import sys
import time


def step_elo():
    from features import elo
    elo.build()


def step_stakes():
    from features import stakes
    stakes.build()


def step_matching():
    from collectors import matching
    players, clubs = matching.build()
    print(f"  {len(players)} joueurs FotMob reliés à Transfermarkt, {len(clubs)} clubs")


def step_quarterly():
    from features import quarterly
    df = quarterly.build()
    print(f"  {len(df):,} lignes joueur x trimestre")


def step_performance():
    from models import performance
    performance.build()


def step_hmm():
    from models import hmm
    hmm.build()


def step_valuation():
    from models import valuation
    valuation.build()


def step_events():
    from models import events
    events.build()


def step_collective():
    from models import collective
    collective.build()


def step_calibration():
    from simulation import calibration
    calibration.build()


STEPS = [("elo", step_elo), ("stakes", step_stakes), ("matching", step_matching),
         ("quarterly", step_quarterly), ("performance", step_performance), ("hmm", step_hmm),
         ("valuation", step_valuation), ("events", step_events), ("collective", step_collective),
         ("calibration", step_calibration)]


def main():
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else STEPS[0][0]
    names = [n for n, _ in STEPS]
    if start not in names:
        sys.exit(f"Étape inconnue : {start}. Étapes : {', '.join(names)}")
    for name, func in STEPS[names.index(start):]:
        t = time.time()
        print(f"▶ {name}…")
        func()
        print(f"  ✓ {name} ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
