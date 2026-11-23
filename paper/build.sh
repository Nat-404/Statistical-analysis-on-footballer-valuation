#!/bin/sh
# Construit le papier : figures + section Results, puis conversion LaTeX -> HTML (pandoc + MathJax).
# Le PDF s'obtient en imprimant paper.html depuis un navigateur (Fichier > Imprimer > Enregistrer en PDF,
# format Lettre US, marges par défaut), ou en compilant paper.tex avec un moteur LaTeX si installé.
set -e
cd "$(dirname "$0")/.."
.venv/bin/python -W ignore -m paper.make_figures
.venv/bin/python paper/build_html.py
