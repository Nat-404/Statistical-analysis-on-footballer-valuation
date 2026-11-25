# =============================================================================
# paper/build_html.py — Convertit paper.tex en paper.html (sans moteur LaTeX).
#
# Pandoc ne sait pas numéroter les citations d'un environnement thebibliography
# ni les équations : on pré-traite donc le .tex (citations -> [n], \eqref -> (n)
# avec la même numérotation que LaTeX), puis on appelle pandoc + MathJax.
# Le PDF s'obtient ensuite en imprimant paper.html (navigateur ou Playwright).
# =============================================================================

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def preprocess(tex):
    # Les résultats sont insérés en ligne (pandoc ne lit pas \input)
    tex = tex.replace(r"\input{results.tex}", (HERE / "results.tex").read_text(encoding="utf-8"))
    # Chiffres du texte (macros \num...) : remplacés directement par leur valeur
    macros = dict(re.findall(r"\\newcommand\{\\(num\w+)\}\{([^}]*)\}",
                             (HERE / "numbers.tex").read_text(encoding="utf-8")))
    tex = tex.replace(r"\input{numbers.tex}", "")
    tex = re.sub(r"\\(num[A-Za-z]+)(\{\})?", lambda m: macros[m.group(1)], tex)

    # Citations : numéros dans l'ordre de la bibliographie
    keys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    number = {k: i + 1 for i, k in enumerate(keys)}
    tex = re.sub(r"\\cite\{([^}]+)\}",
                 lambda m: "[" + ", ".join(str(number[k.strip()]) for k in m.group(1).split(",")) + "]", tex)
    tex = re.sub(r"\\bibitem\{([^}]+)\}", lambda m: rf"\item[{{[{number[m.group(1)]}]}}]", tex)
    tex = tex.replace(r"\begin{thebibliography}{99}", r"\section*{References}\begin{description}")
    tex = tex.replace(r"\end{thebibliography}", r"\end{description}")

    # Équations : numérotées "section.n" comme amsart, avec \tag pour l'affichage MathJax
    labels, section, count = {}, 0, 0
    out = []
    for line in tex.split("\n"):
        if re.match(r"\\section\{", line):
            section, count = section + 1, 0
        m = re.search(r"\\label\{(eq:[^}]+)\}", line)
        if m:
            count += 1
            labels[m.group(1)] = f"{section}.{count}"
            line = line.replace(m.group(0), rf"\tag{{{section}.{count}}}")
        out.append(line)
    tex = "\n".join(out)
    tex = re.sub(r"\\eqref\{(eq:[^}]+)\}", lambda m: f"({labels.get(m.group(1), '?')})", tex)
    return tex


def main():
    source = preprocess((HERE / "paper.tex").read_text(encoding="utf-8"))
    tmp = HERE / "_paper_html.tex"
    tmp.write_text(source, encoding="utf-8")
    # "?v=..." : force le navigateur à recharger la feuille de style à chaque construction
    import time
    subprocess.run(["pandoc", str(tmp), "-s", "--mathjax", "--toc", "--toc-depth=2", "--number-sections",
                    "-c", f"amsart.css?v={int(time.time())}", "-o", str(HERE / "paper.html")], check=True, cwd=HERE)
    tmp.unlink()
    print("✓ paper/paper.html")


if __name__ == "__main__":
    main()
