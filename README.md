# Statistical analysis on footballer valuation

Football player market-value forecasting over a 6-year horizon: a quarterly hidden Markov model of
career states (injured → elite), event sub-models (transfers, contracts, tournaments) and a
10,000-path Monte Carlo simulation, validated out of sample (2020 → 2026).

- **Example output:** [examples/Rio_Ngumoha_projection.pdf](examples/Rio_Ngumoha_projection.pdf), a 2-page projection fiche
- **Method paper:** [examples/Markov_Football_Valuation_paper.pdf](examples/Markov_Football_Valuation_paper.pdf) (source in `paper/`)

```sh
python pipeline.py            # train every model from the collected data
python Main.py "Rio Ngumoha"  # 2-page projection fiche in outputs/
```

**Window version (macOS):** `sh app/make_app.sh` builds `Football Valuation.app` (also copied to the
Desktop). Double-click it, type a player's name, click **Simulation**: a progress bar follows the run
and the same 2-page fiche is saved to `~/Downloads`.

When a player is analysed, his current-season matches are fetched from FotMob and a press check of
off-field incidents is run on GDELT (2017 to the day of the analysis; shown on the fiche, not used in
the simulation). `--offline` skips both.

Projections are statistical scenarios, not financial advice or official valuations.

## Possible next steps

- **Star decline:** players worth more than €20m are still overestimated by about +0.27 in log
  (about 30 %) at six years in the 2020 → 2026 validation; one-year errors are similar across value
  bands, so the gap builds up over several years and a decline mechanism is still missing.
- **Going public:** the repository is private; before opening it, rewrite the commit author email
  to a GitHub no-reply address and add a licence.
- **GDELT in bulk:** the GDELT DOC API is too heavily rate-limited for a historical collection; using
  GDELT through Google BigQuery (needs a Google Cloud account) would allow testing press incidents
  as a model input instead of an on-demand check shown on the fiche.
