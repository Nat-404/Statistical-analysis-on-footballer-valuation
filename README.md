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

Projections are statistical scenarios, not financial advice or official valuations.
