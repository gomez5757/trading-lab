# Trading Lab

Proyecto base para backtesting y optimizacion por fases usando Python,
GitHub Actions y GitHub Codespaces.

## Uso rapido local

```bash
python -m pip install -e ".[dev]"
python scripts/run_backtest.py --config configs/base.yaml
python scripts/run_optimization_stage.py --config configs/optimization.yaml --stage 0 --total-stages 16
python scripts/merge_leaderboards.py --input-glob "outputs/optimization/stage_*.csv"
```

## En GitHub

- `ci.yml`: ejecuta tests en push y pull request.
- `backtest-manual.yml`: lanza un backtest manual desde Actions.
- `optimization-staged.yml`: divide la optimizacion en 16 jobs y une el leaderboard final.

## Formato de datos

El CSV debe contener:

```text
timestamp,open,high,low,close,volume
```

No subas datos sensibles, claves API ni estrategias privadas a un repo publico.
El mercado ya muerde bastante sin darle cubiertos.
