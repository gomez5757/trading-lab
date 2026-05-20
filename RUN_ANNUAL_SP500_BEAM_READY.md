# Run preparado: Annual SP500 Beam Prediction

Estado: preparado, no ejecutado.

Workflow:

https://github.com/gomez5757/trading-lab/actions/workflows/annual-sp500-beam.yml

Configuracion:

- Config: `configs/annual_sp500_beam.yaml`
- Datos: descarga publica automatica al empezar el workflow
- Panel: `data/public/sp500_annual_daily.csv`
- Features solicitadas: `configs/annual_feature_manifest.csv`
- Total features en manifiesto: 144
- Inicio de datos objetivo: 1980
- Fin configurado: 2025
- Metodo: Beam anual
- Jobs: 64 stages en paralelo
- Por stage: seed pool 500, beam width 32, generations 6, mutations per parent 12
- Locked: cerrado
- Trading: solo prediccion anual del SP500, no reutiliza senales diarias

Artifacts esperados:

- `annual-sp500-public-panel`
- `annual-sp500-feature-audit`
- `annual-sp500-beam-stage-0` ... `annual-sp500-beam-stage-63`
- `annual-sp500-beam-leaderboard`

Que mirar al terminar:

- `annual_sp500_beam_summary.json`
- `annual_sp500_beam_leaderboard.csv`
- `annual_feature_coverage_summary.json`
- `annual_feature_coverage.csv`

Reglas importantes:

- No se abre locked.
- Las 144 features estan en el manifiesto.
- Las features sin fuente publica fiable no se inventan.
- El audit separa features buenas, proxy, parciales y sin datos.
- El Beam solo usa features con datos suficientes en train.

Como arrancarlo manualmente:

1. Abrir el workflow de GitHub.
2. Pulsar `Run workflow`.
3. Dejar `configs/annual_sp500_beam.yaml`.
4. Pulsar el boton verde.

