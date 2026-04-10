python -m src.data.generate_data
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m src.training.train
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m src.training.hyperparameter_search
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m src.visualization.plot_weights
