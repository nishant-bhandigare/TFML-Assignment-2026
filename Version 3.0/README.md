# TFML Assignment - Version 3.0

TensorFlow implementation of the B/0/E noisy character recognition assignment.

## Directory structure

```
Version 3.0/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── checkpoints/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── visualization/
├── webapp/
│   ├── static/
│   └── templates/
├── requirements.txt
└── README.md
```

## First run

1) Create env and install dependencies:

```powershell
cd "D:\MTech - Computer Science and Engineering\2nd Semester\1. TFML\TFML Assignment\Version 3.0"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Generate data and train:

```powershell
python -m src.data.generate_data
python -m src.training.train
```

3) Run evaluation experiments:

```powershell
python -m src.training.hyperparameter_search
python -m src.visualization.plot_weights
```

4) Start web app:

```powershell
python webapp/app.py
```

Open http://127.0.0.1:5000

## Notes for best accuracy

- The model uses standardized inputs, Adam optimizer, early stopping and LR scheduling.
- Architecture search script compares 64-X-3 across multiple X values.
- Sample complexity plot is generated for the report section (part e).
