# Forecasting National Renewable Energy Share

A country-level forecasting and risk-classification tool built for the International Energy Agency (IEA) to help it spot which countries are on track with the renewable transition and which are falling behind.

**Course:** INDENG 290, Energy Analytics, UC Berkeley
**Stakeholder:** International Energy Agency (IEA)
**Term:** Spring 2026

## Team

- Matthew N. Shafik
- Rameen Faisal
- Rimsha Ijaz
- Yifei Zhan

## Project Overview

The IEA tracks energy progress across the world but cannot give every country equal attention. This project gives the agency a baseline forecast of where each country is headed, a risk tier for each country, and a clear answer about which variables actually help predict the renewable share. Together these three pieces form a simple early-warning tool the IEA can update each year as new data comes in.

The work covers 174 countries from 2000 to 2020, using the Global Sustainable Energy Dataset from Kaggle.

## Research Questions

1. **Forecasting (RQ1).** What will each country's renewable energy share be in 2027 to 2029?
2. **Transition Risk (RQ2).** Which countries are at risk of stagnation or decline?
3. **Feature Importance (RQ3).** Which variables most improve forecast performance?

Each question has its own folder with code, notebook, written analysis, and figures.

## Repo Structure

```
energy-analytics/
├── README.md
├── Energy_Report_Final.docx          # Consolidated final report (all three RQs)
├── data_cleaning_EDA.ipynb           # Shared cleaning + EDA pipeline
├── global-data-on-sustainable-energy.csv   # Raw Kaggle dataset
│
├── Question 1 - Forecasting/
│   ├── Question1-forecasting-rimsha-ijaz.ipynb
│   ├── forecast_2027_2029.csv
│   ├── renewable_energy_interactive_dashboard.html
│   └── written analysis of question 1.docx
│
├── Question 2 - Transition Risk/
│   ├── Transition_Risk_Model_Q2_quadratic.ipynb
│   ├── transition_risk_code_quadratic.py
│   ├── Transition_Risk_Analysis.docx
│   └── figures/
│       ├── fig1_risk_distribution.png
│       ├── fig2_top20_at_risk.png
│       ├── fig3_trajectories_curved.png
│       ├── fig4_quadratic_coef.png
│       └── fig5_baseline_vs_forecast.png
│
├── Question 3 - Feature Importance/
│   ├── data_cleaning_EDA_+_research_q3.ipynb
│   ├── RQ3.docx
│   └── figures/
│       ├── forecast accuracy by feature set.png
│       ├── effect of removing feature groups.png
│       ├── lag model actual vs predicted.png
│       ├── full model actual vs predicted.png
│       ├── residual distribution.png
│       └── top feature importances.png
│
└── presentation/
    └── Presentation_Final.pptx
```

## Data

- **Source:** Global Sustainable Energy Dataset on Kaggle
  https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy
- **Coverage:** 176 countries, 2000 to 2020
- **Target variable:** Renewable energy share in total final energy consumption (percent)

The cleaning pipeline standardises column names, removes duplicates, restricts to the 2000 to 2020 window, fills small gaps with interpolation and forward/back fill, drops columns with more than 40 percent missing values, and winsorises the tails at the 1st and 99th percentiles.

## Quick Start

### Requirements

- Python 3.10 or newer
- Jupyter (Lab or Notebook)
- Standard scientific stack: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `plotly`

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-handle>/energy-analytics.git
cd energy-analytics

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
.venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn plotly jupyterlab openpyxl

# 4. Launch Jupyter
jupyter lab
```

### Run order

1. Open `data_cleaning_EDA.ipynb` and run all cells. This produces the cleaned dataset used by every research question.
2. For RQ1, open `Question 1 - Forecasting/Question1-forecasting-rimsha-ijaz.ipynb`. The notebook trains five forecasting models (Naive, AR(1), AR(p), ETS, ARX), runs the rolling backtest, and writes `forecast_2027_2029.csv`.
3. For RQ2, open `Question 2 - Transition Risk/Transition_Risk_Model_Q2_quadratic.ipynb`. It reads the RQ1 forecast, classifies each country into HIGH, MEDIUM, or LOW risk, and writes the figures into the folder.
4. For RQ3, open `Question 3 - Feature Importance/data_cleaning_EDA_+_research_q3.ipynb`. It runs the three nested Ridge models, the leave-one-group-out analysis, and the diagnostic plots.

## Key Findings

- **Forecast.** ETS gave the best balance of accuracy and trend detection (MAE of 2.51 percentage points and 61 percent directional accuracy at the three year horizon). 91 of 174 countries are forecast to see their renewable share decline by 2029 if current trends hold.
- **Risk.** 35 countries are HIGH RISK, 56 are MEDIUM RISK, and 83 are LOW RISK. 40 percent of low-income countries fall into HIGH RISK versus 0 percent of high-income countries, driven by the denominator effect where total energy demand grows faster than renewable capacity.
- **Feature importance.** A model using only lagged renewable share beats every richer feature set. Adding energy, macroeconomic, or development variables makes RMSE up to four times worse. For short-term forecasting, simpler is really better.

## Outputs You Can Open Directly

- `Energy_Report_Final.docx`: full consolidated report covering all three research questions, with figures and tables.
- `forecast_2027_2029.csv`: country-level forecasts for 2027 to 2029, with 2020 baseline and income group.
- `renewable_energy_interactive_dashboard.html`: interactive dashboard for the RQ1 forecasts.
- `presentation/Presentation_Final.pptx`: slide deck used for the project presentation.

## Limitations

- Data ends in 2020, so the model cannot see the Covid recovery, the 2022 energy crisis, the Inflation Reduction Act, or any post-2020 policy shifts.
- Forecast uncertainty grows with horizon. The 2027 to 2029 numbers should be read as a "what if nothing changes" baseline, not point predictions.
- Validation in RQ1 used five hand-picked countries. A larger, randomly selected validation set would be a useful next step.
- Risk thresholds are uniform across countries. Country-specific thresholds tied to IEA Net Zero pathway targets would give a more contextual risk read.

## License and Citation

This work was produced for INDENG 290 at UC Berkeley. The underlying dataset belongs to its original authors on Kaggle. Please cite the dataset and the IEA's Net Zero by 2050 framework if you build on this work.

## Contact

For questions about the project, please reach out to any of the team members listed above.
