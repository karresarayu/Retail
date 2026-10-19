# RetailPulse

A Streamlit dashboard and analysis notebooks for retail sales, customer
segmentation and churn, demand forecasting, and inventory optimization.

## Privacy and repository contents

This public repository contains application and analysis source only. Retail
datasets, customer-level exports, trained models, and generated reports are
excluded because they may contain customer-level information. The dashboard
and notebooks need the corresponding local input data and generated artifacts
to display or reproduce results; those files are not included here.

Notebook outputs are cleared in the published copies to avoid sharing
customer-level values embedded in saved results.

## Run the dashboard

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Place any authorized input data and generated artifacts in their expected
`data/`, `models/`, and `reports/` paths locally. These directories are ignored
by Git.
