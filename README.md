# CortexHR — AI/ML Phase 1

**Author:** Eniyan J  
**Track:** AI/ML  
**Phase:** Phase 1 — Data Audit, Feature Engineering, Baseline Modeling, Evaluation, and Anomaly Detection  
**GitHub Repository:** https://github.com/Eniyan-j/cortexhr  
**Git Tag:** `phase-1`

## Project Overview

CortexHR is an HR analytics project focused on auditing employee attrition data, preparing features for machine learning, establishing a baseline attrition prediction model, evaluating prediction quality and calibration, and detecting unusual attendance patterns.

The project emphasizes data quality, avoiding target leakage, reproducible experiments, and interpretable evaluation results.

## Phase 1 Tasks

| Task | Description | Main Evidence |
|---|---|---|
| Task 1 | Data audit and exploration | `01_data_audit_exploration.ipynb` |
| Task 2 | Feature engineering and reusable pipeline | `02_feature_pipeline.ipynb`, `src/features.py` |
| Task 3 | Baseline attrition model | `03_baseline.ipynb` |
| Task 4 | Model evaluation and probability calibration | `04_evaluation.ipynb` |
| Task 5 | Attendance anomaly detection | `05_attendance_anomaly.ipynb` |

## Repository Structure

```text
cortexhr/
├── 01_data_audit_exploration.ipynb
├── 02_feature_pipeline.ipynb
├── 03_baseline.ipynb
├── 04_evaluation.ipynb
├── 05_attendance_anomaly.ipynb
├── src/
│   ├── features.py(build_pipeline)
│   └── feature.py(add_attrition_features)
├── reports/
├── models/
├── data/
│   ├── raw/
│   └── processed/
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

The `data/` directory is expected to contain local input and processed datasets when available. Dataset files should not be committed if they contain confidential employee information or restricted answer-key data.

## Technologies Used

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Jupyter Notebook / JupyterLab
- Docker
- Git and GitHub

## Environment Setup

Docker is used to provide a reproducible Python environment.

### Prerequisites

Install Docker Desktop and Git. Clone the repository:

```bash
git clone https://github.com/Eniyan-j/cortexhr.git
cd cortexhr
```

Ensure that the authorized input dataset is available at:

```text
data/raw/02_CortexHR_AI_ML_Dataset.xlsx
```

The dataset is not guaranteed to be included in the repository. If it is restricted, obtain it through the authorized project channel.

### Build the Docker image

```bash
docker build -t cortexhr .
```

### Start JupyterLab

**Windows PowerShell:**

```powershell
docker run --rm -p 127.0.0.1:8888:8888 -v "${PWD}:/workspace" cortexhr
```

**Linux or macOS:**

```bash
docker run --rm -p 127.0.0.1:8888:8888 -v "$(pwd):/workspace" cortexhr
```

Open the local URL printed by Jupyter in the terminal. If the Docker image does not start Jupyter automatically, check the startup command in the `Dockerfile`.

The container mounts the repository at `/workspace`, allowing notebooks to access project files.

## Running the Notebooks

Run the notebooks in the following order:

1. **`01_data_audit_exploration.ipynb`** — inspect data quality, missing values, duplicates, invalid values, outliers, and potential leakage risks.
2. **`02_feature_pipeline.ipynb`** — build and validate the feature engineering workflow.
3. **`03_baseline.ipynb`** — train and compare baseline attrition models.
4. **`04_evaluation.ipynb`** — evaluate prediction performance and probability calibration.
5. **`05_attendance_anomaly.ipynb`** — detect unusual attendance records and evaluate detection against available labeled records.

Run each notebook from top to bottom. Some later
