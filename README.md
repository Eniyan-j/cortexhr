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



**Phase 2 Project | AI/ML Engineering**

CortexHR is an AI/ML-based HR analytics and data auditing project designed to identify employee attrition risk, detect attendance anomalies, flag payroll irregularities, expose model predictions through an API, and support HR policy queries through a chatbot.

The project uses structured HR datasets to develop machine learning models, apply anomaly detection techniques, generate explanations for flagged payroll records, and build API-based services.

---

## Project Objectives

- Analyze HR datasets and validate data quality.
- Develop a machine learning model to identify employee attrition risk.
- Detect unusual attendance patterns using anomaly detection.
- Identify potential payroll irregularities and financial risks.
- Generate understandable explanations for flagged payroll records.
- Expose model predictions through a FastAPI application.
- Develop an HR chatbot using intent classification and FAQ-based responses.
- Organize the project into reproducible notebooks, reusable modules, model artifacts, tests, and reports.

---

## Technology Stack

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** TF-IDF, Logistic Regression
- **API Development:** FastAPI, Pydantic
- **Development Environment:** JupyterLab
- **Containerization:** Docker
- **Dataset Format:** Excel
- **Version Control:** Git and GitHub

---

## Project Structure

```text
cortexhr/
│
├── data/
│   └── raw/
│       └── 02_CortexHR_AI_ML_Dataset.xlsx
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_attrition_analysis.ipynb
│   ├── 03_attrition_model.ipynb
│   ├── 04_attendance_analysis.ipynb
│   ├── 05_attendance_anomaly_detector.ipynb
│   ├── 06_payroll_outlier_detector.ipynb
│   ├── 07_payroll_flag_explanations.ipynb
│   └── 08_api.ipynb
│
├── 08_api/
│   └── app/
│       └── main.py
│
├── models/
│   ├── attendance_detector.joblib
│   ├── attrition_preprocessor.joblib
│   └── metadata.json
│
├── reports/
│   └── payroll_features_task6.csv
│
├── src/
├── tests/
├── Dockerfile
├── .gitignore
└── README.md
```

*Note: The structure above describes the main project components. The exact filenames and available artifacts may vary according to the files committed to the repository.*

---

## Tasks and Implementation

### Task 1: HR Data Audit

Inspected the provided HR datasets to understand their structure, columns, data types, missing values, duplicate records, and data quality issues.

The audit established a foundation for subsequent machine learning and analytics tasks.

### Task 2: Employee Attrition Analysis

Analyzed employee attrition data to understand the distribution of employees who left and those who stayed.

The target variable contained:

- Employees who stayed: 2,040
- Employees who left: 349
- Total records: 2,389

The observed attrition rate was approximately 14.6%.

### Task 3: Attrition Prediction Model

Developed a baseline Logistic Regression model to estimate whether an employee would leave within 12 months.

The baseline model achieved:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.702 |
| Precision-Recall AUC | 0.293 |
| Recall | 0.596 |
| F1-score | 0.360 |

These results provide a baseline for evaluating future model improvements. The predictions should be treated as decision-support signals rather than definitive judgments about employees.

### Task 4: Attendance Data Analysis

Analyzed monthly attendance features to understand attendance patterns and prepare the data for anomaly detection.

The work focused on identifying unusual attendance behavior and preparing features for an anomaly detection model.

### Task 5: Attendance Anomaly Detection

Implemented an attendance anomaly detection workflow using Isolation Forest and rule-based checks.

The workflow included:

- Monthly attendance feature engineering.
- Statistical anomaly detection.
- Zero-attendance rule checks.
- Comparison against available anomaly labels.
- Evaluation using precision, recall, and F1-score.

The combined detection workflow identified 87 records in the evaluated dataset. Against the available labels, it achieved approximately 39.4% precision, 58% recall, and an F1-score of 0.469.

These results provide a baseline for improving anomaly detection.

### Task 6: Payroll Anomaly Detection

Developed a payroll anomaly detection workflow using engineered payroll features, rule-based checks, and Isolation Forest.

The workflow examined potential irregularities such as:

- Duplicate payments.
- Negative net pay.
- Zero net pay.
- Unusual payroll values.
- Statistical outliers.

The engineered payroll feature dataset contained 7,355 records and 23 initial features, with additional derived features created during processing.

The combined detection workflow flagged 219 records for further review.

These flags indicate records requiring investigation and do not, by themselves, establish that a payroll error or fraud has occurred.

### Task 7: Payroll Flag Explanations and Financial Risk

Extended the payroll anomaly workflow by generating reviewable flag records with reason codes and financial risk information.

The generated output contained 219 flagged records and included information such as result identifiers, reason codes, and scoring timestamps.

The recorded total money at risk was approximately **₹42,382,443.93**.

The recorded reason-code distribution was:

| Reason Code | Flagged Records |
|---|---:|
| Statistical outlier | 176 |
| Double payment | 30 |
| Negative net pay | 8 |
| Zero net pay | 5 |
| **Total** | **219** |

A 20-record review sample was also prepared, containing five records from each of the four recorded reason-code categories.

The purpose of this task was to make the model output more interpretable and support human review of potentially problematic payroll records.

**Important:** The calculated money-at-risk figure represents an investigation estimate from flagged records. It should not be interpreted as confirmed financial loss.

### Task 8: FastAPI Integration

Developed the foundation for serving the HR audit and anomaly detection functionality through a FastAPI application.

The API work included:

- Application and request/response schema development.
- Integration of saved model artifacts.
- Reason-code configuration.
- API documentation and setup instructions.
- Automated test definitions.
- Preparation of endpoints for model-related functionality.

The project also included a test file containing six tests.

API behavior and test results should be confirmed against the current application and execution logs before being reported as fully validated.

### Task 9: HR Chatbot and Intent Classification

Developed an NLP-based HR chatbot workflow using the provided chatbot training data, test data, and HR FAQ dataset.

The implementation included:

- Text preprocessing.
- TF-IDF vectorization.
- Logistic Regression intent classification.
- Evaluation on the supplied test dataset.
- HR FAQ and response-template inspection.
- Intent and response configuration.
- Preparation for integration with the FastAPI `/v1/chat` endpoint.

The training dataset contained 858 examples across 33 intents, with 132 examples in the supplied test set.

The baseline intent classifier achieved **75.76% test accuracy**.

The evaluation identified confusion between some related intents, including `hr_policy_leave` and `working_hours`. This highlights the need for further improvements in training examples, preprocessing, and intent handling.

The chatbot is intended to assist with general HR policy questions. Responses involving employee-specific or restricted information should follow appropriate authentication and authorization controls.

### Task 10: Integration, Testing, and Documentation

Organized the project components for integration, validation, and final documentation.

The planned validation workflow includes:

- Checking model artifact availability.
- Verifying API request and response formats.
- Testing chatbot intent predictions and responses.
- Checking reason-code consistency.
- Running automated tests.
- Reviewing generated outputs.
- Documenting limitations and possible improvements.

Final integration status should be determined from the actual application execution, test output, and endpoint responses.

---

## How to Run the Project

### Prerequisites

Install the following software:

- Python 3.12 or a compatible version.
- Docker Desktop.
- Git.

### Run Using Docker

Build the Docker image from the project directory:

```bash
docker build -t growmar-audit .
```

Start the JupyterLab environment:

```bash
docker run --rm -p 127.0.0.1:8888:8888 -v "${PWD}:/workspace" growmar-audit
```

Open the JupyterLab URL displayed in the terminal.

On Windows PowerShell, `${PWD}` can be used as the current-directory volume path when running the command from the project directory.

### Running the Notebooks

1. Open the project in JupyterLab.
2. Navigate to the relevant notebook.
3. Confirm that the required dataset is available.
4. Run the notebook cells in sequence.
5. Review the generated metrics, tables, and reports.

The dataset path used in the Docker environment is:

```text
/workspace/data/raw/02_CortexHR_AI_ML_Dataset.xlsx
```

### Running the API

The API application is located at:

```text
08_api/app/main.py
```

The exact launch command depends on the module structure and imports in the current implementation.

Once the application is running, FastAPI's interactive documentation is normally available at:

```text
http://127.0.0.1:8000/docs
```

The API must be started successfully before this documentation endpoint can be accessed.

---

## Model Evaluation Summary

| Component | Method | Recorded Result |
|---|---|---|
| Attrition prediction | Logistic Regression | ROC-AUC: 0.702 |
| Attendance anomaly detection | Isolation Forest + rules | F1-score: 0.469 |
| Payroll anomaly detection | Isolation Forest + rules | 219 records flagged |
| Payroll explanations | Reason-code generation | 219 output records |
| HR chatbot | TF-IDF + Logistic Regression | Test accuracy: 75.76% |

The metrics above represent results from the individual workflows. They do not independently establish that the entire application has passed end-to-end testing.

---

## Limitations

- The models depend on the quality and representativeness of the supplied datasets.
- Anomaly detection may generate false positives and requires human review.
- Payroll flags indicate potential irregularities, not confirmed fraud.
- Attrition predictions should not be used as the sole basis for employment decisions.
- Chatbot intent classification can confuse semantically similar questions.
- FAQ-based answers are limited to the available policy content.
- API integration and automated tests must be verified against the current implementation.
- Sensitive employee information must be handled securely, with appropriate access controls.

---

## Future Improvements

- Improve model performance through feature selection and hyperparameter tuning.
- Evaluate anomaly detection using additional labeled examples.
- Improve payroll explanations with clearer evidence and traceable calculations.
- Strengthen API input validation, exception handling, and automated testing.
- Improve chatbot accuracy with more representative training examples.
- Add confidence thresholds and a safe fallback for uncertain chatbot predictions.
- Enforce role-based authorization for employee-specific information.
- Add structured logging, monitoring, and reproducible model evaluation.
- Develop a dashboard for HR analytics and anomaly review.

---

## Conclusion

CortexHR demonstrates the application of machine learning, anomaly detection, NLP, and API development to HR analytics and auditing.

The project covers employee attrition prediction, attendance anomaly detection, payroll flag generation, interpretable review outputs, and an HR chatbot prototype.

The work establishes a foundation for further model evaluation, API validation, and integration into a more complete HR analytics solution.

---

## Disclaimer

This project is intended for educational and analytical purposes. Model predictions and anomaly flags should be reviewed by qualified personnel before any operational action is taken. The system does not independently establish employee misconduct, payroll fraud, or confirmed financial loss.