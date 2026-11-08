# CortexHR — AI/ML-Powered HR Analytics and Data Auditing

**Author:** Eniyan J  
**Track:** AI/ML Engineering  
**Project:** CortexHR  
**Phases:** Phase 1, Phase 2 and Phase 3  
**GitHub Repository:** [https://github.com/Eniyan-j/cortexhr](https://github.com/Eniyan-j/cortexhr)

---

## Project Overview

CortexHR is an AI/ML-powered HR analytics and data auditing project designed to demonstrate how machine learning, anomaly detection, natural language processing, explainable AI, fairness auditing and API development can support Human Resources operations.

The project covers employee attrition prediction, attendance anomaly detection, payroll irregularity identification, resume ranking, fairness evaluation, explainable attrition risk reporting and fair compensation recommendations.

The implementation is organized into three development phases, covering 15 tasks. Each phase builds on the previous work, progressing from foundational data auditing and baseline modeling to integrated ML workflows, explainability and fairness-aware decision support.

The project emphasizes:

- Data quality and validation.
- Reproducible feature engineering.
- Prevention of target leakage.
- Model evaluation and probability calibration.
- Explainable machine learning.
- Fairness and protected-attribute analysis.
- Human oversight in HR-related decisions.
- Reusable code, model artifacts and documented evidence.

---

## Project Objectives

1. Audit and understand structured HR datasets.
2. Develop reusable employee attrition feature-engineering pipelines.
3. Build and evaluate employee attrition prediction models.
4. Detect unusual attendance and payroll behavior.
5. Generate explanations for detected payroll irregularities.
6. Develop REST APIs to expose ML functionality.
7. Build an HR chatbot using NLP and intent classification.
8. Implement resume ranking and fairness auditing.
9. Improve attrition model performance and calibration.
10. Generate explainable employee attrition risk reports.
11. Develop fair compensation prediction and pay-band recommendations.
12. Evaluate potential bias and introduce appropriate decision-support guardrails.

---

# Phase 1 — Data Audit, Feature Engineering and Baseline Modeling

**Git Tag:** `phase-1`

Phase 1 establishes the foundational data-processing and machine-learning workflows for the CortexHR project.

The phase focuses on understanding the HR datasets, developing reusable feature-engineering logic, building baseline attrition models, evaluating probability predictions and detecting attendance anomalies.

## Tasks and Implementation

### Task 1: HR Data Audit and Exploration

**Notebook:** `phase-1/01_data_audit_exploration.ipynb`

Performed exploratory data analysis and data-quality validation on the provided HR datasets.

Key activities:

- Dataset structure and dimensionality inspection.
- Column and data-type validation.
- Missing-value analysis.
- Duplicate-record identification.
- Invalid-value and outlier inspection.
- Target distribution analysis.
- Preliminary target-leakage investigation.

The task established an understanding of the available data and identified potential quality issues before model development.

### Task 2: Feature Engineering and Reusable Pipeline

**Notebook:** `phase-1/02_feature_pipeline.ipynb`

**Reusable module:** `src/features.py`

Developed a reusable feature-engineering workflow to transform raw employee data into structured model-ready features.

Key activities:

- Employee attrition feature generation.
- Data cleaning and transformation.
- Feature aggregation.
- Training and inference feature consistency.
- Pipeline construction and validation.
- Prevention of target leakage.

The reusable pipeline provides a consistent foundation for subsequent attrition modeling tasks.

### Task 3: Baseline Attrition Prediction Model

**Notebook:** `phase-1/03_baseline.ipynb`

Developed baseline machine-learning models to estimate whether an employee would leave within a 12-month observation horizon.

The processed attrition dataset contained:

| Metric | Value |
|---|---:|
| Total observations | 2,389 |
| Employees who stayed | 2,040 |
| Employees who left | 349 |
| Attrition rate | 14.61% |

A Logistic Regression baseline was developed to provide an initial performance reference.

Recorded results:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.702 |
| Precision-Recall AUC | 0.293 |
| Recall | 0.596 |
| F1-score | 0.360 |

These results established a baseline for subsequent model improvements.

### Task 4: Model Evaluation and Probability Calibration

**Notebook:** `phase-1/04_evaluation.ipynb`

Evaluated the baseline model to understand its predictive performance, classification behavior and probability quality.

Key activities:

- Classification performance evaluation.
- ROC-AUC analysis.
- Precision-Recall evaluation.
- Threshold-based performance analysis.
- Probability calibration.
- Model comparison and evaluation reporting.

The objective was to assess not only how well the model distinguished between employees who left and stayed, but also how reliably its predicted probabilities represented attrition risk.

### Task 5: Attendance Anomaly Detection

**Notebook:** `phase-1/05_attendance_anomaly.ipynb`

Developed an attendance anomaly detection workflow combining statistical and machine-learning techniques.

Methods included:

- Attendance feature engineering.
- Isolation Forest anomaly detection.
- Rule-based anomaly identification.
- Zero-attendance checks.
- Comparison against available anomaly labels.
- Precision, recall and F1-score evaluation.

Recorded combined detection results:

| Metric | Result |
|---|---:|
| Flagged records | 87 |
| Precision | 39.4% |
| Recall | 58% |
| F1-score | 0.469 |

The detected anomalies represent records requiring additional investigation rather than confirmed employee misconduct.

---

# Phase 2 — Payroll Analytics, APIs and HR Chatbot

**Git Tag:** `phase-2`

Phase 2 expands the project beyond the initial attrition and attendance workflows.

It introduces payroll anomaly detection, explainable payroll flag generation, API development and an NLP-based HR chatbot.

## Tasks and Implementation

### Task 6: Payroll Anomaly Detection

**Notebook:** `phase-2/06_payroll_outlier_detector.ipynb`

Developed a payroll anomaly detection workflow to identify unusual payroll records and potential financial irregularities.

The implementation included:

- Payroll data inspection.
- Payroll feature engineering.
- Duplicate payment investigation.
- Negative net-pay checks.
- Zero net-pay checks.
- Statistical outlier detection.
- Isolation Forest modeling.
- Rule-based detection.

The engineered payroll dataset contained 7,355 records and 23 initial features, with additional derived features created during processing.

The combined workflow flagged 219 payroll records for further review.

These flags indicate potential irregularities and do not independently establish payroll fraud or confirmed financial loss.

### Task 7: Payroll Flag Explanations and Financial Risk

**Notebook:** `phase-2/07_payroll_flag_explanations.ipynb`

Extended the payroll anomaly detection workflow by generating interpretable explanations for flagged records.

The implementation included:

- Reason-code generation.
- Flag classification.
- Payroll risk investigation.
- Financial exposure estimation.
- Reviewable output generation.
- Sample record inspection.

Recorded output:

| Reason Code | Flagged Records |
|---|---:|
| Statistical outlier | 176 |
| Double payment | 30 |
| Negative net pay | 8 |
| Zero net pay | 5 |
| **Total** | **219** |

The recorded estimated money at risk was approximately ₹42,382,443.93.

A 20-record review sample was also prepared, containing five records from each of the four reason-code categories.

The estimated money-at-risk figure represents an investigation estimate from flagged records, not confirmed financial loss.

### Task 8: FastAPI Integration and Anomaly API

**Notebook:** `phase-2/08_api.ipynb`

**Application:** `08_api/app/main.py`

Developed a FastAPI application to expose HR anomaly detection functionality through API endpoints.

Key implementation areas:

- FastAPI application setup.
- Request and response schema definitions.
- Integration of saved ML artifacts.
- Attendance and payroll model integration.
- Reason-code configuration.
- Health-check endpoint.
- API testing and endpoint documentation.

The health endpoint returned:

```json
{
  "status": "healthy",
  "loaded_artifacts": [
    "attendance",
    "payroll",
    "reason_codes"
  ]
}
```

The API provides an interface for accessing the project's anomaly detection functionality.

### Task 9: HR Chatbot Intent Classification

**Notebook:** `phase-2/09_chatbot_intent_classifier.ipynb`

Developed an NLP-based HR chatbot intent classification workflow.

The chatbot uses text preprocessing, TF-IDF vectorization and Logistic Regression to identify employee HR queries.

Implementation included:

- Text preprocessing.
- TF-IDF feature extraction.
- Logistic Regression classification.
- Intent prediction.
- Model evaluation.
- HR FAQ and response-template integration.

Dataset details:

| Metric | Value |
|---|---:|
| Training examples | 858 |
| Number of intents | 33 |
| Test examples | 132 |
| Test accuracy | 75.76% |

The evaluation identified confusion between semantically related intents, including `hr_policy_leave` and `working_hours`.

The results provide a baseline for further improvements in chatbot accuracy and response handling.

### Task 10: Chatbot Memory, Permissions and Service Integration

**Notebook:** `phase-2/10_chatbot_memory_premission.ipynb`

Extended the HR chatbot workflow towards a service-integrated application with session handling and role-based access concepts.

Key implementation areas:

- Mock HR service integration.
- Employee and HR role handling.
- Session management.
- Conversation context handling.
- Chat endpoint integration.
- Role-aware response handling.
- API interaction and testing.

The mock service included employee, HR and administrator role examples.

The project also integrated chatbot functionality with the `/v1/chat` API workflow.

The purpose was to move beyond standalone intent classification and demonstrate how an HR chatbot can interact with supporting application services.

Employee-specific and restricted information should remain protected by appropriate authentication and authorization controls.

---

# Phase 3 — Resume Intelligence, Advanced Attrition Modeling and Fair Compensation

**Git Tag:** `phase-3`

Phase 3 introduces advanced machine-learning workflows focused on resume ranking, fairness auditing, improved attrition prediction, explainable risk reporting and fair compensation recommendations.

This phase extends CortexHR from foundational HR analytics towards more advanced and explainable decision-support workflows.

## Tasks and Implementation

### Task 11: Resume Ranking and Candidate Recommendation

**Notebook:** `phase-3/11_rank_resume.ipynb`

Developed a resume-ranking workflow to evaluate candidate profiles against available job requirements.

The implementation included:

- Resume dataset inspection.
- Job requirement analysis.
- Resume feature processing.
- Candidate and job matching.
- Resume ranking.
- Ranking evaluation.
- Top-k recommendation analysis.

Dataset details:

| Metric | Value |
|---|---:|
| Resumes | 185 |
| Job descriptions | 10 |
| Candidate-job labels | 555 |
| Shortlisted labels | 95 |

The evaluation included job-wise ranking performance and top-k candidate relevance checks.

The workflow demonstrates how structured candidate information can support recruitment screening.

The ranking results are intended to assist human recruitment review and should not be used as the sole basis for hiring decisions.

### Task 12: Resume Ranking Fairness Audit

**Notebook:** `phase-3/12_resume_fairness_audit.ipynb`

Developed a fairness-audit workflow to examine potential disparities in resume-ranking outcomes.

Key activities:

- Protected-attribute inspection.
- Resume ranking outcome analysis.
- Group-based fairness evaluation.
- Comparison of selection and ranking outcomes.
- Fairness report generation.
- Identification of potential sources of bias.

Generated fairness analysis artifacts are stored under:

```text
reports/task12/
```

The objective was to investigate whether candidate-ranking behavior could produce unequal outcomes across demographic groups.

Fairness metrics are diagnostic tools and should be interpreted alongside dataset quality, group representation, job relevance and the limitations of the available labels.

### Task 13: Advanced Attrition Prediction — V2 Model

**Notebook:** `phase-3/13_beat_baseline_attrV2model.ipynb`

Developed an improved employee attrition prediction model to outperform the initial baseline workflow.

The implementation introduced an XGBoost-based model with sigmoid probability calibration.

Dataset details:

| Metric | Value |
|---|---:|
| Observations | 2,389 |
| Model input features | 61 |
| Employees who left | 349 |
| Employees who stayed | 2,040 |
| Attrition rate | 14.61% |

Model evaluation:

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Original V1 baseline | 0.7018 | 0.2932 | 0.2063 |
| V2 raw model | 0.7223 | 0.3471 | — |
| V2 sigmoid-calibrated model | 0.7230 | 0.3521 | 0.1120 |

Additional calibrated model result:

| Metric | Result |
|---|---:|
| Expected Calibration Error (ECE) | 0.0150 |

The V2 model improved ranking performance compared with the original baseline and produced better-calibrated probability estimates.

The resulting model artifact and its metadata are stored in:

```text
models/
├── attrition-v2.joblib
└── attrition-v2_metadata.json
```

The model is designed to provide 12-month attrition-risk estimates for HR decision support.

### Task 14: Explainable Attrition Risk Reports

**Notebook:** `phase-3/14_explainable_risk_reports.ipynb`

Developed an explainable attrition risk-reporting workflow using the calibrated V2 attrition model and SHAP-based feature explanations.

A major part of this task involved reconstructing historical attendance and payroll features required by the V2 model but unavailable directly in the original scoring dataset.

Key implementation areas:

- Historical attendance data validation.
- Historical payroll data validation.
- Duplicate employee-month investigation.
- Original training feature inspection.
- Reuse of existing feature-engineering logic.
- Missing feature reconstruction.
- V2 model integration.
- SHAP explanation generation.
- Employee-level risk categorization.
- Final report schema validation.

The scoring population contained 430 employees.

The risk-report output was designed with 11 columns:

| Column | Description |
|---|---|
| prediction_id | Unique risk prediction identifier |
| employee_id | Employee reference |
| department_id | Department reference |
| as_of_date | Prediction observation date |
| risk_score_12m | Calibrated 12-month attrition-risk estimate |
| risk_band | Assigned risk category |
| top_reason_1 | First SHAP-based model influence |
| top_reason_2 | Second SHAP-based model influence |
| top_reason_3 | Third SHAP-based model influence |
| model_version | Model version |
| model_note | Prediction and usage information |

Recorded risk-band distribution:

| Risk Band | Employees |
|---|---:|
| High | 43 |
| Medium | 86 |
| Low | 301 |
| **Total** | **430** |

The high-risk cutoff was approximately `0.212859`, and the medium-risk cutoff was approximately `0.136231`.

Frequently identified model-influencing features included:

- Last engagement score.
- Months since last raise.
- Manager changes in the previous 12 months.
- Work-from-home share.
- Employee tenure.

The SHAP explanations indicate features influencing the model's predictions. They do not establish the actual causes of employee attrition.

The resulting report is intended to support structured HR review, not automate employment decisions.

### Task 15: Fair Pay Bands and Career-Gap Fairness

**Notebook:** `phase-3/15_fair_pay_bands.ipynb`

Developed a compensation prediction and recommendation workflow with fairness considerations.

The objective was to estimate reasonable salary ranges using job-related attributes while evaluating potential disparities associated with protected or sensitive attributes.

Dataset details:

| Metric | Value |
|---|---:|
| Salary training records | 2,934 |
| Male records | 1,652 |
| Female records | 1,222 |
| Other gender records | 60 |
| Career-gap records | 385 |
| Records without a career gap | 2,549 |

The implementation included:

- Salary dataset inspection.
- Protected-attribute integration.
- Feature selection.
- Salary prediction model development.
- Baseline and advanced model comparison.
- Gender-based error analysis.
- Career-gap fairness investigation.
- Role-specific salary guardrails.
- Pay-band recommendation generation.

Protected attributes such as gender and age, along with career-gap information, were excluded from the salary prediction feature set. They were retained for fairness analysis rather than direct model prediction.

Recorded model evaluation:

| Model | MAE | R² |
|---|---:|---:|
| Baseline | 115,094 | 0.93 |
| Linear Regression | 81,830 | 0.97 |
| Gradient Boosting | 108,465 | 0.94 |

Fairness analysis included gender-based error comparisons and career-gap group comparisons.

The workflow also introduced role-specific minimum and maximum compensation guardrails to keep recommendations within defined job-level ranges.

The final output includes recommended low, middle and high pay bands, along with guardrail-adjustment information.

These recommendations are intended to support compensation analysis and human review rather than automatically determine employee salaries.

---

# Repository Structure

The CortexHR project is maintained in one consolidated GitHub repository, with separate folders for each development phase.

```text
cortexhr/
│
├── README.md
│
├── phase-1/
│   ├── 01_data_audit_exploration.ipynb
│   ├── 02_feature_pipeline.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_attendance_anomaly.ipynb
│
├── phase-2/
│   ├── 06_payroll_outlier_detector.ipynb
│   ├── 07_payroll_flag_explanations.ipynb
│   ├── 08_api.ipynb
│   ├── 09_chatbot_intent_classifier.ipynb
│   └── 10_chatbot_memory_premission.ipynb
│
├── phase-3/
│   ├── 11_rank_resume.ipynb
│   ├── 12_resume_fairness_audit.ipynb
│   ├── 13_beat_baseline_attrV2model.ipynb
│   ├── 14_explainable_risk_reports.ipynb
│   └── 15_fair_pay_bands.ipynb
│
├── 05_Starter/
│
├── 08_api/
│   └── app/
│       ├── main.py
│       └── schemas.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── task11/
│   ├── task12/
│   ├── task14/
│   └── task15/
│
├── src/
│   └── features.py
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The structure describes the principal project components. Additional model files, reports, configuration files and generated artifacts may be present depending on the task and execution environment.

The `data/` directory may contain local source and processed datasets. Confidential employee information, restricted source datasets and answer-key files should not be committed to the public repository.

Generated evidence and task-specific reports are organized under `reports/` where applicable.

---

# Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Natural Language Processing | TF-IDF, Logistic Regression |
| API Development | FastAPI, Pydantic |
| Data Storage | CSV, Excel |
| Model Persistence | Joblib |
| Development Environment | Jupyter Notebook, JupyterLab |
| Containerization | Docker |
| Version Control | Git, GitHub |

---

# Environment Setup

Docker is used to provide a reproducible development environment.

## Prerequisites

Install:

- Git.
- Docker Desktop.
- Python 3.12 or a compatible version, if running notebooks outside Docker.
- JupyterLab, if running notebooks locally.

## Clone the Repository

```bash
git clone https://github.com/Eniyan-j/cortexhr.git
cd cortexhr
```

## Dataset Setup

Place the authorized input dataset in:

```text
data/raw/02_CortexHR_AI_ML_Dataset.xlsx
```

The dataset is not guaranteed to be included in the public repository.

Obtain restricted or confidential datasets only through the authorized project channel.

## Build the Docker Image

From the repository root:

```bash
docker build -t cortexhr .
```

## Start JupyterLab

**Windows PowerShell:**

```powershell
docker run --rm -p 127.0.0.1:8888:8888 -v "${PWD}:/workspace" cortexhr
```

**Linux or macOS:**

```bash
docker run --rm -p 127.0.0.1:8888:8888 -v "$(pwd):/workspace" cortexhr
```

Open the JupyterLab URL printed in the terminal.

The repository is mounted at:

```text
/workspace
```

Start JupyterLab from the repository root so that notebooks can locate the shared `data/`, `models/`, `reports/` and `src/` directories.

---

# Running the Project

Run the notebooks in numerical order, beginning with Phase 1 and progressing through Phase 3.

```text
Phase 1
  01 → 02 → 03 → 04 → 05

Phase 2
  06 → 07 → 08 → 09 → 10

Phase 3
  11 → 12 → 13 → 14 → 15
```

Each phase builds on the outputs and supporting functionality established by earlier work.

Before executing a notebook:

1. Confirm the required dataset is available.
2. Confirm the relevant model artifacts exist, where required.
3. Start JupyterLab from the repository root.
4. Check the notebook's expected input paths.
5. Execute the cells in the intended order.
6. Review the outputs, metrics and saved artifacts.

Some tasks depend on artifacts generated by previous notebooks. Running an individual notebook may therefore require the corresponding earlier outputs.

---

# Running the FastAPI Application

The API application is located at:

```text
08_api/app/main.py
```

The application uses FastAPI and Pydantic schemas.

From the repository root, the following command can be used when the application dependencies and import structure are configured accordingly:

```bash
python -m uvicorn app.main:app --app-dir 08_api --host 127.0.0.1 --port 8000 --reload
```

After starting the application, access the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The API must start successfully before its documentation or endpoints can be accessed.

---

# Model Evaluation Summary

The following table summarizes selected recorded results across the 15 tasks.

| Component | Method | Recorded Result |
|---|---|---|
| Baseline attrition prediction | Logistic Regression | ROC-AUC: 0.702 |
| Attendance anomaly detection | Isolation Forest + rules | F1-score: 0.469 |
| Payroll anomaly detection | Isolation Forest + rules | 219 records flagged |
| Payroll explanations | Reason-code generation | 219 flagged records |
| HR chatbot | TF-IDF + Logistic Regression | 75.76% accuracy |
| Resume ranking | Job-specific ranking | 185 resumes, 10 jobs |
| Advanced attrition prediction | Calibrated XGBoost | ROC-AUC: 0.7230 |
| Advanced attrition prediction | Calibrated XGBoost | PR-AUC: 0.3521 |
| Advanced attrition prediction | Calibrated XGBoost | Brier score: 0.1120 |
| Explainable attrition reporting | SHAP + calibrated V2 | 430 employee reports |
| Fair compensation prediction | Linear Regression | MAE: 81,830; R²: 0.97 |

These are results from individual task workflows. They should not be interpreted as proof that every component has passed complete end-to-end integration testing.

---

# Git Versioning

The project uses Git tags to identify completed development phases.

| Phase | Git Tag | Tasks |
|---|---|---|
| Phase 1 | `phase-1` | 1–5 |
| Phase 2 | `phase-2` | 6–10 |
| Phase 3 | `phase-3` | 11–15 |

The GitHub repository maintains the full project history in one consolidated repository.

Repository:

[https://github.com/Eniyan-j/cortexhr](https://github.com/Eniyan-j/cortexhr)

---

# Limitations and Responsible Use

CortexHR is an educational and analytical project. The models and workflows have limitations that must be considered before operational use.

- Model performance depends on the quality and representativeness of the available datasets.
- Anomaly detection may generate false positives.
- Payroll flags indicate potential irregularities and require investigation.
- Estimated financial exposure is not equivalent to confirmed financial loss.
- Attrition predictions are probabilistic estimates, not definitive judgments about employees.
- SHAP explanations describe model influences, not proven causal factors.
- Resume ranking may reflect limitations or bias in historical data and labels.
- Fairness metrics depend on sample sizes, group representation and the selected evaluation methodology.
- Salary recommendations should be reviewed in relation to actual responsibilities, job levels and market conditions.
- HR chatbot responses require appropriate authorization and access controls.
- Sensitive employee information must be protected through appropriate data-handling practices.

No model output should be used as the sole basis for hiring, promotion, termination, compensation or other consequential employment decisions.

---

# Future Improvements

Potential future development areas include:

- Improving attrition prediction through additional feature engineering and model evaluation.
- Monitoring model drift and calibration over time.
- Improving anomaly detection using additional verified labels.
- Enhancing payroll explanation traceability.
- Expanding automated API testing and exception handling.
- Improving chatbot intent recognition and fallback handling.
- Strengthening authentication and role-based authorization.
- Improving resume-ranking evaluation and fairness monitoring.
- Introducing additional pay-equity measures and compensation guardrails.
- Developing interactive HR analytics dashboards.
- Introducing structured logging and model monitoring.
- Establishing reproducible model-training and deployment workflows.

---

# Conclusion

CortexHR demonstrates the application of machine learning and AI engineering concepts to practical HR analytics and data auditing problems.

Across three development phases and 15 tasks, the project progresses from data exploration and feature engineering to anomaly detection, API development, NLP, resume ranking, advanced attrition modeling, explainable AI and fairness-aware compensation recommendations.

The project provides a foundation for building more reliable, interpretable and responsible HR decision-support systems.

---

## Disclaimer

CortexHR is intended for educational, analytical and demonstration purposes.

The model predictions, anomaly flags, resume rankings and salary recommendations are decision-support outputs. They require human review and must not be treated as definitive evidence of employee behavior, misconduct, financial loss or candidate suitability.

The project does not independently establish causality, guarantee fairness or replace qualified HR, legal and financial review.
