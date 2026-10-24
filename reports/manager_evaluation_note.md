# CortexHR Attrition Model — Evaluation Note

## Purpose
The CortexHR model estimates employee attrition risk using available employee features. It helps HR prioritize follow-up and support, not make automatic employment decisions.

## Model performance
Logistic Regression achieved a ROC-AUC of 0.702, PR-AUC of 0.293, and accuracy of 0.691. ROC-AUC measures ranking ability, while PR-AUC summarizes precision and recall for the leaver class. Accuracy alone does not show how effectively the model identifies employees who leave.

## Risk-band findings
The High band contains 239 records, including 85 actual leavers. Its leaver rate is 35.6%, and it captures 24.4% of all actual leavers. The Medium band contains 478 records and 112 actual leavers. The Low band contains 1,672 records and 152 actual leavers. The top 10% captures approximately 2.44 times the share of leavers expected from random selection of 10% of records.

## Understanding the scores
The original calibration curve indicates that the model tends to assign higher risks than observed departure rates justify. Its scores should not be treated as reliable individual probabilities. Sigmoid calibration improved the Brier score in our experiment, but requires further validation.

## Recommended use and limitations
HR can use the ranked list to prioritize confidential, supportive check-ins and investigate workplace issues. A high score does not guarantee departure, and a low score does not guarantee retention. HR must review each case in context, protect privacy, assess fairness, and never rely on a score alone for employment decisions. Risk-band counts must be reconciled to unique employees before producing an employee-level action list.

## Conclusion
The baseline is a starting point for prioritizing reviews, not a decision-making system ready for operational use. Calibration and generalization require further validation.
