# Bias Log - {{cookiecutter.project_slug}}

## Overview
This document records bias testing performed on the model as part of EU AI Act compliance.
Complete this document before promoting any model version to production.

**Model:** {{cookiecutter.model_name}}
**LOB:** {{cookiecutter.lob_display_name}}
**Owner:** {{cookiecutter.engineer_group}}

---

## 1. Protected Attributes
List the attributes that were tested for bias:

| Attribute | Type | Included in model? | Notes |
|---|---|---|---|
| Age | Demographic | Yes / No | |
| Gender | Demographic | Yes / No | |
| Nationality | Demographic | Yes / No | |
| <!-- Add more --> | | | |

---

## 2. Bias Testing Results
Document the results of bias testing per protected attribute:

### 2.1 <!-- Attribute name -->
| Metric | Group A | Group B | Difference | Acceptable? |
|---|---|---|---|---|
| Accuracy | | | | Yes / No |
| False Positive Rate | | | | Yes / No |
| False Negative Rate | | | | Yes / No |

**Finding:**
<!-- Describe the finding and any remediation taken -->

---

## 3. Fairness Thresholds
Document the fairness thresholds used for this model:

| Metric | Threshold | Justification |
|---|---|---|
| Max accuracy difference between groups | 5% | |
| Max false positive rate difference | 5% | |
| <!-- Add more --> | | |

---

## 4. Remediation Actions
Document any actions taken to address bias findings:

| Issue | Action taken | Date | Owner |
|---|---|---|---|
| | | | |

---

## 5. Sign-off
| Role | Name | Date | Signature |
|---|---|---|---|
| Model owner | | | |
| Data scientist | | | |
| Compliance officer | | | |