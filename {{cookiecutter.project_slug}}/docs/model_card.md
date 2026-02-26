# Model Card - {{cookiecutter.project_slug}}

## Model Overview
| Field | Value |
|---|---|
| **Model name** | {{cookiecutter.model_name}} |
| **Project** | {{cookiecutter.project_slug}} |
| **LOB** | {{cookiecutter.lob_display_name}} |
| **Owner** | {{cookiecutter.engineer_group}} |
| **Created** | YYYY-MM-DD |
| **Last updated** | YYYY-MM-DD |
| **Model type** | Classification / Time Series / Clustering |
| **Framework** | scikit-learn / Prophet / other |

## Business Purpose
<!-- Describe the business problem this model solves -->

## Input Data
| Feature | Description | Type | Source |
|---|---|---|---|
| feature_1 | Description | Float | Table name |
| feature_2 | Description | Float | Table name |

## Output
| Output | Description | Type |
|---|---|---|
| prediction | Description | Float / Integer |

## Training Data
| Field | Value |
|---|---|
| **Catalog** | {{cookiecutter.catalog_name}} |
| **Schema** | {{cookiecutter.schema_name}} |
| **Table** | feature_table |
| **Date range** | YYYY-MM-DD to YYYY-MM-DD |
| **Row count** | |

## Model Performance
| Metric | Value | Threshold |
|---|---|---|
| Accuracy | | |
| F1 Score | | |
| ROC AUC | | |

## Limitations and Risks
<!-- Describe known limitations, edge cases, and risks -->

## EU AI Act Risk Classification
See `risk_classification.md` for full risk assessment.

| Field | Value |
|---|---|
| **Risk tier** | Minimal / Limited / High / Unacceptable |
| **Justification** | |

## Monitoring
| Field | Value |
|---|---|
| **Monitoring type** | Databricks Lakehouse Monitoring (Standard) |
| **Feature drift** | Enabled |
| **Prediction drift** | Enabled |
| **Ground truth tracking** | Out of scope - project team responsibility |

## Changelog
| Date | Version | Change | Author |
|---|---|---|---|
| YYYY-MM-DD | 1.0 | Initial version | |