# EU AI Act Risk Classification - {{cookiecutter.project_slug}}

## Overview
This document records the EU AI Act risk classification for this AI system.
Complete this document before deploying any model to production.

**Model:** {{cookiecutter.model_name}}
**LOB:** {{cookiecutter.lob_display_name}}
**Owner:** {{cookiecutter.engineer_group}}
**Classification date:** YYYY-MM-DD
**Next review date:** YYYY-MM-DD

---

## 1. Risk Tier Classification

| Risk Tier | Description | Examples |
|---|---|---|
| **Unacceptable** | Prohibited AI practices | Social scoring, real-time biometric surveillance |
| **High** | Significant risk to health, safety, or fundamental rights | HR decisions, credit scoring, safety components |
| **Limited** | Specific transparency obligations | Chatbots, emotion recognition |
| **Minimal** | No specific obligations | Spam filters, recommendation systems |

**This model's risk tier:** <!-- Minimal / Limited / High / Unacceptable -->

**Justification:**
<!-- Explain why this risk tier was selected -->

---

## 2. High-Risk Checklist
*Complete this section only if the model is classified as High-Risk.*

### 2.1 Risk Management
- [ ] Risk management system established and documented
- [ ] Risks identified and evaluated throughout lifecycle
- [ ] Residual risks acceptable

### 2.2 Data Governance
- [ ] Training data relevant, representative, and free of errors
- [ ] Data governance practices documented
- [ ] GDPR compliance verified for all data sources

### 2.3 Technical Documentation
- [ ] Model card completed (`model_card.md`)
- [ ] Bias log completed (`bias_log.md`)
- [ ] System architecture documented
- [ ] Training and validation methodology documented

### 2.4 Transparency
- [ ] Logging and traceability enabled
- [ ] Instructions for use documented
- [ ] Limitations clearly communicated to end users

### 2.5 Human Oversight
- [ ] Human-in-the-loop controls defined
- [ ] Override mechanism available to end users
- [ ] Responsible person identified

### 2.6 Accuracy and Robustness
- [ ] Performance metrics meet defined thresholds
- [ ] Model tested for robustness against errors
- [ ] Bias testing completed and documented

---

## 3. Compliance Responsibilities

| Responsibility | Owner |
|---|---|
| Technical documentation | {{cookiecutter.engineer_group}} |
| Bias testing | {{cookiecutter.engineer_group}} |
| Human oversight design | LOB Business Owner |
| Legal/regulatory sign-off | Compliance Officer |
| Platform infrastructure | Hub Platform Team |

---

## 4. Review History
| Date | Reviewer | Outcome | Notes |
|---|---|---|---|
| YYYY-MM-DD | | Approved / Rejected | |