# PPMI Prediction Definition

## Overview

This document defines the prediction framework for the PPMI risk prediction project.

The first prediction experiment aims to predict the time to Parkinson's disease (PD) conversion using baseline clinical information only.

The framework is designed to support future extensions including:

- alternative PD conversion definitions
- competing outcomes (MSA, DLB, death)
- multi-state models
- longitudinal prediction models
- multimodal prediction models

The prediction pipeline is therefore configuration-driven and separates:

1. population definition
2. feature extraction
3. outcome definition
4. modelling strategy


---

# 1. Prediction Question

## Primary research question

Can baseline clinical characteristics predict the time until conversion to Parkinson's disease in individuals at risk of developing PD?

The prediction task is formulated as a survival analysis problem:

\[
X_{baseline} \rightarrow (T_{PD}, \delta_{PD})
\]

where:

- \(X_{baseline}\) represents baseline clinical features
- \(T_{PD}\) represents time from baseline to PD conversion or censoring
- \(\delta_{PD}\) indicates whether PD conversion occurred


---

# 2. Prediction Population

## Dataset

Populatio'n:

PPMI risk cohort

The prediction population is derived from the previously generated:
- ppmi_risk_cohort_baseline.csv
- ppmi_risk_cohort_longitudinal.csv


The cohort definition is handled independently by:
- PPMICohortBuilder


## Inclusion criteria

Participants must:

- belong to the PPMI risk cohort
- have a baseline visit
- have sufficient follow-up according to cohort inclusion criteria


## Baseline requirement

The prediction start point is: EVENT_ID == BL


Every participant should have exactly one baseline visit.

This should be validated during dataset construction.


---

# 3. Index Date

The index date is defined as the baseline visit.
- index_event = BL


All predictors must originate from information available at this time point.


---

# 4. Outcome Definition

## Primary outcome

The primary outcome is: PD conversion, defined as the occurrence of a Parkinson's disease diagnosis.

PPMI diagnosis encoding:
- PRIMDIAG == 1
corresponding to idiopathic Parkinson's disease



## Outcome time

Event time is defined as:
- time = first PD diagnosis date - baseline date


The first PD diagnosis date is determined according to the selected PD conversion definition.


---

# 5. PD Conversion Definitions

PD conversion is configurable because different definitions may be appropriate depending on the scientific question.

Possible definitions include:


## First PD diagnosis

A participant is considered converted when:
- PRIMDIAG == 1

occurs for the first time.

Advantages:

- sensitive
- captures earliest diagnosis

Limitations:

- susceptible to transient diagnosis changes


## Persistent PD

A participant is considered converted when PD diagnosis is sustained.

Example definition:

- first PD diagnosis followed by continued PD diagnosis
- or PD diagnosis occurring on multiple visits

Advantages:

- higher specificity

Limitations:

- delays event time


## PD at end of follow-up

A participant is considered converted if the final available diagnosis is PD.

Advantages:

- reflects final clinical status

Limitations:

- ignores earlier diagnostic uncertainty


The selected definition is controlled through configuration.


---

# 6. Censoring

Participants without PD conversion are censored.

The censoring date is determined by:

1. last available follow-up visit

or

2. death date, if death occurs before PD conversion


For survival analysis:
- event = 1 indicates PD conversion.

- event = 0 indicates censored follow-up.

---

# 7. Predictor Definition

## First experiment

The first model uses: baseline clinical data only

No information after baseline may be used.


## Allowed features

Examples:

- demographics
- clinical assessments
- motor scores
- cognitive scores
- non-motor symptoms
- baseline laboratory/clinical variables


## Excluded information

The following are not allowed:

- post-baseline visits
- future diagnosis information
- longitudinal trajectories
- outcomes derived from follow-up

---

# 8. Modelling Framework

## Primary analysis

The first analysis will use survival models.

Possible models:

- Cox proportional hazards model
- penalised Cox regression
- survival random forest
- gradient boosting survival models


Model input: baseline features

Model output: risk of PD conversion over time



---

# 9. Future Extensions

The framework should support:


## Competing outcomes

Examples:

- MSA
- DLB
- death


Potential approaches:

- competing risk survival models
- cause-specific hazard models


## Multi-state modelling

Possible states:

At risk
|
v
PD conversion
|
v
Advanced disease / death



## Longitudinal modelling

Future models may incorporate:

- repeated clinical visits
- disease progression trajectories
- time-varying covariates


## Multimodal modelling

Future predictors may include:

- clinical
- imaging
- biomarkers
- genetics


---

# 10. Configuration

Prediction experiments should be defined through configuration files.

Example: ´configs/prediction/ppmi.yaml´


The configuration should specify:

- prediction population
- index event
- outcome definition
- censoring rules
- feature modalities
- modelling strategy


---

# Summary

The first prediction experiment is:

**Baseline clinical survival prediction of Parkinson's disease conversion in the PPMI risk cohort.**

The prediction framework is designed to allow future extension to alternative outcomes, longitudinal data, multimodal data, and more complex survival modelling approaches.