# Outcome Definition

## Background

The primary outcome of this project is clinical conversion to Parkinson's disease (PD)
among participants enrolled in the PPMI at-risk cohorts.

Unlike many prediction problems, the diagnosis recorded during longitudinal follow-up
is not always stable. Some participants receive a PD diagnosis that is subsequently
revised, while others receive competing diagnoses such as multiple system atrophy (MSA)
or dementia with Lewy bodies (DLB).

Consequently, defining PD conversion requires explicit rules.

This document describes the proposed outcome definition before implementation.

---

# Cohort

The outcome is defined only for participants included in the PPMI at-risk cohort.

Participants satisfy:

- Baseline visit (`EVENT_ID == "BL"`)
- Baseline diagnosis (`PRIMDIAG`) in:
    - 17 (No PD nor other neurological disorder)
    - 23 (Prodromal non-motor PD)
    - 24 (Prodromal motor PD)
    - 25 (Prodromal synucleinopathy)
- At least two visits (baseline + follow-up)

---

# Diagnosis Codes

The project stores diagnosis codes exactly as provided by PPMI.

Important diagnoses are

| Code | Diagnosis |
|------:|-----------|
|1|Idiopathic Parkinson's disease|
|5|Dementia with Lewy bodies (DLB)|
|11|Multiple system atrophy (MSA)|
|17|No PD nor other neurological disorder|
|23|Prodromal non-motor PD|
|24|Prodromal motor PD|
|25|Prodromal synucleinopathy|
|97|Other neurological disorder|

---

# Visit Ordering

Diagnosis trajectories are ordered using the PPMI visit identifiers
(`EVENT_ID`) rather than the visit date.

For example

BL

↓

V01

↓

V02

↓

...

This ensures a consistent visit sequence independent of date formatting.

---

# Diagnosis Trajectories

Several diagnosis patterns are observed in the PPMI data.

## 1. Stable PD conversion

Example

23 → 23 → 24 → 17 → 25 → 1 → 1 → 1

Interpretation

Participant converts to PD and remains diagnosed with PD.

Proposed classification

**Confirmed PD conversion**

---

## 2. Temporary diagnostic reversal

Example

24 → 1 → 1 → 1 → 23 → 1 → 97 → 1 → 1

Interpretation

PD diagnosis is temporarily revised before returning to PD.

Possible explanations include

- diagnostic uncertainty
- clinical reassessment
- coding errors

These participants require further investigation.

---

## 3. Transient PD diagnosis

Example

17 → 17 → 1 → 17 → 17 → 17

Interpretation

A PD diagnosis is assigned once but subsequently removed.

These participants should not automatically be considered confirmed PD converters.

---

## 4. Competing neurodegenerative diagnosis

Example

23 → 24 → 1 → 5 → 11 → 11

Interpretation

Participant progresses to another neurodegenerative disorder.

Potential competing diagnoses include

- MSA
- DLB

These participants may either

- be analysed as competing risks, or
- be treated as censored in the primary analysis.

This decision remains to be finalised.

---

## 5. Stable PD followed by diagnostic revision

Example

17 → 17 → 17 → 17 → 1 → 1 → 1 → 25

Interpretation

Participant receives multiple consecutive PD diagnoses before later
being reclassified.

These participants should be flagged for review.

---

## 6. PD diagnosed at the final observed visit

Example

17 → 17 → 17 → 17 → 17 → 1

Interpretation

The participant is diagnosed with PD during the final available visit.

No subsequent follow-up exists to determine whether the diagnosis
remains stable.

---

## 7. Short follow-up ending in PD

Examples

25 → 1

17 → 1

Interpretation

Participants have only one follow-up after baseline.

Future diagnostic stability is unknown.

---

# Candidate Outcome Variables

Rather than storing only a binary outcome, the outcome table should
capture sufficient information for multiple analyses.

Candidate variables include

| Variable | Description |
|----------|-------------|
|PATNO|Participant identifier|
|first_pd_visit|First visit with PD diagnosis|
|first_pd_year|PPMI follow-up year of first PD diagnosis|
|first_pd_date|Visit date of first PD diagnosis|
|final_diagnosis|Diagnosis at final observed visit|
|ever_pd|Whether PD is ever diagnosed|
|ever_msa|Whether MSA is ever diagnosed|
|ever_dlb|Whether DLB is ever diagnosed|
|diagnosis_sequence|Complete diagnosis trajectory|
|diagnosis_changes|Number of diagnosis transitions|

Additional variables may be added following further exploration.

---

# Proposed Outcome Categories

The current proposal is to classify participants into one of several
outcome categories rather than using only a binary PD outcome.

Candidate categories include

| Category | Description |
|----------|-------------|
|Confirmed PD|PD diagnosed and remains PD until end of follow-up|
|Probable PD|Final observed diagnosis is PD but no future confirmation exists|
|Transient PD|PD diagnosed but subsequently removed|
|Competing diagnosis|Participant ultimately diagnosed with MSA, DLB, or another neurological disorder|
|No PD|No PD diagnosis during follow-up|

These definitions remain provisional.


---
## Outcome design principles

The outcome definition is based on longitudinal diagnostic trajectories rather than
single-visit diagnosis codes.

The following principles are applied:

1. Diagnosis trajectories are ordered chronologically using PPMI visit order
   (EVENT_ID: BL, V01, V02, ...).

2. The first occurrence of a diagnosis is considered the first observed diagnosis
   event.

3. A diagnosis of Parkinson's disease (PD) is defined using PRIMDIAG == 1.

4. Diagnostic changes after the first PD diagnosis are considered when defining
   PD conversion status.

5. Competing neurodegenerative diagnoses (MSA and DLB) are retained separately
   rather than being merged with PD.

---
## PD conversion outcome

A participant is considered to have experienced a PD conversion event if
they receive a first diagnosis of idiopathic Parkinson's disease
(PRIMDIAG == 1) during follow-up.

Participants with a first PD diagnosis are classified into trajectory categories:

| Category | Definition |
|---|---|
| persistent_pd | First PD diagnosis followed by subsequent PD diagnoses |
| pd_end_of_followup | First PD diagnosis occurs at the final observed visit |
| pd_reversal | Non-PD diagnosis occurs after first PD diagnosis |
| pd_competing_diagnosis | MSA or DLB diagnosis occurs after first PD diagnosis |

The primary PD conversion outcome includes:

- persistent_pd
- pd_end_of_followup

because these represent participants with observed PD diagnosis without evidence
of subsequent diagnostic reassignment.

---

## Exploratory trajectory findings

Among 2247 participants in the risk cohort:

| PD trajectory category | n |
|---|---:|
| persistent_pd | 88 |
| pd_end_of_followup | 74 |
| pd_reversal | 20 |
| pd_competing_diagnosis | 4 |

A total of 186 participants had at least one observed PD diagnosis.

---

## Competing diagnoses

Participants developing alternative neurodegenerative diagnoses are not treated
as PD converters.

Competing diagnoses include:

| Diagnosis | PRIMDIAG |
|---|---:|
| Dementia with Lewy bodies | 5 |
| Multiple system atrophy | 11 |

Participants with PD followed by MSA/DLB are classified separately as:

pd_competing_diagnosis

---

## Death outcome

Death is identified using:

- Death_Status
- Death_Date

Death timing relative to PD diagnosis is recorded.

Participants are classified as:

| Category | Definition |
|---|---|
| no_death | No recorded death |
| death_without_pd | Death before any observed PD diagnosis |
| pd_then_death | PD diagnosis occurs before death |

Death may be considered as a competing event in time-to-event analyses.

---

## Sensitivity analyses

Because PD diagnosis may occur near the end of follow-up or may show diagnostic
instability, sensitivity analyses may include:

Primary definition:
- persistent_pd
- pd_end_of_followup

Broad definition:
- all participants with PRIMDIAG == 1 at any visit

Excluded/secondary analyses:
- pd_reversal
- pd_competing_diagnosis

---


# Primary Analysis

The exact definition of PD conversion has **not yet been finalised**.

The exploration analyses suggest that multiple reasonable definitions
exist.

The implementation should therefore be sufficiently flexible to support
alternative outcome definitions without modifying the underlying cohort.

This document will be updated as the outcome definition is refined.

# Open Questions

The following methodological questions remain unresolved.

- Should PD conversion be defined as the first PD diagnosis or only after sustained PD diagnosis?
- How many consecutive PD diagnoses constitute confirmation?
- How should participants with competing diagnoses (e.g., MSA or DLB) be handled?
- Should participants with transient PD diagnoses be excluded or analysed separately?
- Should death before PD diagnosis be treated as a competing event or as censoring?
- How should participants whose final recorded diagnosis is PD be classified when no further follow-up exists?
- Should sensitivity analyses compare alternative outcome definitions?