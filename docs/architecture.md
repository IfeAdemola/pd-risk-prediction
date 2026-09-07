# Model Development Architecture

## Overview

This project separates the **definition of the modelling cohort and exploratory analysis** from the **formal modelling pipeline**.

The modelling pipeline described here assumes that the evaluation cohort and relevant data have already been defined. It takes the prepared data through:

1. Configuration
2. Dataset construction
3. Modality selection
4. Modality-specific preprocessing
5. Multimodal fusion
6. Survival model fitting
7. Cross-validation
8. Performance evaluation
9. Experiment artifact generation

The main entry point for this pipeline is:

```text
scripts/run_experiment.py
```

The `run_experiment.py` script primarily acts as an **orchestrator**. Most of the actual data processing, preprocessing, fusion, modelling, validation, and evaluation logic lives in the `pd_risk.modelling` package.

---

# High-Level Architecture

```text
                         Configuration
                              │
                              ▼
                     ┌─────────────────┐
                     │   ppmi.yaml     │
                     │                 │
                     │ • modalities    │
                     │ • preprocessing │
                     │ • model         │
                     │ • validation    │
                     │ • calibration   │
                     └────────┬────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   ModelDatasetBuilder   │
                 │                         │
                 │ Dataset construction    │
                 │ Modality data           │
                 │ Targets                 │
                 │ Identifiers             │
                 │ Feature metadata        │
                 └────────────┬────────────┘
                              │
                              ▼
                    Enabled modalities
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        Preprocessing    Preprocessing    Preprocessing
             │                │                │
       ┌─────┴─────┐    ┌─────┴─────┐          │
       │ Clinical  │    │ DAT       │          ...
       │ MRI       │    │ Biospec.  │
       │ Genetics  │    │           │
       └─────┬─────┘    └─────┬─────┘
             │                │
             └────────────────┘
                      │
                      ▼
               ┌─────────────┐
               │    Fusion   │
               └──────┬──────┘
                      │
                      ▼
              ┌───────────────┐
              │ Survival Model│
              │               │
              │ Cause-specific│
              │ Cox model     │
              └───────┬───────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Cross-validation│
              └───────┬─────────┘
                      │
                      ▼
              ┌────────────────┐
              │   Evaluation   │
              │                │
              │ • Uno C-index  │
              │ • Brier score  │
              │ • Calibration  │
              │ • Skill score  │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │    Artifacts   │
              └────────────────┘
```

---

# 1. Configuration

**Location:**

```text
configs/prediction/ppmi.yaml
```

The experiment is configuration-driven.

`run_experiment.py` loads the YAML configuration and uses it to determine:

* Which modalities are enabled
* Feature configuration files for modalities
* Preprocessing parameters
* Survival model parameters
* Cross-validation strategy
* Calibration settings
* Other experiment-level parameters

The configuration therefore provides the main mechanism for changing an experiment without modifying the orchestration code.

### Main configuration sections used

```text
modalities
preprocessing
analysis
    model
    validation
    calibration
```

---

# 2. Dataset Construction

**Module:**

```text
pd_risk.modelling.dataset
```

**Main class:**

```text
ModelDatasetBuilder
```

The `ModelDatasetBuilder` is responsible for constructing the dataset used by the modelling pipeline.

It is initialized with:

* The prediction configuration
* The project root

After calling:

```python
builder.build()
```

the experiment expects the builder to provide several important objects:

```text
builder.modality_data
builder.modality_features
builder.targets_df
builder.identifiers
```

### Outputs

#### Modality data

```text
builder.modality_data
```

Contains the data for each modelling modality.

Examples:

```text
clinical
dat
mri
biospecimen
genetics
```

#### Modality feature metadata

```text
builder.modality_features
```

Provides information about the features within each modality, including feature types used by preprocessing.

#### Targets

```text
builder.targets_df
```

The modelling pipeline extracts:

```text
PATNO
time_to_event
event_type
```

These define the survival outcome used by the experiment.

#### Identifiers

```text
builder.identifiers
```

Contains identifiers associated with observations and/or samples.

---

# 3. Modality Selection

Enabled modalities are determined directly from the experiment configuration.

A modality is included when:

```yaml
enabled: true
```

in its configuration.

This allows experiments to use different combinations of data sources without changing the core experiment code.

For example:

```text
Clinical only

Clinical + DAT

Clinical + MRI

Clinical + DAT + MRI

Clinical + DAT + MRI + Biospecimen + Genetics
```

The enabled modalities determine which modality data and preprocessing pipelines are passed to the experiment.

---

# 4. Modality-Specific Preprocessing

**Module:**

```text
pd_risk.modelling.preprocessing
```

The experiment uses a factory pattern to construct the appropriate preprocessor for each enabled modality.

### Clinical

```text
ClinicalPreprocessor
```

Uses feature metadata to distinguish:

* Continuous features
* Binary features
* Categorical features

### DAT

```text
DATPreprocessor
```

Uses a modality-specific feature configuration.

### MRI

```text
MRIPreprocessor
```

Uses a modality-specific feature configuration.

### Biospecimen

```text
BiospecimenPreprocessor
```

Uses a modality-specific feature configuration.

### Genetics

```text
GeneticsPreprocessor
```

Uses a modality-specific feature configuration.

The preprocessors are supplied to the experiment as **factories**, allowing preprocessing objects to be created when required during the experiment.

This is particularly relevant for cross-validation because preprocessing should be fitted using the training data rather than globally before the validation split.

> **Implementation detail to verify:** The exact fitting/application behaviour of each preprocessor should be documented in the preprocessing module, particularly with respect to prevention of data leakage.

---

# 5. Multimodal Fusion

**Module:**

```text
pd_risk.modelling.fusion
```

**Main entry point:**

```text
build_fusion(config)
```

The fusion component combines the processed representations from the enabled modalities into the representation used by the downstream survival model.

The exact fusion strategy is configuration-dependent and should be documented in the fusion module.

Conceptually:

```text
Clinical representation ──┐
DAT representation ───────┤
MRI representation ───────┤
Biospecimen representation┤──► Fusion ──► Model input
Genetics representation ──┘
```

---

# 6. Survival Model

**Module:**

```text
pd_risk.modelling.survival
```

**Model:**

```text
CauseSpecificCoxModel
```

The current experiment uses a cause-specific Cox proportional hazards model.

The model is constructed using the configured penalization parameter:

```text
config["analysis"]["model"]["penalizer"]
```

The model is provided to the experiment as a factory.

This allows the experiment framework to instantiate a fresh model when required, rather than reusing a fitted model across folds.

---

# 7. Cross-Validation

**Module:**

```text
pd_risk.modelling.splitting
```

**Class:**

```text
CrossValidationSplitter
```

The splitter is configured through:

```text
config["analysis"]["validation"]
```

Cross-validation is orchestrated by the `SurvivalExperiment`.

Conceptually:

```text
Dataset
   │
   ▼
Cross-validation split
   │
   ├── Training data
   │       │
   │       ├── Fit preprocessing
   │       ├── Transform training data
   │       ├── Fit fusion
   │       └── Fit survival model
   │
   └── Validation data
           │
           ├── Apply fitted preprocessing
           ├── Apply fitted fusion
           └── Generate predictions
```

The precise splitting strategy, number of folds, randomisation, stratification, and any censoring/event-specific constraints should be documented in the splitting module.

---

# 8. Experiment Orchestration

**Module:**

```text
pd_risk.modelling.experiment
```

**Class:**

```text
SurvivalExperiment
```

This is the central orchestration component of the modelling pipeline.

`run_experiment.py` constructs the experiment by supplying:

* Modality data
* Survival targets
* Identifiers
* Cross-validation splitter
* Preprocessor factories
* Fusion factory
* Model factory
* Evaluator
* Configuration

Conceptually:

```text
                    SurvivalExperiment
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       Splitter       Preprocessors      Fusion
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                     Survival Model
                           │
                           ▼
                       Evaluation
```

The `SurvivalExperiment` therefore represents the **execution layer** of an individual modelling experiment.

---

# 9. Evaluation

**Module:**

```text
pd_risk.modelling.evaluation
```

**Class:**

```text
SurvivalEvaluator
```

The evaluator calculates survival-model performance metrics.

The current experiment reports, when available:

* Uno C-index
* Brier score
* Null Brier score
* Brier skill score

Calibration is configured through:

```text
config["analysis"]["calibration"]["n_groups"]
```

The evaluator therefore provides the performance summary consumed by the experiment and ultimately reported by `run_experiment.py`.

---

# 10. Experiment Artifacts

**Module:**

```text
pd_risk.modelling.artifacts
```

**Class:**

```text
ExperimentArtifacts
```

After the experiment has completed, an `ExperimentArtifacts` object is created using:

* The completed experiment
* The experiment configuration

Calling:

```python
artifacts.save()
```

creates the experiment output directory and saves the experiment artifacts.

The exact contents of this directory should be documented separately once the `ExperimentArtifacts` implementation is reviewed.

---

# 11. Experiment Execution Flow

The complete execution flow from `scripts/run_experiment.py` is:

```text
1. Load configuration
        │
        ▼
2. Determine enabled modalities
        │
        ▼
3. Build ModelDatasetBuilder
        │
        ▼
4. Construct dataset
        │
        ├── modality data
        ├── modality metadata
        ├── targets
        └── identifiers
        │
        ▼
5. Construct preprocessing factories
        │
        ├── ClinicalPreprocessor
        ├── DATPreprocessor
        ├── MRIPreprocessor
        ├── BiospecimenPreprocessor
        └── GeneticsPreprocessor
        │
        ▼
6. Construct fusion factory
        │
        ▼
7. Construct survival model factory
        │
        ▼
8. Construct cross-validation splitter
        │
        ▼
9. Construct evaluator
        │
        ▼
10. Construct SurvivalExperiment
        │
        ▼
11. Run experiment
        │
        ├── cross-validation
        ├── preprocessing
        ├── fusion
        ├── model fitting
        ├── prediction
        └── evaluation
        │
        ▼
12. Save experiment artifacts
        │
        ▼
13. Print completion summary
```

---

# 12. Separation of Responsibilities

The architecture intentionally separates **what is being done** from **how it is executed**.

| Component                 | Responsibility                      |
| ------------------------- | ----------------------------------- |
| `ppmi.yaml`               | Define experiment configuration     |
| `ModelDatasetBuilder`     | Construct modelling dataset         |
| `*Preprocessor`           | Prepare individual modalities       |
| `build_fusion()`          | Combine modality representations    |
| `CauseSpecificCoxModel`   | Fit survival model                  |
| `CrossValidationSplitter` | Define validation splits            |
| `SurvivalEvaluator`       | Calculate performance metrics       |
| `SurvivalExperiment`      | Orchestrate the modelling workflow  |
| `ExperimentArtifacts`     | Save experiment outputs             |
| `run_experiment.py`       | Assemble and execute the experiment |

This separation allows individual components to be modified or replaced without rewriting the entire pipeline.

---

# 13. Scope of This Architecture

This architecture describes the pipeline **after the modelling cohort has been defined**.

It does not currently describe:

* Exploratory data analysis
* Cohort definition
* Inclusion/exclusion criteria
* Missing-data exploration
* Feature exploration
* Feature selection
* Determination of the final modelling cohort
* Exploratory model development
* Decisions made during model development

Those processes should be documented separately.

A useful conceptual separation is:

```text
                    DATA
                      │
                      ▼
          ┌──────────────────────┐
          │ Exploration & Cohort │
          │      Definition      │
          └──────────┬───────────┘
                     │
                     ▼
              Final Cohort
                     │
                     ▼
          ┌──────────────────────┐
          │   Modelling Pipeline │
          │                      │
          │ Dataset              │
          │ Preprocessing        │
          │ Fusion               │
          │ Survival Model       │
          │ Cross-validation     │
          │ Evaluation           │
          │ Artifacts            │
          └──────────────────────┘
```

The boundary between these two areas should be kept explicit because the cohort-definition process is part of the scientific methodology, whereas `run_experiment.py` represents the reproducible execution of the modelling pipeline once that cohort has been established.

---

# 14. Key Entry Points

### Primary experiment entry point

```text
scripts/run_experiment.py
```

### Configuration

```text
configs/prediction/ppmi.yaml
```

### Dataset

```text
pd_risk.modelling.dataset
```

### Preprocessing

```text
pd_risk.modelling.preprocessing
```

### Fusion

```text
pd_risk.modelling.fusion
```

### Survival modelling

```text
pd_risk.modelling.survival
```

### Validation

```text
pd_risk.modelling.splitting
```

### Evaluation

```text
pd_risk.modelling.evaluation
```

### Experiment orchestration

```text
pd_risk.modelling.experiment
```

### Artifacts

```text
pd_risk.modelling.artifacts
```
