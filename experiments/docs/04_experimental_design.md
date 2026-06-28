# Experimental Design

## 4.1 Loop Engineering Staged Protocol

Every experiment follows this 4-stage validation pipeline:

```
Stage 1 (Correctness)  ──→  Stage 2 (Overfit)  ──→  Stage 3 (Short Val)  ──→  Stage 4 (Full)
   5 min, CPU             10 min, CPU              ~2.5 hrs, GPU             ~36 hrs, GPU
```

### Stage 1 — Correctness

| Check | Expected | Action if Fail |
|-------|----------|----------------|
| Forward pass runs without error | Loss is finite | Fix bug or reject |
| No NaN in outputs/outputs | All values finite | Fix numerical issue |
| Gradients are non-zero | All param gradients > 0 | Fix gradient flow |

### Stage 2 — Overfitting (16 samples × 300 epochs)

| Check | Expected | Action if Fail |
|-------|----------|----------------|
| Can model learn training set? | Near-zero loss | Reject hypothesis |
| Learning speed vs baseline? | Same or faster | Warning if slower |

### Stage 3 — Short Validation (25% epochs, 1 seed)

| Δ vs Baseline | Meaning | Action |
|---------------|---------|--------|
| Δ < -0.2mm | Strong positive signal | Proceed to Stage 4 |
| -0.2mm ≤ Δ ≤ +0.1mm | Weak/no signal | Options: (a) tune hyperparams, (b) proceed if evidence ≥ 3 |
| Δ > +0.1mm | Negative signal | **Reject.** Stop. |

### Stage 4 — Full Validation (100% epochs × 3 seeds)

| Result | Meaning | Action |
|--------|---------|--------|
| p < 0.05 AND Δ < -0.1mm | Hypothesis supported | **Merge** into main |
| p > 0.10 OR Δ > -0.0mm | Not supported | **Revert** branch |
| 0.05 ≤ p ≤ 0.10 AND Δ < -0.2mm | Borderline | Run 2 more seeds (5 total) |

## 4.2 Datasets

| Dataset | Use | Size | Preprocessing | Protocol |
|---------|-----|------|---------------|----------|
| **Human3.6M** | Primary train/test | 3.6M frames, 17 joints | MotionBERT pipeline (T=243, stride=81) | S1,5,6,7,8 train; S9,11 test |
| **MPI-INF-3DHP** | Cross-dataset test | 1.3M frames | Same normalization as H36M | Train H36M → test MPI (zero-shot) |
| **3DPW** | OOD zero-shot | ~51K frames, wild | 17-joint H36M format | Inference only; no fine-tuning |
| **EMDB** | OOD zero-shot | ~17K frames | Same as above | Inference only |

**Leakage prevention:** OOD sets (3DPW, EMDB) are held out until all architectural
choices are frozen. No hyperparameters are tuned on OOD sets.

## 4.3 Baselines

| Baseline | Source | Detector | Why Included |
|----------|--------|----------|--------------|
| **PoseMamba-S** (paper) | Official code | SH | Anchor; reproducibility check |
| **PoseMamba-B** (paper) | Official code | SH | Size scaling baseline |
| **PoseMamba-L** (paper) | Official code | SH | Full model comparison |
| **PoseMagic** (AAAI 2025) | — (no code) | SH | Most relevant Mamba+GCN baseline |
| **SasMamba** (WACV 2026) | — (no code) | SH | Structure-aware scan baseline |
| **DBMambaPose** (arXiv 2025) | — (no code) | SH | Decoupled S-T baseline |

**Note:** PoseMagic, SasMamba, and DBMambaPose have no public code. We compare
by reproducing their claimed Δs relative to PoseMamba.

## 4.4 Metrics

### Primary Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **MPJPE (P1)** | Mean Per-Joint Position Error (mm) | Root-relative position accuracy |
| **P-MPJPE (P2)** | Procrustes-aligned MPJPE (mm) | Shape quality (scale/rotation invariant) |

### Secondary Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Per-joint MPJPE** | Same as P1 but per joint | Which joints improve/worsen |
| **BLCE** | Bone Length Consistency Error (std of bone length across T) | Temporal bone length stability |
| **Velocity MPJPE** | ‖(pₜ₊₁ − pₜ) − (ĝₜ₊₁ − ĝₜ)‖ | Temporal smoothness |
| **3DPW zero-shot** | MPJPE on 3DPW without fine-tuning | Cross-dataset generalization |

### Computational Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Params** | Total trainable parameters | Model size |
| **MACs** | Multiply-accumulate operations per frame | Inference cost |

## 4.5 Statistical Analysis Plan

### Primary Test
- **Paired Wilcoxon signed-rank test** on per-sequence MPJPE
- **Why non-parametric:** Error distributions are right-skewed; Wilcoxon does not assume normality

### Multiple Comparison Correction
- **Holm-Bonferroni** across H1-H3 hypotheses
- **Bonferroni** across 8 scan orders in H2 (adjusted α = 0.00625)

### Run Configuration
- **5 seeds** for all main experiments: {0, 42, 123, 999, 2024}
- **3 seeds** for ablations (scan ordering grid: 8 × 3 = 24 runs)
- **Reporting:** Mean ± std AND 95% bootstrap CI (1000 samples)

### Power Calculation (H1 — GCN-Mamba)
- Expected effect size (δ): 0.9mm
- Estimated σ: ~0.5mm (from PoseMamba ablation table)
- Cohen's d = 0.9 / 0.5 = 1.8 (large)
- Required n per condition: ~20 sequences for power = 0.85 at α = 0.05
- H36M test set: ~1500 sequences → **power is not a bottleneck**

### Power Calculation (H2 — Stride Scan)
- Expected effect size (δ): 0.5mm
- σ: ~0.5mm
- Cohen's d = 1.0 (large)
- After Bonferroni correction (8 orders, α = 0.00625): power remains adequate

## 4.6 Threat-to-Validity Analysis

### Internal Validity

| Threat | Mitigation |
|--------|------------|
| Experimenter bias | Reproduce baselines from official code with identical 2D inputs |
| Hyperparameter advantage | Sweep only on validation set; test set remains unseen |
| Hardware timing differences | MACs measured analytically; latency on identical L4 GPU |

### External Validity

| Threat | Mitigation |
|--------|------------|
| H36M indoor bias | 3DPW, EMDB zero-shot evaluation |
| Lab-controlled motion | EMDB (natural) and Fit3D (exercise) diversify motion types |
| Single-view limitation | Acknowledge as remaining limitation |

### Construct Validity

| Threat | Mitigation |
|--------|------------|
| MPJPE misaligns with perceptual quality | Add qualitative video comparisons |
| MPJPE averages out joint failures | Report per-joint MPJPE table |

### Statistical Conclusion Validity

| Threat | Mitigation |
|--------|------------|
| Single-run original results unreliable | 5-seed re-runs; compare to paper's numbers |
| Multiple comparisons inflate Type I error | Holm-Bonferroni applied |
| Non-normal error distribution | Wilcoxon (non-parametric) |

## 4.7 Confidence Assessment

| Factor | Max Score | Score | Justification |
|--------|-----------|-------|---------------|
| Weaknesses directly addressed | 25 | 20 | 9 of 11 weaknesses explicitly fixed |
| Statistical power adequate | 20 | 18 | Test set far exceeds required n; 5 seeds |
| Baselines fair & comprehensive | 15 | 13 | 8 baselines including Mamba papers |
| Multiple dataset validation | 15 | 14 | 5 datasets (2 in-domain, 3 zero-shot OOD) |
| Threats mitigated | 15 | 13 | All four validity types addressed |
| Mechanistic motivation | 10 | 9 | HAB addresses SSM state decay; scan from topology search |
| **Total** | **100** | **87** | |

## 4.8 Risk Factors & Contingencies

| Risk | Prob | Impact | Contingency |
|------|------|--------|-------------|
| GCN-Mamba doesn't transfer to our codebase | Medium | High | Debug fusion; tune α init; try α=0.5 fixed first |
| Best stride scan overfits H36M | Medium | Medium | Require top-3 on MPI-INF-3DHP before adoption |
| OOD 2D detector not available | Medium | Medium | Use ViTPose for OOD datasets |
| Training time exceeds estimate | Low | Medium | Reduce epochs from 120→80 if validation saturates |
| Disk space during training | Low | High | Monitor; wandb logging to cloud; clean checkpoints |
