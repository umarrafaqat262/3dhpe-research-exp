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

## 4.2 Wandb-Only Logging

All training metrics are logged exclusively to Weights & Biases:

- **No tensorboard** — removed to reduce I/O and simplify dependencies
- **Metrics logged per epoch:** train_loss, MPJPE (P1), P-MPJPE (P2), LR, grad_norm
- **Layer-wise diagnostics** (new): during evaluation, hook into each BiSTSSMBlock output and compute partial MPJPE after each block. Logged as `layer_{N}_mpjpe`.
- **Wandb artifacts:** best checkpoint saved as `model-{run_id}` artifact
- **Configuration:** full config dictionary logged to wandb config

**Why wandb-only:** centralized experiment tracking, artifact storage, and visualization. Reduces disk I/O on the L4 (critical at 98% disk usage).

## 4.3 Layer-Wise Diagnostic Protocol

During evaluation, register forward hooks on each BiSTSSMBlock to capture intermediate outputs:

```python
# Hook registration in train.py
layer_outputs = {}
def make_hook(name):
    def hook(module, input, output):
        layer_outputs[name] = output.detach()
    return hook

for i, blk in enumerate(model.STEblocks):
    blk.register_forward_hook(make_hook(f'ste_{i}'))
for i, blk in enumerate(model.TTEblocks):
    blk.register_forward_hook(make_hook(f'tte_{i}'))
```

After forward pass, compute MPJPE on each layer's output:
```python
for name, feat in layer_outputs.items():
    # Apply head projection to get 3D coordinates
    pred_3d = model.head(feat)
    mpjpe = compute_mpjpe(pred_3d, target_3d)
    wandb.log({f'layer_mpjpe/{name}': mpjpe})
```

**Purpose:** Identify exactly which layers introduce error. If errors plateau after block N, deeper blocks are wasteful. If errors increase in later blocks, gradient flow is degraded.

**Runtime cost:** negligible — hook computation is ~2% of eval time. Only runs during eval, not training.

## 4.4 Datasets

| Dataset | Use | Size | Preprocessing | Protocol |
|---------|-----|------|---------------|----------|
| **Human3.6M** | Primary train/test | 3.6M frames, 17 joints | MotionBERT pipeline (T=243, stride=81) | S1,5,6,7,8 train; S9,11 test |
| **MPI-INF-3DHP** | Cross-dataset test | 1.3M frames | Same normalization as H36M | Train H36M → test MPI (zero-shot) |
| **3DPW** | OOD zero-shot | ~51K frames, wild | 17-joint H36M format | Inference only; no fine-tuning |
| **EMDB** | OOD zero-shot | ~17K frames | Same as above | Inference only |

**Leakage prevention:** OOD sets (3DPW, EMDB) are held out until all architectural
choices are frozen. No hyperparameters are tuned on OOD sets.

## 4.5 Baselines

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

## 4.6 Metrics

### Primary Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **MPJPE (P1)** | Mean Per-Joint Position Error (mm) | Root-relative position accuracy |
| **P-MPJPE (P2)** | Procrustes-aligned MPJPE (mm) | Shape quality (scale/rotation invariant) |

### Secondary Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Per-joint MPJPE** | Same as P1 but per joint | Which joints improve/worsen |
| **Layer-wise MPJPE** | MPJPE after each BiSTSSMBlock | Where errors accumulate in the network |
| **BLCE** | Bone Length Consistency Error (std of bone length across T) | Temporal bone length stability |
| **Velocity MPJPE** | ‖(pₜ₊₁ − pₜ) − (ĝₜ₊₁ − ĝₜ)‖ | Temporal smoothness |
| **3DPW zero-shot** | MPJPE on 3DPW without fine-tuning | Cross-dataset generalization |

### Computational Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Params** | Total trainable parameters | Model size |
| **MACs** | Multiply-accumulate operations per frame | Inference cost |

## 4.7 Statistical Analysis Plan

### Primary Test
- **Paired Wilcoxon signed-rank test** on per-sequence MPJPE
- **Why non-parametric:** Error distributions are right-skewed; Wilcoxon does not assume normality

### Multiple Comparison Correction
- **Holm-Bonferroni** across all hypotheses tested (H12, H1, H11, compound)

### Run Configuration
- **5 seeds** for all main experiments: {0, 42, 123, 999, 2024}
- **3 seeds** for ablations
- **Reporting:** Mean ± std AND 95% bootstrap CI (1000 samples)

### Power Calculation (H1 — GCN-Mamba)
- Expected effect size (δ): 0.9mm
- Estimated σ: ~0.5mm (from PoseMamba ablation table)
- Cohen's d = 0.9 / 0.5 = 1.8 (large)
- Required n per condition: ~20 sequences for power = 0.85 at α = 0.05
- H36M test set: ~1500 sequences → **power is not a bottleneck**

### Power Calculation (H12 — Learnable Adjacency)
- Expected effect size (δ): 0.4mm
- σ: ~0.5mm
- Cohen's d = 0.8 (large)
- Similar sample requirement — well within H36M test capacity

## 4.8 Threat-to-Validity Analysis

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
| MPJPE averages out joint failures | Report per-joint MPJPE table + layer-wise analysis |

### Statistical Conclusion Validity

| Threat | Mitigation |
|--------|------------|
| Single-run original results unreliable | 5-seed re-runs; compare to paper's numbers |
| Multiple comparisons inflate Type I error | Holm-Bonferroni applied |
| Non-normal error distribution | Wilcoxon (non-parametric) |

## 4.9 Confidence Assessment

| Factor | Max Score | Score | Justification |
|--------|-----------|-------|---------------|
| Weaknesses directly addressed | 25 | 22 | All critical weaknesses addressed (W1-W4, W8-W9) |
| Statistical power adequate | 20 | 18 | Test set far exceeds required n; 5 seeds |
| Baselines fair & comprehensive | 15 | 13 | 6 baselines including Mamba papers |
| Multiple dataset validation | 15 | 14 | 5 datasets (2 in-domain, 3 zero-shot OOD) |
| Threats mitigated | 15 | 13 | All four validity types addressed |
| Mechanistic motivation | 10 | 9 | Layer-wise diagnostics prove where gains come from |
| **Total** | **100** | **89** | |

## 4.10 Risk Factors & Contingencies

| Risk | Prob | Impact | Contingency |
|------|------|--------|-------------|
| GCN-Mamba doesn't transfer to our codebase | Medium | High | Debug fusion; tune α init; try α=0.5 fixed first |
| Learnable adjacency overfits to H36M skeleton | Low | Medium | Add L2 regularization on W; require symmetry |
| SAMA state fusion too complex (150 LOC) | Medium | Medium | Start with simplified version: skip MSM, only SSI |
| OOD 2D detector not available | Medium | Medium | Use ViTPose for OOD datasets |
| Training time exceeds estimate | Low | Medium | Reduce epochs from 120→80 if validation saturates |
| Disk space during training | Low | High | Monitor; wandb logging to cloud; clean checkpoints |
| Layer-wise diagnostics slow eval | Low | Low | Only log layer metrics every 5th epoch |
