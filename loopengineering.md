# Autonomous Research Scientist — Loop Engineering Guide

> A self-contained protocol for autonomous scientific discovery.
> Every claim demands evidence. Every decision demands a reason.
> The goal is not to change code — it is to produce knowledge.

---

## 0. Core Commitments

1. **One hypothesis per experiment.** A combined intervention cannot be interpreted. If A+B improves performance, you do not know which caused it.

2. **Every hypothesis must be falsifiable.** State the condition under which the hypothesis is wrong *before* implementing. If you cannot, the hypothesis is not ready.

3. **Baselines are sacred.** An improvement measured against a moving baseline is not an improvement. Reproduce first. Branch after.

4. **Effect size matters more than significance.** p < 0.05 with Δ = 0.05mm is worthless. Δ > 0.3mm with p = 0.10 is worth more seeds.

5. **Negative results are knowledge.** A well-executed experiment that disproves a hypothesis is as valuable as one that confirms it.

6. **The simplest intervention is the best.** If you can test with a config change, do that before writing code. If 10 lines suffice, do not write 100.

---

## 1. Initialization

### 1.1 Survey

Read the project root, README, all top-level `.md` files. Ask the user for the goal. Output a one-paragraph project summary.

### 1.2 Read the Paper

Extract:
- **Claim:** what does it assert? (exact metrics)
- **Architecture:** components and why each exists
- **Assumptions:** what does it take for granted?
- **Acknowledged limitations:** what does it say it cannot do?
- **Unspoken limitations:** what does the architecture implicitly preclude?

Unspoken limitations are where novel improvements live.

### 1.3 Read the Code

Map every paper component to its implementation. Output a table:

| Paper Component | Code File | Lines | Verified? |
|----------------|-----------|-------|-----------|
| | | | yes/no/partial |

Find: deviations from paper, bugs, inefficiencies, missing assertions.

### 1.4 Survey the Literature

Search:

```
1. "<target> limitations"
2. "<target> improved"
3. "beyond <architecture> <task>"
4. "SOTA <benchmark> 2025 2026"
5. "<target> site:github.com"
6. "<architecture> ablation study"
```

For each relevant paper record: method, key idea, result, code. Synthesize into a limitations table:

| Limitation | Papers | Proven Fixes | Effect | Cost |
|------------|--------|--------------|--------|------|
| | | | | |

### 1.5 Reproduce Baseline

Train with published config. Evaluate with official protocol. Compare to published numbers.

**Pass:** within ±0.5mm (or ±2% relative) of published value.

If reproduction fails, isolate cause systematically:
1. Same data?
2. Same config?
3. Code deviations from paper?
4. Random seed sensitivity?
5. Hardware/software differences?

Do not proceed until baseline is stable.

---

## 2. Hypothesis Generation

### 2.1 From Literature

Each limitation in the literature table is a candidate hypothesis. Frame it as:

```
If [limitation] affects the system,
and [intervention] addresses it in related work,
then implementing [intervention] in this codebase should improve [metric].
```

### 2.2 From Code Audit

Every bug or deviation found during initialization is a candidate. Fixing real bugs improves the system regardless of literature support.

### 2.3 From First Principles

Identify the fundamental bottlenecks:

| Bottleneck Class | Questions |
|-----------------|-----------|
| Information flow | Can gradients propagate effectively to all inputs? |
| Capacity | Is the model large enough for the function? |
| Inductive bias | Does the architecture match the problem structure? |
| Optimization | Is the training setup (LR, schedule, loss) optimal? |
| Data | Is the data representation fully utilized? |

For each bottleneck, design the simplest intervention that tests whether it is indeed a bottleneck.

### 2.4 The Anti-Hypothesis Requirement

For every hypothesis, state:

```
If [hypothesis] is true, then experiment [E] will produce [R].
If experiment [E] produces [not R], then [hypothesis] is false.
```

If you cannot state what would falsify it, you have not designed a testable hypothesis.

---

## 3. Hypothesis Ranking

### 3.1 Priority Levels

| Priority | Criteria | Action |
|----------|----------|--------|
| **P0** | 0 params, literature-proven, Δ > 0.3mm | Do immediately |
| **P1** | Evidence score ≥ 3, Δ > 0.2mm, cost ≤ 2 days | Do after P0 |
| **P2** | Evidence score ≥ 1, Δ > 0.1mm, any cost | Do if P0/P1 exhausted |
| **P3** | All others | Defer indefinitely |

**Evidence scores:** 5 = proven on same benchmark, 3 = multiple papers same domain, 1 = single paper or related domain, 0 = speculative.

### 3.2 Dependency Ordering

1. **Training-only experiments** (LR schedule, clipping, EMA) — independent, can batch
2. **Input experiments** (confidence, bone vectors) — independent of scan changes
3. **Spatial experiments** (scan ordering) — modify model, sequential with other model changes
4. **Architecture experiments** (decoupled blocks) — modify model structure
5. **Combinations** — test only after individual components are confirmed

---

## 4. Pre-Implementation Validation

Before writing code, check:

| Check | Question | Fail = delay/reject |
|-------|----------|---------------------|
| Mechanism | Why should this work? Be specific about the bottleneck. | Cannot articulate mechanism |
| Conflict | Does this change interact badly with existing modules? | Confirmed conflict without workaround |
| Simplicity | Is there a simpler way to test the same hypothesis? | Simpler option exists |
| Falsification | Can the experiment produce a result that disproves the hypothesis? | Every outcome "confirms" |

---

## 5. Staged Experimentation

This is the core loop. Each stage has a single question, expected outcome, and decision rule.

### Setup

```
git checkout -b exp/<phase><number>-<desc>
```

Make the minimum change. Verify numerical correctness (no NaN, loss decreases, gradients flow). Commit:

```
git add <changed files>
git commit -m "exp/<id>: <description>"
```

### Stage 1 — Correctness (CPU, 5 min)

| Question | Expected | Actual | Action |
|----------|----------|--------|--------|
| Does the code run without error? | Loss decreases | NaN / divergence | Fix bug or reject |
| Are outputs finite? | No NaN | NaN found | Fix or reject |
| Do gradients exist? | Non-zero gradients | Zero gradients | Fix or reject |

**If any check fails:** debug and fix, or reject the branch.

### Stage 2 — Overfitting (CPU, 10 min)

Train on 16-32 samples for 300 epochs.

| Question | Expected | Actual | Action |
|----------|----------|--------|--------|
| Can the model learn the training set? | Near-zero loss | Cannot overfit | Reject hypothesis |
| Does it learn as fast as baseline? | Same speed | Slower | Warning — proceed if Δ is promising |

### Stage 3 — Short Validation (GPU, ~3 hrs, 20% epochs)

Train on full dataset for 20% of total epochs.

| Δ vs Baseline | Meaning | Action |
|---------------|---------|--------|
| Δ < -0.2mm | Strong signal | Proceed to Stage 4 |
| -0.2mm ≤ Δ ≤ +0.1mm | Weak signal | Options: (a) tune hyperparams and re-Stage-3, (b) proceed to Stage 4 if evidence is strong (score ≥ 3) |
| Δ > +0.1mm | Negative signal | **Reject.** Do not waste GPU on Stage 4. |

### Stage 4 — Full Validation (GPU, ~36 hrs, 100% epochs × 3 seeds)

| Result | Meaning | Action |
|--------|---------|--------|
| p < 0.05 AND Δ < -0.1mm | Hypothesis supported | **Merge** |
| p > 0.10 OR Δ > -0.0mm | Not supported | **Revert** |
| 0.05 ≤ p ≤ 0.10 AND Δ < -0.2mm | Borderline | Run 2 more seeds (5 total) |

---

## 6. Post-Experiment Actions

### If Merged

```
1. git checkout master
2. git merge --squash exp/<id>
3. git commit -m "Merge exp/<id>: <description>"
4. git branch -D exp/<id>
5. git tag baseline-<id>
```

Then update the project state:

| Artifact | Update |
|----------|--------|
| Leaderboard | Add new row with merged Δ |
| Hypothesis bank | Mark hypothesis as `Confirmed` |
| Lessons | Add entry in "What Works" with Δ and analysis |
| Next experiment | Re-rank remaining hypotheses given new baseline |

### If Reverted

```
1. git checkout master
2. git branch -D exp/<id>
```

Then update:

| Artifact | Update |
|----------|--------|
| Hypothesis bank | Mark as `Falsified` with Δ and notes |
| Lessons | Add entry in "What Doesn't Work" |
| Next experiment | Re-rank — move to next untested hypothesis |

### Experiment Journal

Create `experiments/exp/<id>/README.md` with:

```markdown
## Exp <id>: <name>

**Status:** ✅ Merged / ❌ Reverted
**Branch:** exp/<id>
**Baseline:** X.X mm

### Hypothesis
<one line>

### Change
<what changed, where, LOC>

### Results

| Metric | Baseline | Ours | Δ | p |
|--------|----------|------|---|--|
| MPJPE | | | | |
| Params | | | | |

### Analysis
<why it worked or didn't>

### Lessons
<what to repeat or avoid>
```

---

## 7. The Hypothesis Bank

Maintained in `RESEARCH_PLAN.md`. Single source of truth for what to do next.

```
## Hypothesis Bank

| ID | Hypothesis | Ev | Est Δ | Cost | Priority | Status |
|----|-----------|----|-------|------|----------|--------|
| A2 | Confidence input improves MPJPE | 3 | -0.5mm | Config | P0 | Ready |
| B1 | Bone vectors improve MPJPE | 3 | -0.5mm | 15 LOC | P1 | Ready |
| ... | ... | ... | ... | ... | ... | ... |
```

**Statuses:** `Ready` → `In Progress` → `Confirmed` | `Falsified`

After every experiment, re-rank and update. At the start of each loop cycle, pick the highest-priority `Ready` hypothesis.

---

## 8. Stopping Condition

Stop when **at least 3 of 4** conditions are met:

| # | Condition | Check |
|---|-----------|-------|
| 1 | Last 3 experiments all failed (rejected at Stage 3 or reverted at Stage 4) | Count consecutive fails |
| 2 | No remaining hypothesis with Priority ≤ P1 | Check hypothesis bank |
| 3 | No new hypotheses generated in the last 2 cycles | Check cycle log |
| 4 | Total improvement < 0.2mm over the last 5 experiments | Check leaderboard |

If stopped, document the final state and publish results.

---

## Appendix A: Decision Flow

```
START CYCLE
  │
  ├── Read hypothesis bank
  ├── Pick top Ready hypothesis
  ├── Run Pre-Implementation Validation
  │     └── Fail → skip hypothesis (mark Deferred)
  │
  ├── git checkout -b exp/<id>
  ├── Implement minimum change
  ├── git commit
  │
  ├── Stage 1 (correctness)
  │     └── Fail → fix or git branch -D
  │
  ├── Stage 2 (overfit)
  │     └── Fail → git branch -D, mark Falsified
  │
  ├── Stage 3 (short val, 20%)
  │     ├── Δ < -0.2mm → Stage 4
  │     ├── -0.2mm ≤ Δ ≤ +0.1mm → tune or skip to Stage 4 if strong evidence
  │     └── Δ > +0.1mm → git branch -D, mark Falsified
  │
  ├── Stage 4 (full, 3 seeds)
  │     ├── p < 0.05, Δ < -0.1mm → MERGE
  │     ├── borderline → 2 more seeds
  │     └── p > 0.10 → REVERT
  │
  ├── Write experiment journal
  ├── Update leaderboard, hypothesis bank, lessons
  │
  └── LOOP (go to START CYCLE)
```

## Appendix B: Quick Reference

| Action | Command |
|--------|---------|
| Start experiment | `git checkout -b exp/<id>-<name>` |
| Stage 1 | `python validate.py --config <config> --device cpu` |
| Stage 2 | Train on 16 samples, 300 epochs |
| Stage 3 | Train 20% epochs with `--config <config>` |
| Stage 4 | `bash experiments/scripts/stage4.sh <config> <seed>` |
| Merge | `git checkout master && git merge --squash exp/<id> && git commit` |
| Revert | `git checkout master && git branch -D exp/<id>` |
| Evaluate checkpoint | `python train.py --evaluate best_epoch.bin --config <config>` |
| File structure | See `RESEARCH_PLAN.md` |

## Appendix C: Domain Reference

This section is populated during Phase 0. It contains the project-specific skeleton definition, benchmark details, and config parameters. See `RESEARCH_PLAN.md` for the current state.
