# ResearchForge-ECRM v2 research protocol

ResearchForge-ECRM is a research-development framework, not a claim that an
LLM can autonomously validate scientific truth. Its job is to make model and
algorithm development more efficient, auditable, and less repetitive under a
fixed experimental budget.

## V2 operating loop

1. The researcher writes or approves an editable project brief: objective,
   metric, constraints, data regime, and authority boundaries.
2. The controller proposes a research branch and an evolution strategy.
3. A task adapter runs a bounded, reproducible evaluation in an isolated
   evaluator.
4. The RDG records the complete lineage: hypothesis, genome, experiment,
   finding, failure diagnosis, and claim.
5. ECRM compresses the episode into a procedural lesson or negative evidence.
6. The next branch score combines policy value, uncertainty, contextual memory
   support, and a repeated-failure penalty.
7. The researcher reviews any action that changes data access, cost, scope,
   safety posture, or the interpretation of a scientific claim.

## What is new in v2

| Mechanism | Why it exists | Implementation rule |
|---|---|---|
| Context-conditioned retrieval | Text similarity alone causes harmful transfer. | Score retrieval by semantic match, domain/task/objective/model compatibility, retention, and measured NTR. |
| Procedural lessons | Full trajectories are verbose and transfer poorly. | Store a compact action + conditions + outcome statement; keep the detailed episode in the RDG. |
| Selective retention | A raw experiment log becomes a noisy retrieval corpus. | Retain failures and successes exceeding the decision-time baseline by a minimum margin; consolidate duplicate lessons. |
| Strategy portfolio | Scalar fitness tends to saturate one mutation family. | Track optimization, architecture, and data strategies separately; favor underexplored, promising families. |
| Evidence-aware policy | Memory must affect action, not merely prompt text. | Apply a bounded positive/negative memory adjustment to UCB branch acquisition. |
| Promotion gates | A lucky local score is not a research result. | Require held-out evaluation, seeds, provenance, and claim support before promotion. |

## Promotion standard

Do not promote a candidate model, strategy, or memory rule because it improves
one validation run. A promotion requires all of the following:

- A preregistered task objective and non-leaking validation/test split.
- At least five independent seeds for stochastic experiments.
- A fixed compute or wall-clock budget shared by every ablation.
- A saved model genome, command, dependency version, random seed, data
  fingerprint, and RDG evidence chain.
- A comparison against the current champion and a relevant no-memory baseline.
- Failure and negative-transfer analysis, not only the best score.
- Human review for a scientific claim or external release.

## Required ablation ladder

Run the same task, seed set, and budget at each rung. Do not use a later rung's
result to change an earlier rung.

| Condition | Adds | Question answered |
|---|---|---|
| A0 | Fixed baseline model | What is the task floor? |
| A1 | LLM or heuristic proposal only | Does proposal generation help? |
| A2 | Evolution plus evaluator | Does iterative search help? |
| A3 | Full RDG provenance | Does typed developmental state improve attribution? |
| A4 | Positive procedural lessons | Can experience reuse improve efficiency? |
| A5 | Negative evidence | Does explicit failure knowledge reduce repetition? |
| A6 | Context compatibility + NTR control | Does memory avoid harmful transfer? |
| A7 | Strategy portfolio + policy | Does memory-guided, diverse search improve the full loop? |

Report final held-out score, research efficiency, evaluations until first
improvement, failure repetition rate, memory utility, negative-transfer rate,
memory half-life, and research reliability score. Report confidence intervals
or paired seed-wise comparisons; do not report a single best run as evidence.

## Continuous improvement cycle

Every system change is a candidate, not an automatic upgrade:

1. Write the hypothesis and expected trade-off in the RDG.
2. Add a deterministic unit test for the intended safety or correctness
   behavior.
3. Run the ablation protocol with the old and candidate versions.
4. Promote only if the candidate improves the specified metric without a
   regression in reliability, cost, or NTR beyond its predefined tolerance.
5. Archive the result as positive or negative evidence, then update the policy
   only from the validated record.

## Current limits

The repository's digits example proves that the controller can run real local
training and evaluation. It is not a trained ECG, CIFAR-10, Sentinel, or
algorithm-discovery system. Domain adapters, real datasets, GPU backends,
external tool credentials, and complete multi-seed benchmark reports must be
added before making domain or state-of-the-art claims.
