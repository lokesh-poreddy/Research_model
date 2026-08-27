# Literature-informed design positioning

This document records which current research-agent ideas inform
ResearchForge-ECRM v2. It does not claim to reproduce another system's
results, code, models, or infrastructure.

| System or benchmark | Useful lesson | ResearchForge-ECRM v2 response |
|---|---|---|
| AI Scientist-v2 | Progressive agentic tree search can expand multiple research paths. | Preserve every path in a typed RDG and compare branches under the same evaluator budget. |
| Google Co-Scientist | Specialized agents can generate, critique, and rank hypotheses with scientist collaboration. | Keep human approval gates and require evidence-supported claims; this project does not compete on biomedical hypothesis generation. |
| AlphaEvolve | LLM creativity needs an executable evaluator and a population of candidate programs. | Treat code/model evolution as one RDG operation and keep evaluator provenance separate from the proposal agent. |
| CORAL | Long-running multi-agent evolution benefits from persistence, isolated workspaces, resource management, and health controls. | Add bounded execution, approval gates, and an execution ledger before scaling to asynchronous workers. |
| SeaEvo | Natural-language strategy should be first-class evolutionary state, not a transient reflection. | Track mutation strategy families in a quality-diversity portfolio and retain procedural lessons. |
| External-memory continual-learning research | External memory has its own representation, retrieval, forgetting, and negative-transfer problems. | Weight memory by contextual compatibility and NTR; retain only compact lessons and negative evidence rather than every trace. |
| ScienceAgentBench and SciAgentArena | Scientific agents need realistic, reproducible assessment beyond prose quality. | Use RDE-Bench-style multi-seed, fixed-budget ablations and record reliability alongside score. |

## Core v2 claim to test

> Context-conditioned procedural memory, including explicit negative evidence,
> will improve branch-selection efficiency and reduce repeated or harmful
> transfer compared with flat experiment history and similarity-only retrieval.

That is a falsifiable claim. It succeeds only if A6/A7 in the protocol improve
the full metric set under matched budgets—not because the interface has more
agents, APIs, or generated text.

## Primary sources

- [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- [Google Co-Scientist](https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/)
- [AlphaEvolve](https://arxiv.org/abs/2506.13131)
- [CORAL](https://arxiv.org/abs/2604.01658)
- [SeaEvo](https://arxiv.org/abs/2604.24372)
- [When Continual Learning Moves to Memory](https://arxiv.org/abs/2604.27003)
- [ScienceAgentBench](https://arxiv.org/abs/2410.05080)
- [SciAgentArena](https://arxiv.org/abs/2606.12736)
