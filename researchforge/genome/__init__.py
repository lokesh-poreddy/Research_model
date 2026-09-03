"""ResearchForge Genome package.

RF-1.0.0-alpha.2 additions:
  - TargetModelGenome (TMG): versioned successor to ModelGenome
  - ResearchSystemGenome (RSG): governs the research strategy
  - schema.py: deterministic IDs, fingerprinting, validation utilities
  - migration.py: TMG/RSG schema migration registry

RF-0.x ModelGenome and operators are preserved without change for
backward compatibility.
"""
# RF-0.x (unchanged, backward compat)
from .model_genome import ModelGenome, GENOME_SCHEMA, DEFAULT_GENOMES  # noqa: F401
from .operators import (                                                  # noqa: F401
    STRATEGIES,
    apply_strategy,
    param_mutation,
    increase_capacity,
    add_regularization,
    tune_learning_dynamics,
    change_family,
    feature_preprocessing,
    crossover,
    random_perturbation,
)

# RF-1.0.0-alpha.2 (new)
from .schema import (                                                     # noqa: F401
    GENOME_SCHEMA_VERSION_TMG,
    GENOME_SCHEMA_VERSION_RSG,
    deterministic_genome_id,
    genome_fingerprint,
    validate_genome,
)
from .target_model_genome import (                                        # noqa: F401
    TargetModelGenome,
    TMGCapabilities,
    GENOME_SCHEMA_TMG,
    TMG_OPERATORS,
)
from .research_system_genome import (                                     # noqa: F401
    ResearchSystemGenome,
    ResearchMemoryConfig,
    ResearchValidityConfig,
    ResearchRetrievalConfig,
    ExecutionConfig,
    OperatorConfig,
    TerminationConfig,
    GENOME_SCHEMA_RSG,
    RSG_EVOLUTION_OPERATORS,
)
from .migration import migrate_tmg, migrate_rsg                           # noqa: F401
