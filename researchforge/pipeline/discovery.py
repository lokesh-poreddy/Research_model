"""Algorithm-Discovery Pipeline: retrieve -> recombine -> synthesize ->
unit-test -> pilot -> validate (design doc Sec. 6 flowchart).

`Synthesizer` is a pluggable interface. `HeuristicSynthesizer` maps a chosen
strategy directly onto a genome-evolution operator -- deterministic, offline,
no network calls. `LLMSynthesizer` sketches how a real language model would
instead *propose* the mutation or a new architecture from natural language,
consistent with the design doc's own framing:

    "The LLM is not the research novelty. The LLM is replaceable."
    (source doc, Sec. "Why this is stronger than 'ResearchForge' alone")

It is deliberately left unimplemented in this offline reference build: there
are no API credentials configured in this sandbox, and wiring one up is a
config/credentials change, not an architecture change (see README's
"Extending toward the full design" section for exactly what that change looks
like).
"""
from __future__ import annotations
import random
from typing import List, Protocol

from ..genome.model_genome import ModelGenome, GENOME_SCHEMA
from ..genome.operators import apply_strategy


class Synthesizer(Protocol):
    def synthesize(self, strategy: str, base: ModelGenome, rng: random.Random,
                    population: List[ModelGenome]) -> ModelGenome: ...


class HeuristicSynthesizer:
    """Deterministic strategy -> genome-operator mapping. No network, no LLM."""

    def synthesize(self, strategy: str, base: ModelGenome, rng: random.Random,
                    population: List[ModelGenome]) -> ModelGenome:
        return apply_strategy(strategy, base, rng, population=population)


class LLMSynthesizer:
    """Documented extension point, not wired to a live model here.

    A production build would call an LLM (e.g. the Anthropic Messages API)
    with the current RDG context plus retrieved ECRM memories, ask it to
    propose a Model Genome edit as JSON, and validate the result against
    `genome.model_genome.GENOME_SCHEMA` before use -- e.g.:

        response = client.messages.create(
            model="claude-...",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt_with_rdg_context}],
        )
        candidate = json.loads(extract_json(response.content))
        jsonschema.validate(candidate, GENOME_SCHEMA)   # reject or repair on failure

    That call site is exactly where this class's `synthesize()` would live;
    everything upstream (ECRM retrieval, RDG bookkeeping) and downstream
    (unit_test, evaluate_genome) is already synthesizer-agnostic.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "LLMSynthesizer is a documented extension point, not wired up in "
            "this offline reference implementation. Use HeuristicSynthesizer, "
            "or implement synthesize() with a real API call (see docstring).")


def unit_test(genome: ModelGenome) -> bool:
    """Sanity check: does the genome validate against the schema, pass the
    resource-bound safety check, and build a real, instantiable estimator?
    (design doc Sec. 6 'Unit Test & Sanity Check', extended with the
    safety_check() layer from genome.model_genome)."""
    try:
        genome.validate()
        if genome.safety_check():
            return False
        genome.build_estimator()
        return True
    except Exception:
        return False
