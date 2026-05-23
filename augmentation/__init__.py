"""
AgentMind — Cognitive Apprenticeship Data Augmentation Pipeline.

Multi-layer synthetic data expansion from seeds to 100K agentic trajectories.

Layers:
  1. Semantic Query Mutation   — paraphrase, compress, noise-inject
  2. Tool Order Mutation       — reorder/restructure multi-tool chains
  3. Observation Mutation      — realistic environment outputs
  4. Retry / Recovery Mutation — failure recovery strategies
  5. Planner Style Mutation    — cognitive trajectory diversity
  6. Graph-based Generation    — NetworkX DAG → linearized traces
  7. Adversarial Mutation      — property-based stress testing
  8. Environment Simulation    — Faker/Mimesis world states
"""

from .core import (
    TOOL_NAMES, TOOL_DEFS,
    parse_tool_calls,
    rebuild_from_segments,
    apply_positional,
    validate_sample,
    EntropyScorer,
    DuplicateDetector,
)
from .semantic_mutator import SemanticMutator
from .trajectory_mutator import TrajectoryMutator
from .observation_mutator import ObservationMutator
from .graph_generator import GraphGenerator
from .adversarial_mutator import AdversarialMutator
from .environment_generator import EnvironmentGenerator
from .dataset_expander import DatasetExpander

__all__ = [
    "TOOL_NAMES", "TOOL_DEFS",
    "parse_tool_calls",
    "rebuild_from_segments",
    "apply_positional",
    "validate_sample",
    "EntropyScorer",
    "DuplicateDetector",
    "SemanticMutator",
    "TrajectoryMutator",
    "ObservationMutator",
    "GraphGenerator",
    "AdversarialMutator",
    "EnvironmentGenerator",
    "DatasetExpander",
]
