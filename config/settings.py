"""Configuration settings for the Maintenance RAG system."""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AzureOpenAISettings:
    """Azure OpenAI configuration settings."""

    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"))
    embedding_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"))
    chat_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"))
    enrichment_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENRICHMENT_DEPLOYMENT", "gpt-4o-mini"))


@dataclass
class EmbeddingSettings:
    """Embedding configuration settings."""

    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100


@dataclass
class ChunkingSettings:
    """Text chunking configuration settings."""

    chunk_size: int = 2000  # ~500 tokens
    chunk_overlap: int = 400  # ~100 tokens
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


@dataclass
class RetrievalSettings:
    """Retrieval configuration settings."""

    # Dense retrieval
    dense_top_k: int = 50

    # Sparse retrieval (BM25)
    sparse_top_k: int = 50

    # RRF fusion
    rrf_k: int = 60

    # Multi-tier retrieval
    tier1_top_k: int = 10  # Equipment-specific
    tier2_top_k: int = 20  # Equipment-type
    tier3_top_k: int = 20  # Generic/Universal
    tier4_top_k: int = 30  # Semantic

    # Score boosting
    equipment_specific_boost: float = 1.5
    equipment_type_boost: float = 1.2
    universal_boost: float = 1.3
    safety_critical_boost: float = 1.4
    procedure_overlap_boost: float = 0.1  # Per overlapping procedure

    # Final results
    final_top_k: int = 5
    min_relevance_score: float = 0.5


@dataclass
class EnrichmentSettings:
    """LLM enrichment configuration settings."""

    temperature: float = 0.1
    max_tokens: int = 300
    batch_size: int = 10

    # Confidence thresholds
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.70

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class GenerationSettings:
    """LLM generation configuration settings."""

    temperature: float = 0.3
    max_tokens: int = 500


@dataclass
class AppSettings:
    """Application-level settings."""

    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    chroma_persist_directory: str = "chroma_db"

    # Data validation
    min_description_length: int = 10
    max_description_length: int = 10000

    # Required columns
    lessons_required_columns: list = field(default_factory=lambda: [
        "lesson_id", "title", "description", "root_cause",
        "corrective_action", "category"
    ])
    lessons_optional_columns: list = field(default_factory=lambda: [
        "equipment_tag", "date", "severity"
    ])
    jobs_required_columns: list = field(default_factory=lambda: [
        "job_id", "job_title", "job_description"
    ])
    jobs_optional_columns: list = field(default_factory=lambda: [
        "equipment_tag", "job_type", "planned_date"
    ])

    # Enrichment columns (auto-generated)
    enrichment_columns: list = field(default_factory=lambda: [
        "specificity_level", "equipment_type", "equipment_family",
        "applicable_to", "procedure_tags", "lesson_scope",
        "safety_categories", "enrichment_confidence",
        "enrichment_timestamp", "enrichment_reviewed"
    ])


@dataclass
class Settings:
    """Main settings container."""

    azure_openai: AzureOpenAISettings = field(default_factory=AzureOpenAISettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    enrichment: EnrichmentSettings = field(default_factory=EnrichmentSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    app: AppSettings = field(default_factory=AppSettings)

    def validate(self) -> bool:
        """Validate that required settings are configured."""
        if not self.azure_openai.endpoint:
            return False
        if not self.azure_openai.api_key:
            return False
        return True

    def get_validation_errors(self) -> list[str]:
        """Get list of validation errors."""
        errors = []
        if not self.azure_openai.endpoint:
            errors.append("AZURE_OPENAI_ENDPOINT is not configured")
        if not self.azure_openai.api_key:
            errors.append("AZURE_OPENAI_API_KEY is not configured")
        return errors


# Global settings instance
settings = Settings()


# Standardized taxonomy for enrichment
EQUIPMENT_TYPES = [
    "centrifugal_pump", "reciprocating_pump", "positive_displacement_pump",
    "heat_exchanger", "shell_tube_exchanger", "plate_exchanger",
    "compressor", "centrifugal_compressor", "reciprocating_compressor",
    "valve", "gate_valve", "globe_valve", "ball_valve", "check_valve",
    "tank", "vessel", "reactor", "column", "tower",
    "motor", "generator", "transformer",
    "instrument", "sensor", "transmitter", "controller",
    "pipe", "fitting", "flange", "gasket",
    "seal", "mechanical_seal", "bearing",
]

EQUIPMENT_FAMILIES = [
    "rotating_equipment",
    "static_equipment",
    "instrumentation",
    "electrical",
    "piping",
    "structural",
]

PROCEDURE_TAGS = [
    "installation", "commissioning", "startup", "shutdown",
    "inspection", "maintenance", "repair", "replacement",
    "calibration", "testing", "alignment", "balancing",
    "lubrication", "cleaning", "flushing",
    "isolation", "lockout_tagout", "permit_to_work",
    "confined_space", "hot_work", "cold_work",
    "lifting", "rigging", "scaffolding",
    "welding", "grinding", "cutting",
    "torque_spec", "quality_control", "verification",
    "training", "documentation",
]

SAFETY_CATEGORIES = [
    "lockout_tagout", "permit_to_work", "confined_space",
    "hot_work", "pressure_release", "toxic_exposure",
    "flammable", "explosive", "electrical_hazard",
    "fall_protection", "ppe_requirement", "emergency_response",
    "environmental", "spill_prevention",
]

LESSON_SCOPES = ["specific", "general", "universal"]
SPECIFICITY_LEVELS = ["equipment_id", "equipment_type", "generic"]
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
CATEGORIES = ["mechanical", "electrical", "safety", "process", "instrumentation"]
