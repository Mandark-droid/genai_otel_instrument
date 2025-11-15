"""Configuration classes for evaluation and safety features.

This module defines configuration dataclasses for all evaluation and safety features
including PII detection, toxicity detection, bias detection, prompt injection detection,
restricted topics, and hallucination detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


class PIIMode(str, Enum):
    """PII detection mode."""

    DETECT = "detect"  # Only detect and report PII
    REDACT = "redact"  # Detect and redact PII entities
    BLOCK = "block"  # Block requests/responses containing PII


class PIIEntityType(str, Enum):
    """PII entity types supported for detection."""

    CREDIT_CARD = "CREDIT_CARD"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    IBAN_CODE = "IBAN_CODE"
    IP_ADDRESS = "IP_ADDRESS"
    PERSON = "PERSON"
    PHONE_NUMBER = "PHONE_NUMBER"
    US_SSN = "US_SSN"
    US_BANK_NUMBER = "US_BANK_NUMBER"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    US_PASSPORT = "US_PASSPORT"
    LOCATION = "LOCATION"
    DATE_TIME = "DATE_TIME"
    NRP = "NRP"  # Named Recognized Person
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    URL = "URL"
    CRYPTO = "CRYPTO"  # Cryptocurrency wallet addresses
    UK_NHS = "UK_NHS"


@dataclass
class PIIConfig:
    """Configuration for PII detection and protection.

    Attributes:
        enabled: Whether PII detection is enabled
        mode: Detection mode (detect, redact, or block)
        entity_types: Set of PII entity types to detect
        redaction_char: Character to use for redaction (default: "*")
        threshold: Confidence threshold for detection (0.0-1.0)
        gdpr_mode: Enable GDPR-specific entity types and rules
        hipaa_mode: Enable HIPAA-specific entity types and rules
        pci_dss_mode: Enable PCI-DSS-specific entity types and rules
        custom_patterns: Custom regex patterns for additional PII detection
        allow_list: Entities to exclude from detection
    """

    enabled: bool = False
    mode: PIIMode = PIIMode.DETECT
    entity_types: Set[PIIEntityType] = field(
        default_factory=lambda: {
            PIIEntityType.CREDIT_CARD,
            PIIEntityType.EMAIL_ADDRESS,
            PIIEntityType.IP_ADDRESS,
            PIIEntityType.PERSON,
            PIIEntityType.PHONE_NUMBER,
            PIIEntityType.US_SSN,
        }
    )
    redaction_char: str = "*"
    threshold: float = 0.7
    gdpr_mode: bool = False
    hipaa_mode: bool = False
    pci_dss_mode: bool = False
    custom_patterns: Optional[dict] = None
    allow_list: Optional[List[str]] = None

    def __post_init__(self):
        """Validate configuration and apply compliance modes."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        # Apply GDPR mode - add EU-specific entities
        if self.gdpr_mode:
            self.entity_types.update(
                {
                    PIIEntityType.IBAN_CODE,
                    PIIEntityType.UK_NHS,
                    PIIEntityType.NRP,
                }
            )

        # Apply HIPAA mode - add healthcare entities
        if self.hipaa_mode:
            self.entity_types.update(
                {
                    PIIEntityType.MEDICAL_LICENSE,
                    PIIEntityType.US_PASSPORT,
                    PIIEntityType.DATE_TIME,
                }
            )

        # Apply PCI-DSS mode - ensure credit card detection
        if self.pci_dss_mode:
            self.entity_types.add(PIIEntityType.CREDIT_CARD)
            self.entity_types.add(PIIEntityType.US_BANK_NUMBER)


@dataclass
class ToxicityConfig:
    """Configuration for toxicity detection.

    Attributes:
        enabled: Whether toxicity detection is enabled
        threshold: Toxicity score threshold (0.0-1.0)
        use_perspective_api: Use Google Perspective API (requires API key)
        perspective_api_key: API key for Perspective API
        use_local_model: Use local Detoxify model as fallback
        categories: Toxicity categories to check
        block_on_detection: Block toxic content instead of just logging
    """

    enabled: bool = False
    threshold: float = 0.7
    use_perspective_api: bool = False
    perspective_api_key: Optional[str] = None
    use_local_model: bool = True
    categories: Set[str] = field(
        default_factory=lambda: {
            "toxicity",
            "severe_toxicity",
            "identity_attack",
            "insult",
            "profanity",
            "threat",
        }
    )
    block_on_detection: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        if self.use_perspective_api and not self.perspective_api_key:
            raise ValueError("perspective_api_key is required when use_perspective_api is True")


@dataclass
class BiasConfig:
    """Configuration for bias detection.

    Attributes:
        enabled: Whether bias detection is enabled
        threshold: Bias score threshold (0.0-1.0)
        bias_types: Types of bias to detect
        use_fairlearn: Use Fairlearn library for ML-based detection
        sensitive_attributes: Attributes to check for bias
        check_prompts: Check prompts for biased language
        check_responses: Check responses for biased language
        block_on_detection: Block content when bias is detected
    """

    enabled: bool = False
    threshold: float = 0.5
    bias_types: Set[str] = field(
        default_factory=lambda: {
            "gender",
            "race",
            "ethnicity",
            "religion",
            "age",
            "disability",
            "sexual_orientation",
            "political",
        }
    )
    use_fairlearn: bool = False
    sensitive_attributes: Optional[List[str]] = None
    check_prompts: bool = True
    check_responses: bool = True
    block_on_detection: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")


@dataclass
class PromptInjectionConfig:
    """Configuration for prompt injection detection.

    Attributes:
        enabled: Whether prompt injection detection is enabled
        threshold: Detection confidence threshold (0.0-1.0)
        use_ml_model: Use ML-based classifier for detection
        check_patterns: Check for known injection patterns
        patterns: Custom injection patterns to detect
        block_on_detection: Block detected injection attempts
        log_attempts: Log all injection attempts for analysis
    """

    enabled: bool = False
    threshold: float = 0.7
    use_ml_model: bool = True
    check_patterns: bool = True
    patterns: Optional[List[str]] = None
    block_on_detection: bool = False
    log_attempts: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")


@dataclass
class RestrictedTopicsConfig:
    """Configuration for restricted topics detection.

    Attributes:
        enabled: Whether restricted topics detection is enabled
        restricted_topics: Optional list of specific topics to restrict
        threshold: Classification confidence threshold (0.0-1.0)
        block_on_detection: Block content matching restricted topics
    """

    enabled: bool = False
    restricted_topics: Optional[List[str]] = None
    threshold: float = 0.5
    block_on_detection: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")


@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection.

    Attributes:
        enabled: Whether hallucination detection is enabled
        threshold: Hallucination score threshold (0.0-1.0)
        check_citations: Check for citation validity
        check_hedging: Check for hedge words
    """

    enabled: bool = False
    threshold: float = 0.7
    check_citations: bool = True
    check_hedging: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
