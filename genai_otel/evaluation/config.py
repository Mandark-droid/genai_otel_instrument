"""Configuration classes for evaluation and safety features.

This module defines configuration dataclasses for all evaluation and safety features
including PII detection, toxicity detection, bias detection, prompt injection detection,
restricted topics, and hallucination detection.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


def _env_bool(name: str, default: str) -> bool:
    """Read a boolean environment variable ("true"/"false")."""
    return os.getenv(name, default).lower() == "true"


def _is_hardening_profile() -> bool:
    """True when a locked-down deployment profile (bank/bfsi/strict) is active."""
    return os.getenv("GENAI_PROFILE", "").lower() in ("strict", "bfsi", "bank")


class PIIMode(str, Enum):
    """PII detection mode."""

    DETECT = "detect"  # Only detect and report PII
    REDACT = "redact"  # Detect and add a redacted copy of the text as an attribute
    BLOCK = "block"  # Flag PII on the span (post-call annotation; does NOT block the call)


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
    # India-specific PII types
    IN_AADHAAR = "IN_AADHAAR"  # 12-digit Aadhaar number (XXXX XXXX XXXX)
    IN_PAN = "IN_PAN"  # Permanent Account Number (ABCDE1234F)
    IN_UPI = "IN_UPI"  # UPI address (user@bankname)
    IN_PHONE = "IN_PHONE"  # Indian phone number (+91/91/0 prefix, starts with 6-9)
    IN_IFSC = "IN_IFSC"  # IFSC bank branch code (e.g. HDFC0001234)


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
            PIIEntityType.IN_AADHAAR,
            PIIEntityType.IN_PAN,
            PIIEntityType.IN_UPI,
            PIIEntityType.IN_PHONE,
            PIIEntityType.IN_IFSC,
        }
    )
    redaction_char: str = "*"
    threshold: float = 0.5
    gdpr_mode: bool = False
    hipaa_mode: bool = False
    pci_dss_mode: bool = False
    custom_patterns: Optional[dict] = None
    allow_list: Optional[List[str]] = None
    # When True, the detector MUST NOT fetch spaCy/HuggingFace models from the
    # network at runtime (air-gapped sites). Mirrors OTelConfig.air_gapped; the
    # orchestrator may plumb OTelConfig.air_gapped into this field.
    air_gapped: bool = field(default_factory=lambda: _env_bool("GENAI_AIR_GAPPED", "false"))
    # Upper bound (chars) for redacted-text attributes exported on spans so a
    # near-full copy of the prompt/response cannot bloat the span. 0 = unlimited.
    # Mirrors OTelConfig.content_max_length (may be plumbed from it).
    content_max_length: int = field(
        default_factory=lambda: int(os.getenv("GENAI_CONTENT_MAX_LENGTH", "200"))
    )

    def __post_init__(self):
        """Validate configuration and apply compliance modes."""
        # A locked-down deployment profile forces air-gapped behaviour.
        if _is_hardening_profile():
            self.air_gapped = True
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
        block_on_detection: Annotate the span (status ERROR + ``*.blocked``
            attribute) when toxic content is detected. NOTE: this is a
            post-call detective annotation only - it does NOT interrupt or
            block the in-flight LLM call. Real pre-call blocking is handled by
            a separate hook that calls ``detect()`` before the provider call.
        allow_external_egress: When False, the detector MUST NOT call the
            external Google Perspective API (customer text must not leave the
            application boundary). Mirrors OTelConfig.allow_external_egress.
        air_gapped: When True, the detector MUST NOT download Detoxify weights
            at runtime. Mirrors OTelConfig.air_gapped.
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
    api_timeout: int = 30
    # Egress / air-gap posture. These mirror the same-named fields on OTelConfig
    # and default from the shared env vars so a hardened deployment is protected
    # even if the orchestrator does not explicitly plumb them through.
    allow_external_egress: bool = field(
        default_factory=lambda: _env_bool("GENAI_ALLOW_EXTERNAL_EGRESS", "true")
    )
    air_gapped: bool = field(default_factory=lambda: _env_bool("GENAI_AIR_GAPPED", "false"))

    def __post_init__(self):
        """Validate configuration and apply the deployment hardening profile."""
        # A locked-down deployment profile (bank/bfsi/strict) blocks all
        # third-party egress and runtime downloads. Applied BEFORE validation so
        # that disabling Perspective does not then require an API key.
        if _is_hardening_profile():
            self.allow_external_egress = False
            self.air_gapped = True
            self.use_perspective_api = False

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
        block_on_detection: Annotate the span (status ERROR + ``*.blocked``
            attribute) when bias is detected. Post-call detective annotation
            only - it does NOT interrupt the in-flight call.
    """

    enabled: bool = False
    threshold: float = 0.4
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
        block_on_detection: Annotate the span (status ERROR + ``*.blocked``
            attribute) when an injection attempt is detected. Post-call
            detective annotation only - it does NOT interrupt the in-flight call.
        log_attempts: Log all injection attempts for analysis
    """

    enabled: bool = False
    threshold: float = 0.5
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
        block_on_detection: Annotate the span (status ERROR + ``*.blocked``
            attribute) when a restricted topic is detected. Post-call detective
            annotation only - it does NOT interrupt the in-flight call.
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
