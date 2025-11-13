# Technical Design Document: v0.2.0 - Evaluation & Safety Features

**Version**: 0.2.0
**Status**: Design Phase
**Authors**: Development Team
**Date**: 2025-11-13

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Component Design](#component-design)
4. [API Design](#api-design)
5. [Integration Points](#integration-points)
6. [Metrics & Attributes](#metrics--attributes)
7. [Implementation Plan](#implementation-plan)
8. [Testing Strategy](#testing-strategy)
9. [Migration & Compatibility](#migration--compatibility)

---

## Executive Summary

Version 0.2.0 introduces **evaluation metrics** and **safety guardrails** as opt-in features for GenAI observability. These capabilities enable:

- **Real-time content safety monitoring** (toxicity, bias, PII)
- **Security protection** (prompt injection, restricted topics)
- **Quality evaluation** (hallucination detection)
- **Compliance support** (GDPR, HIPAA, PCI-DSS)

All features are:
- ✅ Opt-in via configuration
- ✅ Zero-code for basic usage
- ✅ Extensible for custom implementations
- ✅ Compatible with existing instrumentation

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application Code                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              GenAI OTEL Instrumentation                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ LLM Provider │  │  Framework   │  │   Evaluation │      │
│  │Instrumentors │  │Instrumentors │  │  & Safety    │      │
│  │  (OpenAI,    │  │ (LangChain,  │  │  Processors  │◄─┐   │
│  │  Anthropic)  │  │  CrewAI)     │  │              │  │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │   │
│         │                  │                  │          │   │
│         └──────────────────┴──────────────────┘          │   │
│                            │                             │   │
│                            ▼                             │   │
│  ┌─────────────────────────────────────────────────┐    │   │
│  │      Evaluation & Safety Span Processor         │    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────┐ │    │   │
│  │  │PII Detector│  │ Toxicity   │  │   Bias    │ │    │   │
│  │  │  (Presidio)│  │ (Perspective│  │(Fairlearn)│ │    │   │
│  │  └────────────┘  └────────────┘  └───────────┘ │    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────┐ │    │   │
│  │  │  Prompt    │  │ Restricted │  │Hallucin.  │ │    │   │
│  │  │ Injection  │  │   Topics   │  │ Detection │ │    │   │
│  │  └────────────┘  └────────────┘  └───────────┘ │    │   │
│  └─────────────────────────────────────────────────┘    │   │
│                            │                             │   │
└────────────────────────────┼─────────────────────────────┘   │
                             │                                 │
                             ▼                                 │
         ┌───────────────────────────────────┐                │
         │   OpenTelemetry TracerProvider    │                │
         │  ┌─────────────────────────────┐  │                │
         │  │  BatchSpanProcessor          │  │                │
         │  │  ├─ Enriched Spans           │  │                │
         │  │  ├─ Safety Attributes        │  │                │
         │  │  └─ Evaluation Scores        │  │                │
         │  └─────────────────────────────┘  │                │
         └───────────────────────────────────┘                │
                             │                                 │
                             ▼                                 │
         ┌───────────────────────────────────┐                │
         │   OpenTelemetry MeterProvider     │◄───────────────┘
         │  ┌─────────────────────────────┐  │
         │  │  Metrics                     │  │
         │  │  ├─ gen_ai.eval.*            │  │
         │  │  ├─ gen_ai.guardrail.*       │  │
         │  │  └─ Histogram/Counter        │  │
         │  └─────────────────────────────┘  │
         └───────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────┐
         │    OTLP Exporter / Collector      │
         └───────────────────────────────────┘
```

### Key Design Principles

1. **Span Processing Pipeline**: All evaluation/safety checks run as span processors
2. **Async-First**: Detectors run asynchronously to minimize latency impact
3. **Configurable**: All features opt-in with sensible defaults
4. **Extensible**: Plugin architecture for custom detectors
5. **Performance**: Cached models, batch processing, configurable sampling

---

## Component Design

### 1. PII Detection (Sensitive Information Protection)

**Library**: Microsoft Presidio + regex patterns
**Purpose**: Detect and handle PII in LLM inputs/outputs
**Priority**: HIGHEST (compliance-critical)

#### Architecture

```python
class PIIDetector:
    """Detects PII in text using Presidio analyzer."""

    def __init__(self, config: PIIConfig):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.config = config
        self.entities = config.entity_types or DEFAULT_ENTITIES

    def detect(self, text: str) -> PIIDetectionResult:
        """Detect PII in text."""
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language='en'
        )
        return PIIDetectionResult(
            detected=len(results) > 0,
            entities=[{
                'type': r.entity_type,
                'score': r.score,
                'start': r.start,
                'end': r.end
            } for r in results]
        )

    def redact(self, text: str) -> str:
        """Redact PII from text."""
        return self.anonymizer.anonymize(
            text=text,
            analyzer_results=self.analyzer.analyze(text, self.entities)
        ).text
```

#### Configuration

```python
@dataclass
class PIIConfig:
    """Configuration for PII detection."""

    enabled: bool = False
    mode: str = "detect"  # detect, redact, block
    entity_types: List[str] = None  # None = all types
    confidence_threshold: float = 0.5
    block_on_detection: bool = False
    redact_output: bool = False

    # Compliance modes
    gdpr_mode: bool = False
    hipaa_mode: bool = False
    pci_dss_mode: bool = False
```

#### Default Entity Types

- EMAIL_ADDRESS
- PHONE_NUMBER
- CREDIT_CARD
- SSN (Social Security Number)
- IP_ADDRESS
- PERSON (Names)
- LOCATION
- DATE_TIME
- MEDICAL_LICENSE
- US_PASSPORT
- US_DRIVER_LICENSE
- IBAN_CODE
- CRYPTO (Cryptocurrency wallets)

#### Integration Point

```python
class EvaluationSpanProcessor(SpanProcessor):
    """Main span processor for evaluation/safety features."""

    def __init__(self, config: OTelConfig):
        self.pii_detector = PIIDetector(config.pii_config) if config.enable_pii_detection else None
        # ... other detectors

    def on_end(self, span: ReadableSpan):
        """Process span with all enabled detectors."""

        # Extract LLM input/output from span
        llm_input = self._extract_input(span)
        llm_output = self._extract_output(span)

        # Run PII detection
        if self.pii_detector:
            input_pii = self.pii_detector.detect(llm_input)
            output_pii = self.pii_detector.detect(llm_output)

            # Add attributes
            span.set_attribute("gen_ai.pii.input.detected", input_pii.detected)
            span.set_attribute("gen_ai.pii.output.detected", output_pii.detected)

            if input_pii.detected:
                span.set_attribute("gen_ai.pii.input.entities",
                                 [e['type'] for e in input_pii.entities])

            # Record metric
            self.meter.create_counter("gen_ai.pii.detections").add(
                1 if output_pii.detected else 0,
                {"entity_types": ",".join([e['type'] for e in output_pii.entities])}
            )

            # Handle based on mode
            if self.pii_detector.config.mode == "block" and output_pii.detected:
                span.set_attribute("gen_ai.pii.blocked", True)
                raise PIIDetectionError("PII detected in output")
```

---

### 2. Toxicity Detection

**Library**: Perspective API (Google) + local detoxify model (optional)
**Purpose**: Detect toxic, harmful, or offensive content
**Priority**: HIGH (content moderation)

#### Architecture

```python
class ToxicityDetector:
    """Detects toxic content using Perspective API."""

    def __init__(self, config: ToxicityConfig):
        self.api_key = config.perspective_api_key
        self.client = perspectiveapi.Client(self.api_key)
        self.threshold = config.threshold
        self.attributes = config.attributes or DEFAULT_ATTRIBUTES
        # Fallback to local model if API unavailable
        self.local_model = Detoxify('original') if config.use_local_fallback else None

    async def detect(self, text: str) -> ToxicityResult:
        """Detect toxicity in text."""
        try:
            # Try Perspective API first
            response = await self.client.analyze(
                text=text,
                requested_attributes=self.attributes,
                languages=['en']
            )

            scores = {
                attr: response.attribute_scores[attr].summary_score.value
                for attr in self.attributes
            }

            max_score = max(scores.values())

            return ToxicityResult(
                toxic=max_score >= self.threshold,
                scores=scores,
                categories=[k for k, v in scores.items() if v >= self.threshold]
            )

        except Exception as e:
            # Fallback to local model
            if self.local_model:
                local_scores = self.local_model.predict(text)
                return ToxicityResult(
                    toxic=local_scores['toxicity'] >= self.threshold,
                    scores=local_scores,
                    categories=['toxicity'] if local_scores['toxicity'] >= self.threshold else []
                )
            raise
```

#### Perspective API Attributes

- TOXICITY
- SEVERE_TOXICITY
- IDENTITY_ATTACK
- INSULT
- PROFANITY
- THREAT
- SEXUALLY_EXPLICIT
- FLIRTATION

#### Configuration

```python
@dataclass
class ToxicityConfig:
    """Configuration for toxicity detection."""

    enabled: bool = False
    perspective_api_key: Optional[str] = None
    threshold: float = 0.5
    attributes: List[str] = None  # None = all attributes
    use_local_fallback: bool = True
    block_on_detection: bool = False
    severity_levels: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.3,
        'medium': 0.5,
        'high': 0.7,
        'critical': 0.9
    })
```

---

### 3. Bias Detection

**Library**: Fairlearn + custom bias patterns
**Purpose**: Detect demographic and other biases in LLM outputs
**Priority**: MEDIUM-HIGH (fairness)

#### Architecture

```python
class BiasDetector:
    """Detects bias in text using pattern matching and ML models."""

    BIAS_CATEGORIES = {
        'gender': ['he', 'she', 'him', 'her', 'male', 'female'],
        'racial': ['black', 'white', 'asian', 'hispanic', 'race'],
        'age': ['young', 'old', 'elderly', 'millennial', 'boomer'],
        'religious': ['christian', 'muslim', 'jewish', 'hindu', 'atheist'],
        'political': ['liberal', 'conservative', 'democrat', 'republican'],
        'socioeconomic': ['poor', 'rich', 'wealthy', 'disadvantaged']
    }

    def __init__(self, config: BiasConfig):
        self.config = config
        self.nlp = spacy.load('en_core_web_sm')
        self.bias_patterns = self._compile_patterns()

    def detect(self, text: str, context: Optional[str] = None) -> BiasResult:
        """Detect bias in text."""
        doc = self.nlp(text)

        detected_biases = {}
        for category, terms in self.BIAS_CATEGORIES.items():
            if not self.config.categories or category in self.config.categories:
                score = self._calculate_bias_score(doc, terms, category)
                if score >= self.config.threshold:
                    detected_biases[category] = score

        return BiasResult(
            biased=len(detected_biases) > 0,
            scores=detected_biases,
            categories=list(detected_biases.keys())
        )

    def _calculate_bias_score(self, doc, terms, category):
        """Calculate bias score for a category."""
        # Pattern-based detection
        pattern_score = sum(1 for token in doc if token.text.lower() in terms)

        # Context analysis (sentiment around bias terms)
        context_score = self._analyze_context(doc, terms)

        # Combine scores
        final_score = min((pattern_score * 0.3 + context_score * 0.7) / 10, 1.0)
        return final_score
```

#### Configuration

```python
@dataclass
class BiasConfig:
    """Configuration for bias detection."""

    enabled: bool = False
    threshold: float = 0.7
    categories: List[str] = None  # None = all categories
    use_fairlearn: bool = False  # Use Fairlearn for advanced analysis
    block_on_detection: bool = False
```

---

### 4. Prompt Injection Detection

**Library**: Custom patterns + ML classifier
**Purpose**: Detect and block prompt injection attacks
**Priority**: HIGH (security)

#### Architecture

```python
class PromptInjectionDetector:
    """Detects prompt injection and jailbreaking attempts."""

    # Known jailbreaking patterns
    JAILBREAK_PATTERNS = [
        r"ignore (previous|above) (instructions|prompt)",
        r"you are now|from now on",
        r"developer mode|god mode|admin mode",
        r"DAN|do anything now",
        r"pretend (you|to) (are|be)",
        r"roleplay as",
        r"output in (code|script|programming)",
        r"system:\s*jailbreak",
    ]

    def __init__(self, config: PromptInjectionConfig):
        self.config = config
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]

        # Optional: Load ML-based classifier
        if config.use_ml_classifier:
            self.classifier = self._load_classifier()
        else:
            self.classifier = None

    def detect(self, prompt: str) -> InjectionResult:
        """Detect prompt injection in input."""

        # Pattern-based detection
        pattern_matches = []
        for pattern in self.patterns:
            matches = pattern.findall(prompt)
            if matches:
                pattern_matches.extend(matches)

        pattern_score = min(len(pattern_matches) / 3, 1.0)

        # ML-based detection (optional)
        ml_score = 0.0
        if self.classifier:
            ml_score = self.classifier.predict_proba([prompt])[0][1]

        # Combine scores
        final_score = max(pattern_score, ml_score)

        return InjectionResult(
            detected=final_score >= self.config.threshold,
            score=final_score,
            method='pattern' if pattern_score > ml_score else 'ml',
            matched_patterns=pattern_matches[:5]  # Limit to 5
        )
```

#### Configuration

```python
@dataclass
class PromptInjectionConfig:
    """Configuration for prompt injection detection."""

    enabled: bool = False
    threshold: float = 0.7
    use_ml_classifier: bool = False
    block_on_detection: bool = True  # Default to blocking
    custom_patterns: List[str] = field(default_factory=list)
```

---

### 5. Restricted Topics

**Library**: Zero-shot classification (facebook/bart-large-mnli)
**Purpose**: Block content in restricted topic areas
**Priority**: MEDIUM (content filtering)

#### Architecture

```python
class RestrictedTopicsDetector:
    """Detects restricted topics using zero-shot classification."""

    DEFAULT_TOPICS = {
        'medical_advice': 'medical diagnosis or treatment recommendations',
        'legal_advice': 'legal advice or legal opinions',
        'financial_advice': 'investment or financial advice',
        'harmful_instructions': 'instructions for harmful or illegal activities',
        'adult_content': 'sexual or adult content',
        'violence': 'graphic violence or gore'
    }

    def __init__(self, config: RestrictedTopicsConfig):
        self.config = config
        self.topics = config.restricted_topics or self.DEFAULT_TOPICS

        # Load zero-shot classifier
        from transformers import pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def detect(self, text: str) -> TopicResult:
        """Detect restricted topics in text."""

        # Prepare candidate labels
        labels = list(self.topics.values())

        # Classify
        result = self.classifier(text, candidate_labels=labels, multi_label=True)

        # Extract scores
        topic_scores = {
            topic_id: score
            for topic_id, score in zip(self.topics.keys(), result['scores'])
            if score >= self.config.threshold
        }

        return TopicResult(
            restricted=len(topic_scores) > 0,
            topics=list(topic_scores.keys()),
            scores=topic_scores
        )
```

#### Configuration

```python
@dataclass
class RestrictedTopicsConfig:
    """Configuration for restricted topics detection."""

    enabled: bool = False
    restricted_topics: Dict[str, str] = None  # topic_id: description
    threshold: float = 0.7
    block_on_detection: bool = True
```

---

### 6. Hallucination Detection

**Library**: SelfCheckGPT + citation validation
**Purpose**: Detect factual inconsistencies and hallucinations
**Priority**: MEDIUM-HIGH (quality)

#### Architecture

```python
class HallucinationDetector:
    """Detects hallucinations using multiple strategies."""

    def __init__(self, config: HallucinationConfig):
        self.config = config

        # Strategy 1: SelfCheckGPT (consistency check)
        if config.use_selfcheck:
            from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG
            self.selfcheck = SelfCheckMQAG()

        # Strategy 2: Citation validation (for RAG)
        self.citation_validator = CitationValidator() if config.validate_citations else None

        # Strategy 3: Fact-checking against context
        self.fact_checker = FactChecker() if config.check_facts else None

    def detect(self,
               output: str,
               context: Optional[str] = None,
               citations: Optional[List[str]] = None) -> HallucinationResult:
        """Detect hallucinations in LLM output."""

        scores = {}

        # Self-consistency check
        if self.selfcheck:
            consistency_score = self._check_consistency(output)
            scores['consistency'] = consistency_score

        # Citation validation
        if self.citation_validator and citations:
            citation_score = self.citation_validator.validate(output, citations)
            scores['citations'] = citation_score

        # Fact-checking against context
        if self.fact_checker and context:
            fact_score = self.fact_checker.check(output, context)
            scores['factuality'] = fact_score

        # Calculate overall hallucination probability
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        hallucination_prob = 1.0 - avg_score  # Invert: low score = high hallucination

        return HallucinationResult(
            hallucinated=hallucination_prob >= self.config.threshold,
            probability=hallucination_prob,
            scores=scores,
            confidence=1.0 - abs(hallucination_prob - 0.5) * 2  # Confidence in detection
        )
```

#### Configuration

```python
@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection."""

    enabled: bool = False
    threshold: float = 0.8
    use_selfcheck: bool = True
    validate_citations: bool = True
    check_facts: bool = True
    block_on_detection: bool = False  # Usually warn, not block
```

---

## API Design

### Configuration API

```python
from genai_otel import instrument

# Basic usage - all features disabled by default
instrument(
    service_name="my-app",
    endpoint="http://localhost:4318"
)

# Enable specific features
instrument(
    service_name="my-app",
    endpoint="http://localhost:4318",

    # PII Detection
    enable_pii_detection=True,
    pii_mode="redact",  # detect, redact, block
    pii_entity_types=["EMAIL", "PHONE", "SSN", "CREDIT_CARD"],
    pii_gdpr_mode=True,

    # Toxicity Detection
    enable_toxicity_detection=True,
    toxicity_threshold=0.5,
    toxicity_api_key="YOUR_PERSPECTIVE_API_KEY",

    # Bias Detection
    enable_bias_detection=True,
    bias_threshold=0.7,
    bias_categories=["gender", "racial", "age"],

    # Prompt Injection Detection
    enable_prompt_injection_detection=True,
    prompt_injection_threshold=0.7,
    prompt_injection_block=True,

    # Restricted Topics
    enable_restricted_topics=True,
    restricted_topics=["medical_advice", "legal_advice"],

    # Hallucination Detection
    enable_hallucination_detection=True,
    hallucination_threshold=0.8,
    hallucination_validate_citations=True
)
```

### Programmatic API

```python
from genai_otel.evaluation import (
    PIIDetector,
    ToxicityDetector,
    BiasDetector,
    PromptInjectionDetector,
    RestrictedTopicsDetector,
    HallucinationDetector
)

# Use detectors independently
pii_detector = PIIDetector(
    mode="redact",
    entity_types=["EMAIL", "PHONE"]
)

result = pii_detector.detect("My email is john@example.com")
# PIIDetectionResult(detected=True, entities=[{'type': 'EMAIL', 'score': 0.95, ...}])

redacted = pii_detector.redact("My email is john@example.com")
# "My email is <EMAIL>"
```

### Callback API

```python
from genai_otel import instrument
from genai_otel.evaluation import GuardrailViolation

def handle_violation(violation: GuardrailViolation):
    """Custom handler for guardrail violations."""
    print(f"Violation detected: {violation.type}")
    print(f"Severity: {violation.severity}")
    print(f"Score: {violation.score}")

    # Custom logic: log to security system, send alert, etc.
    if violation.severity == "critical":
        security_system.alert(violation)

instrument(
    enable_pii_detection=True,
    enable_toxicity_detection=True,
    on_guardrail_violation=handle_violation
)
```

---

## Metrics & Attributes

### New Metrics

```python
# PII Detection
gen_ai.pii.detections              # Counter - PII detections by entity type
gen_ai.pii.blocked                 # Counter - Blocked requests due to PII
gen_ai.pii.redacted                # Counter - PII redactions performed

# Toxicity Detection
gen_ai.eval.toxicity_score         # Histogram - Toxicity scores (0-1)
gen_ai.guardrail.toxicity_blocked  # Counter - Blocked toxic content

# Bias Detection
gen_ai.eval.bias_score             # Histogram - Bias scores by category
gen_ai.eval.bias_violations        # Counter - Bias threshold violations

# Prompt Injection
gen_ai.guardrail.injection_detected # Counter - Injection attempts
gen_ai.guardrail.injection_blocked  # Counter - Blocked injections

# Restricted Topics
gen_ai.guardrail.topic_blocked     # Counter - Blocked topics by type

# Hallucination
gen_ai.eval.hallucination_score    # Histogram - Hallucination probability
gen_ai.eval.hallucination_detected # Counter - Detected hallucinations
```

### New Span Attributes

```python
# PII Attributes
gen_ai.pii.input.detected: bool
gen_ai.pii.output.detected: bool
gen_ai.pii.input.entities: List[str]
gen_ai.pii.output.entities: List[str]
gen_ai.pii.redacted: bool
gen_ai.pii.blocked: bool

# Toxicity Attributes
gen_ai.eval.toxicity.score: float
gen_ai.eval.toxicity.categories: List[str]
gen_ai.eval.toxicity.severity: str  # low, medium, high, critical
gen_ai.guardrail.toxicity.blocked: bool

# Bias Attributes
gen_ai.eval.bias.detected: bool
gen_ai.eval.bias.score: float
gen_ai.eval.bias.categories: List[str]

# Prompt Injection Attributes
gen_ai.guardrail.injection.detected: bool
gen_ai.guardrail.injection.score: float
gen_ai.guardrail.injection.method: str  # pattern, ml
gen_ai.guardrail.injection.blocked: bool

# Restricted Topics Attributes
gen_ai.guardrail.topic.restricted: bool
gen_ai.guardrail.topic.detected: List[str]
gen_ai.guardrail.topic.blocked: bool

# Hallucination Attributes
gen_ai.eval.hallucination.detected: bool
gen_ai.eval.hallucination.probability: float
gen_ai.eval.hallucination.confidence: float
gen_ai.eval.hallucination.method: str  # consistency, citations, factuality
```

---

## Integration Points

### OTelConfig Extension

```python
@dataclass
class OTelConfig:
    """Extended configuration for v0.2.0."""

    # ... existing fields ...

    # PII Detection
    enable_pii_detection: bool = False
    pii_mode: str = "detect"
    pii_entity_types: Optional[List[str]] = None
    pii_confidence_threshold: float = 0.5
    pii_block_on_detection: bool = False
    pii_redact_output: bool = False
    pii_gdpr_mode: bool = False
    pii_hipaa_mode: bool = False
    pii_pci_dss_mode: bool = False

    # Toxicity Detection
    enable_toxicity_detection: bool = False
    toxicity_threshold: float = 0.5
    toxicity_api_key: Optional[str] = None
    toxicity_attributes: Optional[List[str]] = None
    toxicity_block_on_detection: bool = False

    # Bias Detection
    enable_bias_detection: bool = False
    bias_threshold: float = 0.7
    bias_categories: Optional[List[str]] = None
    bias_block_on_detection: bool = False

    # Prompt Injection Detection
    enable_prompt_injection_detection: bool = False
    prompt_injection_threshold: float = 0.7
    prompt_injection_block: bool = True
    prompt_injection_custom_patterns: Optional[List[str]] = None

    # Restricted Topics
    enable_restricted_topics: bool = False
    restricted_topics: Optional[List[str]] = None
    restricted_topics_threshold: float = 0.7
    restricted_topics_block: bool = True

    # Hallucination Detection
    enable_hallucination_detection: bool = False
    hallucination_threshold: float = 0.8
    hallucination_validate_citations: bool = True
    hallucination_check_facts: bool = True

    # Callbacks
    on_guardrail_violation: Optional[Callable] = None
```

### SpanProcessor Integration

```python
# In auto_instrument.py

def setup_auto_instrumentation(config: OTelConfig):
    """Set up OpenTelemetry with evaluation/safety features."""

    # ... existing setup ...

    # Add evaluation span processor if any feature enabled
    if _any_evaluation_feature_enabled(config):
        eval_processor = EvaluationSpanProcessor(config)
        tracer_provider.add_span_processor(eval_processor)
        logger.info("Evaluation & Safety span processor added")

    # ... rest of setup ...

def _any_evaluation_feature_enabled(config: OTelConfig) -> bool:
    """Check if any evaluation/safety feature is enabled."""
    return any([
        config.enable_pii_detection,
        config.enable_toxicity_detection,
        config.enable_bias_detection,
        config.enable_prompt_injection_detection,
        config.enable_restricted_topics,
        config.enable_hallucination_detection
    ])
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

**Week 1: PII Detection**
- [ ] Day 1-2: Set up Presidio integration
- [ ] Day 3: Implement PIIDetector class
- [ ] Day 4: Integrate with EvaluationSpanProcessor
- [ ] Day 5: Add metrics and attributes
- [ ] Day 6-7: Tests and examples

**Week 2: Toxicity Detection**
- [ ] Day 1-2: Perspective API integration
- [ ] Day 3: Local Detoxify fallback
- [ ] Day 4: ToxicityDetector class
- [ ] Day 5: Integration and metrics
- [ ] Day 6-7: Tests and examples

**Deliverables**:
- ✅ PII detection working (detect, redact, block modes)
- ✅ Toxicity detection working (API + local fallback)
- ✅ 40+ test cases
- ✅ 2 comprehensive examples
- ✅ Metrics dashboard compatible

### Phase 2: Security (Week 3-4)

**Week 3: Prompt Injection Detection**
- [ ] Day 1-2: Pattern-based detection
- [ ] Day 3: ML classifier (optional)
- [ ] Day 4: Integration and blocking
- [ ] Day 5: Metrics
- [ ] Day 6-7: Tests and examples

**Week 4: Bias Detection**
- [ ] Day 1-2: Pattern-based bias detection
- [ ] Day 3: NLP context analysis
- [ ] Day 4: Integration
- [ ] Day 5: Metrics
- [ ] Day 6-7: Tests and examples

**Deliverables**:
- ✅ Prompt injection detection (pattern + ML)
- ✅ Bias detection (6 categories)
- ✅ 30+ test cases
- ✅ Security examples and docs

### Phase 3: Advanced Features (Week 5-6)

**Week 5: Restricted Topics**
- [ ] Day 1-2: Zero-shot classifier setup
- [ ] Day 3: RestrictedTopicsDetector
- [ ] Day 4: Integration
- [ ] Day 5-7: Tests and examples

**Week 6: Hallucination Detection**
- [ ] Day 1-2: SelfCheckGPT integration
- [ ] Day 3: Citation validation
- [ ] Day 4: Fact-checking logic
- [ ] Day 5: Integration
- [ ] Day 6-7: Tests and examples

**Deliverables**:
- ✅ Restricted topics working
- ✅ Hallucination detection (3 strategies)
- ✅ 30+ test cases
- ✅ RAG examples

### Phase 4: Polish & Release (Week 7)

- [ ] Day 1-2: Integration testing
- [ ] Day 3: Performance optimization
- [ ] Day 4: Documentation
- [ ] Day 5: Migration guide
- [ ] Day 6-7: Release prep and testing

---

## Testing Strategy

### Unit Tests

Each detector must have:
- ✅ Initialization tests (with/without config)
- ✅ Detection accuracy tests (true positives/negatives)
- ✅ Threshold behavior tests
- ✅ Mode tests (detect/redact/block)
- ✅ Edge cases (empty input, very long text, special chars)
- ✅ Error handling tests

**Target**: 90%+ code coverage

### Integration Tests

- ✅ Span processor integration
- ✅ Metrics recording
- ✅ Attribute setting
- ✅ Multi-detector coordination
- ✅ Performance impact testing
- ✅ Async execution testing

### Performance Tests

- ✅ Latency impact (target: <50ms per detector)
- ✅ Memory usage (target: <100MB for all detectors)
- ✅ Throughput (target: 100+ requests/sec)
- ✅ Caching effectiveness

### Example Tests

Each feature needs:
- ✅ Basic usage example
- ✅ Advanced configuration example
- ✅ Error handling example
- ✅ Integration example (with LLM calls)

---

## Migration & Compatibility

### Backward Compatibility

✅ **Zero Breaking Changes**
- All new features are opt-in
- Existing instrumentation works unchanged
- No changes to existing metrics/attributes
- No changes to existing API

✅ **Configuration**
- New config fields with sensible defaults (disabled)
- Existing configs continue to work
- Gradual migration path

✅ **Dependencies**
- New dependencies are optional
- Graceful degradation if libraries not installed
- Clear error messages for missing dependencies

### Dependency Management

```python
# Optional dependencies in setup.py
extras_require={
    'evaluation': [
        'presidio-analyzer>=2.2.0',
        'presidio-anonymizer>=2.2.0',
        'detoxify>=0.5.0',
        'spacy>=3.0.0',
        'transformers>=4.0.0',
    ],
    'safety': [
        'google-api-python-client>=2.0.0',  # Perspective API
        'selfcheckgpt>=0.1.0',
    ],
    'all': [
        # All optional dependencies
    ]
}
```

**Installation**:
```bash
# Full v0.2.0 features
pip install genai-otel-instrument[evaluation,safety]

# Or install all extras
pip install genai-otel-instrument[all]
```

### Migration Guide

For users upgrading from v0.1.x to v0.2.0:

1. **No action required** if not using new features
2. **Opt-in to features** by updating config
3. **Install optional dependencies** as needed
4. **Update dashboards** to visualize new metrics (optional)

---

## Security & Privacy Considerations

### Data Handling

- ✅ **No data stored** - all processing is in-memory
- ✅ **No external calls** (except opt-in APIs like Perspective)
- ✅ **PII redaction** before logging/exporting (when enabled)
- ✅ **Configurable sampling** to reduce data exposure

### API Keys

- ✅ Perspective API key via environment variable
- ✅ Never logged or exported
- ✅ Optional - local fallbacks available

### Compliance

- ✅ GDPR mode: stricter PII detection
- ✅ HIPAA mode: medical entity detection
- ✅ PCI-DSS mode: payment card detection
- ✅ Audit logs for blocked content

---

## Success Metrics

### Feature Adoption
- Target: 20% of users enable at least one feature within 3 months
- Target: 10% of users enable PII detection within 1 month

### Performance
- Target: <50ms average latency per detector
- Target: <5% overhead on total request latency
- Target: 99.9% uptime for all detectors

### Accuracy
- PII Detection: >95% precision, >90% recall
- Toxicity Detection: >90% precision, >85% recall
- Prompt Injection: >85% precision, >80% recall

### User Satisfaction
- Target: 4.5+ average rating for new features
- Target: <5% feature removal rate

---

## Future Enhancements (v0.3.0+)

- Custom detector plugins
- Real-time model fine-tuning
- Multi-language support
- Advanced RAG evaluation
- A/B testing framework
- Automated remediation
- Integration with LLM guardrail services

---

## Conclusion

Version 0.2.0 represents a significant enhancement to GenAI observability by adding **evaluation metrics** and **safety guardrails**. The design prioritizes:

1. **Ease of use** - opt-in with sensible defaults
2. **Performance** - minimal latency impact
3. **Extensibility** - plugin architecture for custom detectors
4. **Compliance** - GDPR, HIPAA, PCI-DSS modes
5. **Backward compatibility** - zero breaking changes

Implementation follows a phased approach over 6-7 weeks, with continuous testing and refinement.

---

**Next Steps**: Begin Phase 1 implementation with PII Detection.
