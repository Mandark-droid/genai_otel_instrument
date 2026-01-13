# GenAI Evaluation Metrics Dashboard

## Overview

The **GenAI Evaluation Metrics Dashboard** provides comprehensive visualization and monitoring of all evaluation features in genai-otel-instrument v0.1.30+, including:

- **PII Detection** - Monitor personally identifiable information in prompts and responses
- **Toxicity Detection** - Track toxic and harmful content
- **Bias Detection** - Identify demographic and other biases
- **Prompt Injection Detection** - Security monitoring for injection attacks
- **Restricted Topics** - Track violations of content policies
- **Hallucination Detection** - Monitor factual accuracy and groundedness

## Prerequisites

1. **OpenSearch setup with evaluation fields**:
   ```bash
   cd examples/demo
   bash opensearch-setup.sh
   ```

2. **Enable evaluation metrics** in your application:
   ```python
   from genai_otel import instrument
   from genai_otel.evaluation import PIIConfig, ToxicityConfig, BiasConfig

   instrument(
       enable_pii_detection=True,
       pii_config=PIIConfig(mode="detect", threshold=0.5),
       enable_toxicity_detection=True,
       toxicity_config=ToxicityConfig(threshold=0.7),
       enable_bias_detection=True,
       bias_config=BiasConfig(threshold=0.4)
   )
   ```

3. **Grafana** with OpenSearch datasource configured (already provisioned in demo setup)

## Dashboard Sections

### 1. Evaluation Metrics Overview (Top Row)

Six key metrics providing real-time counts:

- **PII Detections** - Count of PII found in prompts or responses (Red: ≥10, Yellow: ≥1)
- **Toxicity Detections** - Count of toxic content detected (Red: ≥5, Orange: ≥1)
- **Bias Detections** - Count of biased content detected (Red: ≥5, Yellow: ≥1)
- **Prompt Injection Attempts** - Security alert count (Red: ≥1)
- **Hallucination Detections** - Count of potential hallucinations (Orange: ≥10, Yellow: ≥1)
- **Blocked Requests** - Total requests blocked by any evaluation filter (Red: ≥1)

**Use Case**: Quick health check - see if any evaluation issues are occurring right now.

### 2. Detection Trends Over Time

Multi-line time series graph showing trends for all evaluation types:

- Color-coded lines for each detection type
- Smooth interpolation for trend visualization
- Interactive legend with sum and max statistics
- Auto-refresh every 5 seconds

**Use Case**: Identify patterns, spikes, or trends in evaluation metrics over time.

**Example Insights**:
- Spike in PII detections → User input validation issue
- Increasing toxicity → Content moderation needed
- Prompt injection attempts → Security incident investigation

### 3. PII Detection Details

Four visualizations focused on PII:

#### a. Average PII Scores (Gauges)
- **Prompt Score Gauge** - Average confidence score for PII in prompts (0-1)
- **Response Score Gauge** - Average confidence score for PII in responses (0-1)
- Color thresholds: Green (0-0.3), Yellow (0.3-0.7), Red (0.7-1.0)

#### b. PII by Provider & Model (Pie Charts)
- Distribution of PII detections across LLM providers
- Distribution across specific models
- Click segments for drill-down

#### c. Recent PII Detections (Table)
Columns:
- **Trace ID** - Clickable link to Jaeger trace
- **Timestamp** - When detection occurred
- **Provider** - LLM provider (openai, anthropic, etc.)
- **Model** - Specific model used
- **Prompt PII** - Boolean detection in prompt (red if true)
- **Prompt Score** - Confidence score (0-1, color gradient)
- **Prompt Entities** - Number of PII entities found
- **Prompt Blocked** - Whether request was blocked
- **Response PII** - Boolean detection in response
- **Response Score** - Confidence score (0-1, color gradient)
- **Response Entities** - Number of PII entities found
- **Response Blocked** - Whether response was blocked

**Use Case**:
- Audit PII exposures
- Track which models/providers have PII issues
- Investigate specific incidents via Jaeger trace links

### 4. Toxicity & Bias Detection

Two time series bar charts:

#### a. Toxicity Detections Over Time
- Separate bars for prompt vs response toxicity
- Stacked view option available
- Sum and max statistics in legend

#### b. Bias Detections Over Time
- Separate bars for prompt vs response bias
- Compare input bias vs model bias
- Time-based pattern analysis

**Use Case**:
- Compare prompt toxicity (user input) vs response toxicity (model output)
- Identify if bias is in user prompts or model responses
- Track effectiveness of content moderation

### 5. Security: Prompt Injection & Restricted Topics

Table showing all prompt injection attempts with:

- **Trace ID** - Link to full trace
- **Timestamp** - When attempt occurred
- **Provider & Model** - Where injection was attempted
- **Detected** - Boolean (red background if true)
- **Score** - Injection confidence score (0-1, color gradient)
- **Types** - Injection technique types detected
- **Blocked** - Action taken (red if blocked, green if allowed)

**Use Case**:
- Security incident response
- Track attack patterns
- Monitor effectiveness of injection detection
- Compliance and audit trails

### 6. Hallucination Detection

Table showing potential hallucinations with:

- **Trace ID** - Link to full trace
- **Timestamp** - When detected
- **Provider & Model** - Which model hallucinated
- **Detected** - Boolean indicator
- **Score** - Hallucination confidence (0-1, color gradient)
- **Citations** - Number of citations in response
- **Hedge Words** - Count of uncertainty indicators
- **Claims** - Number of factual claims made

**Use Case**:
- Quality assurance for RAG applications
- Identify models with hallucination issues
- Track citation quality
- Compare hallucination rates across providers

## Query Examples

The dashboard uses these OpenSearch queries - you can modify them for custom views:

```bash
# Find all PII detections
evaluation_pii_prompt_detected:true OR evaluation_pii_response_detected:true

# High toxicity only (score > 0.8)
evaluation_toxicity_response_max_score:[0.8 TO *]

# Blocked requests
evaluation_pii_prompt_blocked:true OR evaluation_prompt_injection_blocked:true

# Hallucinations with low citation count
evaluation_hallucination_response_detected:true AND evaluation_hallucination_response_citations:[0 TO 2]

# Specific provider PII issues
gen_ai_system:openai AND evaluation_pii_response_detected:true

# High-confidence prompt injection
evaluation_prompt_injection_detected:true AND evaluation_prompt_injection_score:[0.7 TO *]
```

## Alerts and Monitoring

### Recommended Alerts

1. **Critical PII Exposure**
   - Query: `evaluation_pii_response_detected:true AND evaluation_pii_response_score:[0.7 TO *]`
   - Threshold: 1 occurrence in 5 minutes
   - Action: Page security team

2. **Prompt Injection Attack**
   - Query: `evaluation_prompt_injection_detected:true`
   - Threshold: 3 occurrences in 1 minute
   - Action: Alert security team

3. **High Toxicity Rate**
   - Query: `evaluation_toxicity_response_detected:true`
   - Threshold: > 5% of requests in 15 minutes
   - Action: Enable stricter content filtering

4. **Hallucination Spike**
   - Query: `evaluation_hallucination_response_detected:true`
   - Threshold: > 20% of requests in 1 hour
   - Action: Review model/RAG configuration

### Setting Up Alerts in Grafana

1. Click on any panel
2. Select "More" → "New alert rule"
3. Configure query and threshold
4. Add notification channel (Slack, PagerDuty, email)

## Time Range & Refresh

- **Default Time Range**: Last 6 hours
- **Auto Refresh**: 5 seconds
- **Adjustable**: Use time picker in top-right corner

## Troubleshooting

### No Data Showing

1. **Check OpenSearch pipeline**:
   ```bash
   curl http://localhost:9200/_ingest/pipeline/genai-ingest-pipeline
   ```

2. **Verify evaluation metrics are enabled**:
   ```python
   # In your app
   instrument(enable_pii_detection=True, enable_toxicity_detection=True)
   ```

3. **Check index exists**:
   ```bash
   curl http://localhost:9200/jaeger-span-*/_search?size=1
   ```

4. **Verify evaluation fields**:
   ```bash
   curl http://localhost:9200/jaeger-span-*/_search?q=evaluation_pii_prompt_detected:*
   ```

### Incorrect Scores

- Ensure OpenSearch ingest pipeline version is 2.0+
- Check that fields are mapped as `double` type
- Verify evaluation detectors are properly initialized

### Missing Panels

- Refresh Grafana dashboards: Configuration → Data Sources → OpenSearch → Save & Test
- Check dashboard JSON for syntax errors
- Verify dashboard provisioning is enabled

## Customization

### Adding Custom Panels

1. Click "Add" → "Visualization"
2. Select OpenSearch datasource
3. Write query using evaluation fields
4. Save to dashboard

### Custom Evaluation Queries

All evaluation fields follow this naming pattern:
- `evaluation_{detector}_{location}_{metric}`
- Examples:
  - `evaluation_pii_prompt_detected` (boolean)
  - `evaluation_toxicity_response_max_score` (double)
  - `evaluation_bias_prompt_entity_count` (integer)
  - `evaluation_prompt_injection_types` (keyword)
  - `evaluation_hallucination_response_indicators` (keyword)

### Exporting Data

Use Grafana's export features:
- CSV export from tables
- PNG export from graphs
- JSON export of raw data
- API access for automation

## Related Documentation

- [Evaluation Metrics Guide](../../../../docs/evaluation-metrics.md)
- [OpenSearch Setup](../../opensearch-setup.sh)
- [Main Dashboard](./genai-opensearch-traces-dashboard.json)
- [Metrics Dashboard](./genai-metrics-dashboard.json)

## Version History

- **v1.0** (2026-01-13) - Initial release with full evaluation metrics support
  - 6 detection types
  - 16 panels across 6 sections
  - Auto-refresh and real-time monitoring
  - Jaeger trace integration

## Support

For issues or questions:
- GitHub Issues: https://github.com/Mandark-droid/genai_otel_instrument/issues
- Documentation: Check CLAUDE.md for development guidelines
- Examples: See `examples/*/evaluation_*.py` for usage examples
