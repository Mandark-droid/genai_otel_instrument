# GenAI Evaluation Alerts - Default Configuration

## Overview

This directory contains pre-configured alert rules for GenAI evaluation metrics that are automatically provisioned when Grafana starts. Alerts are organized by severity levels and routed to appropriate notification channels.

## Alert Files

### 1. `contactpoints.yml` - Notification Channels

Defines where alerts are sent:

| Contact Point | Type | Purpose | Default Recipient |
|--------------|------|---------|-------------------|
| **default-email** | Email | General alerts | admin@example.com |
| **security-team** | Email | Critical security alerts | security@example.com |
| **slack-alerts** | Slack | High/Critical priority | #genai-alerts (disabled by default) |
| **console-only** | Console Log | Low priority/informational | Grafana logs |

**To configure email alerts:**
1. Set `GF_SMTP_ENABLED=true` in docker-compose.yml
2. Configure SMTP settings (see below)
3. Update email addresses in `contactpoints.yml`

**To configure Slack:**
1. Create a Slack Incoming Webhook
2. Uncomment and set `url` in `contactpoints.yml`
3. Restart Grafana

### 2. `policies.yml` - Routing Rules

Routes alerts based on severity:

| Severity | Contact Point | Group Wait | Repeat Interval | Continue? |
|----------|---------------|------------|-----------------|-----------|
| **Critical** | security-team + slack | 10s | 1h | Yes (also to slack) |
| **High** | slack-alerts | 30s | 4h | No |
| **Medium** | default-email | 1m | 12h | No |
| **Low** | console-only | 5m | 24h | No |

### 3. `rules.yml` - Alert Definitions

Contains 11 pre-configured alert rules across 4 rule groups.

## Alert Rules

### Critical Security Alerts (1 minute check interval)

#### 1. Critical PII Exposure
- **Trigger**: PII detected with confidence score ≥ 0.7 in responses
- **Window**: Last 5 minutes
- **For**: 1 minute
- **Action**: Immediate notification to security team
- **Use Case**: Data breach prevention, GDPR/HIPAA compliance

#### 2. Prompt Injection Attack
- **Trigger**: 3+ prompt injection attempts in 1 minute
- **Window**: Last 60 seconds
- **For**: 1 minute
- **Action**: Critical security alert
- **Use Case**: Active attack detection, security incident response

#### 3. Blocked Requests Spike
- **Trigger**: 10+ blocked requests in 5 minutes
- **Window**: Last 5 minutes
- **For**: 2 minutes
- **Action**: Security team notification
- **Use Case**: Attack pattern detection, misconfiguration alerts

### High Priority Quality Alerts (5 minute check interval)

#### 4. High Toxicity Rate
- **Trigger**: >5% of responses contain toxic content
- **Window**: Last 15 minutes
- **For**: 5 minutes
- **Action**: Slack notification
- **Use Case**: Content moderation, model quality monitoring

#### 5. High Bias Rate
- **Trigger**: >10% of responses contain biased content
- **Window**: Last 15 minutes
- **For**: 5 minutes
- **Action**: Slack notification
- **Use Case**: AI fairness monitoring, compliance

### Medium Priority Alerts (15 minute check interval)

#### 6. Hallucination Spike
- **Trigger**: >20% of responses show hallucination indicators
- **Window**: Last 1 hour
- **For**: 15 minutes
- **Action**: Email notification
- **Use Case**: RAG quality monitoring, model validation

#### 7. Recurring PII Detection Pattern
- **Trigger**: 5+ PII detections in 30 minutes
- **Window**: Last 30 minutes
- **For**: 10 minutes
- **Action**: Email notification
- **Use Case**: Input validation issues, data handling review

#### 8. Low Citation Quality
- **Trigger**: 10+ hallucinations with <3 citations
- **Window**: Last 1 hour
- **For**: 15 minutes
- **Action**: Email notification
- **Use Case**: RAG citation configuration, context quality

### Low Priority / Informational (1 hour check interval)

#### 9. Evaluation Metrics Health Check
- **Trigger**: No evaluation metrics detected
- **Window**: Last 1 hour
- **For**: 1 hour
- **Action**: Console log
- **Use Case**: System health monitoring, configuration validation

## Configuration

### Quick Start (Email Alerts)

1. **Update docker-compose.yml** to add SMTP configuration:

```yaml
grafana:
  environment:
    # Enable email alerts
    - GF_SMTP_ENABLED=true
    - GF_SMTP_HOST=smtp.gmail.com:587
    - GF_SMTP_USER=your-email@gmail.com
    - GF_SMTP_PASSWORD=your-app-password
    - GF_SMTP_FROM_ADDRESS=your-email@gmail.com
    - GF_SMTP_FROM_NAME=GenAI Alerts

    # Optional: Skip TLS verification for self-signed certs
    # - GF_SMTP_SKIP_VERIFY=true
```

2. **Update email addresses** in `contactpoints.yml`:

```yaml
contactPoints:
  - orgId: 1
    name: default-email
    receivers:
      - uid: default-email-receiver
        type: email
        settings:
          addresses: "your-team@company.com"  # Change this
```

3. **Restart Grafana**:

```bash
cd examples/demo
docker-compose restart grafana
```

### Slack Integration

1. **Create Slack Incoming Webhook**:
   - Go to https://api.slack.com/apps
   - Create new app → Incoming Webhooks
   - Copy webhook URL

2. **Update contactpoints.yml**:

```yaml
- orgId: 1
  name: slack-alerts
  receivers:
    - uid: slack-receiver
      type: slack
      settings:
        url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"  # Add your webhook
        recipient: "#genai-alerts"  # Uncomment this
        username: "Grafana GenAI Alerts"  # Uncomment this
        icon_emoji: ":warning:"  # Uncomment this
```

3. **Restart Grafana**

### PagerDuty Integration

Add a new contact point for critical alerts:

```yaml
- orgId: 1
  name: pagerduty-critical
  receivers:
    - uid: pagerduty-receiver
      type: pagerduty
      settings:
        integrationKey: "YOUR_PAGERDUTY_INTEGRATION_KEY"
        severity: critical
        class: genai_security
        component: evaluation_metrics
```

Update `policies.yml` to route critical alerts to PagerDuty:

```yaml
routes:
  - receiver: pagerduty-critical
    object_matchers:
      - ['severity', '=', 'critical']
    group_wait: 0s
    repeat_interval: 30m
    continue: true
```

## Customizing Alerts

### Adjust Alert Thresholds

Edit `rules.yml` and modify the evaluator params:

```yaml
# Example: Change PII score threshold from 0.7 to 0.8
conditions:
  - evaluator:
      params:
        - 0.8  # Changed from 0.7
      type: gt
```

### Change Alert Timing

Modify these fields in alert rules:

```yaml
# Check every 30 seconds instead of 1 minute
interval: 30s

# Fire after condition is true for 2 minutes
for: 2m

# Look back 10 minutes instead of 5
relativeTimeRange:
  from: 600  # seconds
  to: 0
```

### Add Custom Alerts

Add a new rule to `rules.yml`:

```yaml
- uid: your_custom_alert
  title: Your Custom Alert Title
  condition: C
  data:
    - refId: A
      datasourceUid: opensearch
      model:
        query: "your_opensearch_query"
        # ... rest of query config
    - refId: C
      datasourceUid: __expr__
      model:
        expression: A
        reducer: last
        type: reduce
  noDataState: NoData
  execErrState: Error
  for: 5m
  annotations:
    summary: Short description
    description: Detailed description with context
  labels:
    severity: medium
    category: custom
  isPaused: false
```

### Disable Specific Alerts

Set `isPaused: true` in the alert rule:

```yaml
- uid: evaluation_metrics_health
  title: Evaluation Metrics Health Check
  isPaused: true  # Alert is disabled
```

## Testing Alerts

### 1. Test Contact Points

In Grafana UI:
1. Go to Alerting → Contact points
2. Click on contact point name
3. Click "Test" button
4. Verify you receive the test notification

### 2. Trigger Test Alert

Generate test data that triggers an alert:

```python
# Example: Generate PII detection
from genai_otel import instrument
from genai_otel.evaluation import PIIConfig, PIIMode

instrument(
    enable_pii_detection=True,
    pii_config=PIIConfig(mode="detect", threshold=0.5)
)

# Make requests with PII
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "My SSN is 123-45-6789"}]
)
```

### 3. View Alert State

In Grafana UI:
1. Go to Alerting → Alert rules
2. See "Evaluation Alerts" folder
3. Check alert states: Normal, Pending, Firing

### 4. Silence Alerts

Temporarily disable alerts:
1. Go to Alerting → Silences
2. Create new silence
3. Add matchers (e.g., `severity=medium`)
4. Set duration

## Monitoring & Debugging

### View Alert History

```bash
# In Grafana UI
Alerting → Alert rules → [Alert name] → Show state history
```

### Check Provisioning Logs

```bash
# View Grafana logs
docker-compose logs grafana | grep -i "provisioning\|alert"

# Check for errors
docker-compose logs grafana | grep -i "error\|fail"
```

### Common Issues

#### Alerts Not Firing

1. **Check data source connection**:
   - Configuration → Data sources → OpenSearch → Save & Test

2. **Verify query returns data**:
   - Go to Explore
   - Select OpenSearch
   - Run the alert query manually

3. **Check evaluation interval**:
   - Alert may not have run yet (check interval in rule)

4. **Review alert state**:
   - Alerting → Alert rules → Click alert → "Show state history"

#### Notifications Not Sending

1. **Test contact point**:
   - Alerting → Contact points → Test

2. **Check SMTP settings**:
   ```bash
   docker-compose logs grafana | grep SMTP
   ```

3. **Verify routing policies**:
   - Alerting → Notification policies → View configuration

4. **Check notification log**:
   - Alerting → Contact points → [Contact point] → View notification log

#### Alert Stuck in "Pending"

- Alert condition must be true for the duration specified in `for:`
- Check: `for: 5m` means alert only fires after 5 minutes of continuous violation

## Best Practices

### 1. Alert Fatigue Prevention

- Start with higher thresholds, lower gradually
- Use appropriate `for:` durations to avoid noise
- Group related alerts together
- Use `repeat_interval` to avoid spam

### 2. Security Alert Response

Critical security alerts require immediate action:

1. **PII Exposure**:
   - Investigate trace in Jaeger
   - Check if PII was logged/stored
   - Audit data access
   - Report to security team

2. **Prompt Injection**:
   - Review attack patterns
   - Block malicious IPs if applicable
   - Update input validation
   - Document incident

3. **Blocked Requests Spike**:
   - Check for false positives
   - Investigate source IPs
   - Review evaluation thresholds
   - Update detection rules if needed

### 3. Alert Escalation

Configure escalation based on duration:

```yaml
# Example: Escalate if not resolved in 1 hour
routes:
  - receiver: on-call-team
    object_matchers:
      - ['severity', '=', 'critical']
    group_wait: 1h
    continue: false
```

### 4. Maintenance Windows

Define mute time intervals for scheduled maintenance:

```yaml
# In policies.yml
mute_time_intervals:
  - name: weekend-maintenance
    time_intervals:
      - weekdays: ['saturday', 'sunday']
        times:
          - start_time: '02:00'
            end_time: '06:00'
        location: 'UTC'
```

Apply to policy:

```yaml
policies:
  - orgId: 1
    receiver: default-email
    mute_time_intervals:
      - weekend-maintenance
```

## Alert Summary Table

| Alert Name | Severity | Window | Threshold | Response Time |
|------------|----------|--------|-----------|---------------|
| Critical PII Exposure | Critical | 5 min | Score ≥ 0.7 | Immediate |
| Prompt Injection Attack | Critical | 1 min | ≥ 3 attempts | Immediate |
| Blocked Requests Spike | Critical | 5 min | ≥ 10 blocked | 2 min |
| High Toxicity Rate | High | 15 min | > 5% rate | 5 min |
| High Bias Rate | High | 15 min | > 10% rate | 5 min |
| Hallucination Spike | Medium | 1 hour | > 20% rate | 15 min |
| Recurring PII Pattern | Medium | 30 min | ≥ 5 detections | 10 min |
| Low Citation Quality | Medium | 1 hour | ≥ 10 cases | 15 min |
| Metrics Health Check | Low | 1 hour | No data | 1 hour |

## Support

For issues with alerts:
- Check Grafana logs: `docker-compose logs grafana`
- Review OpenSearch connectivity
- Verify evaluation metrics are enabled in your application
- Test contact points manually

For alert rule customization:
- Grafana alerting docs: https://grafana.com/docs/grafana/latest/alerting/
- Expression queries: https://grafana.com/docs/grafana/latest/panels-visualizations/query-transform-data/expression-queries/

## Version History

- **v1.0** (2026-01-13) - Initial alert configuration
  - 11 alert rules across 4 severity levels
  - Email, Slack, PagerDuty support
  - Auto-provisioning with Grafana startup
