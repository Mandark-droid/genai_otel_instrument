#!/bin/bash

# OpenSearch Setup Script for GenAI Trace Instrumentation
# This script creates an ingest pipeline and index template for extracting GenAI fields from Jaeger spans
# Version 2.4 - COMPLETE: Includes LLM function calling fields + all framework support

OPENSEARCH_URL="${OPENSEARCH_URL:-http://localhost:9200}"
OPENSEARCH_USERNAME="${OPENSEARCH_USERNAME:-}"
OPENSEARCH_PASSWORD="${OPENSEARCH_PASSWORD:-}"

# Build curl auth option if credentials are provided
if [ -n "$OPENSEARCH_USERNAME" ] && [ -n "$OPENSEARCH_PASSWORD" ]; then
    CURL_AUTH="-u ${OPENSEARCH_USERNAME}:${OPENSEARCH_PASSWORD}"
else
    CURL_AUTH=""
fi

echo "Waiting for OpenSearch to be ready..."
until curl -s $CURL_AUTH "$OPENSEARCH_URL/_cluster/health" > /dev/null; do
    echo "Waiting for OpenSearch..."
    sleep 5
done
echo "OpenSearch is ready!"

# Create the GenAI ingest pipeline
echo "Creating genai-ingest-pipeline with COMPREHENSIVE field extraction..."
echo "  - All GenAI semantic conventions"
echo "  - CrewAI framework fields"
echo "  - OpenInference semantic conventions"
echo "  - Tool/MCP instrumentation fields"
echo "  - Smolagents fields"
echo "  - LLM function calling fields (AutoGen)"
echo ""
curl -X PUT $CURL_AUTH "$OPENSEARCH_URL/_ingest/pipeline/genai-ingest-pipeline" \
  -H 'Content-Type: application/json' \
  -d '{
  "description": "GenAI OTel Instrumentation Pipeline v2.4 - COMPLETE field extraction including LLM function calling.",
  "version": 24,
  "processors": [
    {
      "script": {
        "lang": "painless",
        "ignore_failure": true,
        "source": "if (ctx.tags != null) {\n  def tags = ctx.tags.stream().collect(Collectors.toMap(tag -> tag.key, tag -> tag.value));\n  \n  // ==================== OTEL CORE FIELDS ====================\n  if (tags.containsKey(\"otel.status_code\")) {\n    ctx.otel_status_code = tags.get(\"otel.status_code\");\n  }\n  if (tags.containsKey(\"otel.status_description\")) {\n    ctx.otel_status_description = tags.get(\"otel.status_description\");\n  }\n  if (tags.containsKey(\"otel.scope.name\")) {\n    ctx.otel_scope_name = tags.get(\"otel.scope.name\");\n  }\n  if (tags.containsKey(\"otel.scope.version\")) {\n    ctx.otel_scope_version = tags.get(\"otel.scope.version\");\n  }\n  if (tags.containsKey(\"telemetry.sdk.language\")) {\n    ctx.telemetry_sdk_language = tags.get(\"telemetry.sdk.language\");\n  }\n  if (tags.containsKey(\"span.kind\")) {\n    ctx.span_kind = tags.get(\"span.kind\");\n  }\n  \n  // ==================== SESSION & USER TRACKING ====================\n  if (tags.containsKey(\"session.id\")) {\n    ctx.session_id = tags.get(\"session.id\");\n  }\n  if (tags.containsKey(\"user.id\")) {\n    ctx.user_id = tags.get(\"user.id\");\n  }\n  \n  // ==================== GENAI SEMANTIC CONVENTIONS ====================\n  // Core Fields\n  if (tags.containsKey(\"gen_ai.system\")) {\n    ctx.gen_ai_system = tags.get(\"gen_ai.system\");\n  }\n  if (tags.containsKey(\"gen_ai.request.model\")) {\n    ctx.gen_ai_request_model = tags.get(\"gen_ai.request.model\");\n  }\n  if (tags.containsKey(\"gen_ai.request.type\")) {\n    ctx.gen_ai_request_type = tags.get(\"gen_ai.request.type\");\n  }\n  if (tags.containsKey(\"gen_ai.operation.name\")) {\n    ctx.gen_ai_operation_name = tags.get(\"gen_ai.operation.name\");\n  }\n  \n  // Request Metadata\n  if (tags.containsKey(\"gen_ai.request.message_count\")) {\n    ctx.gen_ai_request_message_count = tags.get(\"gen_ai.request.message_count\");\n  }\n  if (tags.containsKey(\"gen_ai.request.first_message\")) {\n    ctx.gen_ai_request_first_message = tags.get(\"gen_ai.request.first_message\");\n  }\n  if (tags.containsKey(\"gen_ai.request.max_tokens\")) {\n    ctx.gen_ai_request_max_tokens = tags.get(\"gen_ai.request.max_tokens\");\n  }\n  if (tags.containsKey(\"gen_ai.request.temperature\")) {\n    ctx.gen_ai_request_temperature = tags.get(\"gen_ai.request.temperature\");\n  }\n  if (tags.containsKey(\"gen_ai.request.top_p\")) {\n    ctx.gen_ai_request_top_p = tags.get(\"gen_ai.request.top_p\");\n  }\n  \n  // Token Usage - Standard fields\n  if (tags.containsKey(\"gen_ai.usage.prompt_tokens\")) {\n    ctx.gen_ai_usage_prompt_tokens = tags.get(\"gen_ai.usage.prompt_tokens\");\n  }\n  if (tags.containsKey(\"gen_ai.usage.completion_tokens\")) {\n    ctx.gen_ai_usage_completion_tokens = tags.get(\"gen_ai.usage.completion_tokens\");\n  }\n  if (tags.containsKey(\"gen_ai.usage.total_tokens\")) {\n    ctx.gen_ai_usage_total_tokens = tags.get(\"gen_ai.usage.total_tokens\");\n  }\n  // Token aliases\n  if (tags.containsKey(\"gen_ai.usage.input_tokens\")) {\n    ctx.gen_ai_usage_input_tokens = tags.get(\"gen_ai.usage.input_tokens\");\n  }\n  if (tags.containsKey(\"gen_ai.usage.output_tokens\")) {\n    ctx.gen_ai_usage_output_tokens = tags.get(\"gen_ai.usage.output_tokens\");\n  }\n  // Alternative token count fields (llm.token_count.*)\n  if (tags.containsKey(\"llm.token_count.prompt\")) {\n    ctx.llm_token_count_prompt = tags.get(\"llm.token_count.prompt\");\n  }\n  if (tags.containsKey(\"llm.token_count.completion\")) {\n    ctx.llm_token_count_completion = tags.get(\"llm.token_count.completion\");\n  }\n  if (tags.containsKey(\"llm.token_count.total\")) {\n    ctx.llm_token_count_total = tags.get(\"llm.token_count.total\");\n  }\n  \n  // Cost Tracking - Legacy and detailed\n  if (tags.containsKey(\"gen_ai.cost.amount\")) {\n    ctx.gen_ai_cost_amount = tags.get(\"gen_ai.cost.amount\");\n  }\n  if (tags.containsKey(\"gen_ai.cost.currency\")) {\n    ctx.gen_ai_cost_currency = tags.get(\"gen_ai.cost.currency\");\n  }\n  if (tags.containsKey(\"gen_ai.usage.cost.total\")) {\n    ctx.gen_ai_usage_cost_total = tags.get(\"gen_ai.usage.cost.total\");\n  }\n  if (tags.containsKey(\"gen_ai.usage.cost.prompt\")) {\n    ctx.gen_ai_usage_cost_prompt = tags.get(\"gen_ai.usage.cost.prompt\");\n  }\n  if (tags.containsKey(\"gen_ai.usage.cost.completion\")) {\n    ctx.gen_ai_usage_cost_completion = tags.get(\"gen_ai.usage.cost.completion\");\n  }\n  \n  // Prompt and Response\n  if (tags.containsKey(\"gen_ai.prompt\")) {\n    ctx.gen_ai_prompt = tags.get(\"gen_ai.prompt\");\n  }\n  if (tags.containsKey(\"gen_ai.response\")) {\n    ctx.gen_ai_response = tags.get(\"gen_ai.response\");\n  }\n  if (tags.containsKey(\"gen_ai.response.id\")) {\n    ctx.gen_ai_response_id = tags.get(\"gen_ai.response.id\");\n  }\n  if (tags.containsKey(\"gen_ai.response.model\")) {\n    ctx.gen_ai_response_model = tags.get(\"gen_ai.response.model\");\n  }\n  if (tags.containsKey(\"gen_ai.response.finish_reason\")) {\n    ctx.gen_ai_response_finish_reason = tags.get(\"gen_ai.response.finish_reason\");\n  }\n  if (tags.containsKey(\"gen_ai.response.finish_reasons\")) {\n    ctx.gen_ai_response_finish_reasons = tags.get(\"gen_ai.response.finish_reasons\");\n  }\n  \n  // Streaming metrics\n  if (tags.containsKey(\"gen_ai.server.ttft\")) {\n    ctx.gen_ai_server_ttft = tags.get(\"gen_ai.server.ttft\");\n  }\n  if (tags.containsKey(\"gen_ai.server.tbt\")) {\n    ctx.gen_ai_server_tbt = tags.get(\"gen_ai.server.tbt\");\n  }\n  if (tags.containsKey(\"gen_ai.streaming.token_count\")) {\n    ctx.gen_ai_streaming_token_count = tags.get(\"gen_ai.streaming.token_count\");\n  }\n  \n  // GPU Metrics\n  if (tags.containsKey(\"gen_ai.gpu.utilization\")) {\n    ctx.gen_ai_gpu_utilization = tags.get(\"gen_ai.gpu.utilization\");\n  }\n  if (tags.containsKey(\"gen_ai.gpu.memory.used\")) {\n    ctx.gen_ai_gpu_memory_used = tags.get(\"gen_ai.gpu.memory.used\");\n  }\n  if (tags.containsKey(\"gen_ai.gpu.temperature\")) {\n    ctx.gen_ai_gpu_temperature = tags.get(\"gen_ai.gpu.temperature\");\n  }\n  if (tags.containsKey(\"gen_ai.gpu.power\")) {\n    ctx.gen_ai_gpu_power = tags.get(\"gen_ai.gpu.power\");\n  }\n  if (tags.containsKey(\"gpu_id\")) {\n    ctx.gpu_id = tags.get(\"gpu_id\");\n  }\n  if (tags.containsKey(\"gpu_name\")) {\n    ctx.gpu_name = tags.get(\"gpu_name\");\n  }\n  \n  // CO2 Tracking\n  if (tags.containsKey(\"gen_ai.co2.emissions\")) {\n    ctx.gen_ai_co2_emissions = tags.get(\"gen_ai.co2.emissions\");\n  }\n  \n  // ==================== OPENINFERENCE SEMANTIC CONVENTIONS ====================\n  if (tags.containsKey(\"openinference.span.kind\")) {\n    ctx.openinference_span_kind = tags.get(\"openinference.span.kind\");\n  }\n  if (tags.containsKey(\"input.value\")) {\n    ctx.input_value = tags.get(\"input.value\");\n  }\n  if (tags.containsKey(\"output.value\")) {\n    ctx.output_value = tags.get(\"output.value\");\n  }\n  if (tags.containsKey(\"output.mime_type\")) {\n    ctx.output_mime_type = tags.get(\"output.mime_type\");\n  }\n  \n  // ==================== TOOL/MCP INSTRUMENTATION ====================\n  if (tags.containsKey(\"tool.name\")) {\n    ctx.tool_name = tags.get(\"tool.name\");\n  }\n  if (tags.containsKey(\"tool.description\")) {\n    ctx.tool_description = tags.get(\"tool.description\");\n  }\n  if (tags.containsKey(\"tool.parameters\")) {\n    ctx.tool_parameters = tags.get(\"tool.parameters\");\n  }\n  \n  // ==================== CREWAI FRAMEWORK FIELDS ====================\n  // CrewAI Version Info\n  if (tags.containsKey(\"crewai_version\")) {\n    ctx.crewai_version = tags.get(\"crewai_version\");\n  }\n  if (tags.containsKey(\"python_version\")) {\n    ctx.python_version = tags.get(\"python_version\");\n  }\n  \n  // Crew Metadata\n  if (tags.containsKey(\"crew_key\")) {\n    ctx.crew_key = tags.get(\"crew_key\");\n  }\n  if (tags.containsKey(\"crew_id\")) {\n    ctx.crew_id = tags.get(\"crew_id\");\n  }\n  if (tags.containsKey(\"crew_fingerprint\")) {\n    ctx.crew_fingerprint = tags.get(\"crew_fingerprint\");\n  }\n  if (tags.containsKey(\"crew_process\")) {\n    ctx.crew_process = tags.get(\"crew_process\");\n  }\n  if (tags.containsKey(\"crew_memory\")) {\n    ctx.crew_memory = tags.get(\"crew_memory\");\n  }\n  if (tags.containsKey(\"crew_number_of_tasks\")) {\n    ctx.crew_number_of_tasks = tags.get(\"crew_number_of_tasks\");\n  }\n  if (tags.containsKey(\"crew_number_of_agents\")) {\n    ctx.crew_number_of_agents = tags.get(\"crew_number_of_agents\");\n  }\n  if (tags.containsKey(\"crew_fingerprint_created_at\")) {\n    ctx.crew_fingerprint_created_at = tags.get(\"crew_fingerprint_created_at\");\n  }\n  if (tags.containsKey(\"crew_agents\")) {\n    ctx.crew_agents = tags.get(\"crew_agents\");\n  }\n  if (tags.containsKey(\"crew_tasks\")) {\n    ctx.crew_tasks = tags.get(\"crew_tasks\");\n  }\n  \n  // Task Metadata\n  if (tags.containsKey(\"task_key\")) {\n    ctx.task_key = tags.get(\"task_key\");\n  }\n  if (tags.containsKey(\"task_id\")) {\n    ctx.task_id = tags.get(\"task_id\");\n  }\n  if (tags.containsKey(\"task_fingerprint\")) {\n    ctx.task_fingerprint = tags.get(\"task_fingerprint\");\n  }\n  if (tags.containsKey(\"task_fingerprint_created_at\")) {\n    ctx.task_fingerprint_created_at = tags.get(\"task_fingerprint_created_at\");\n  }\n  \n  // Agent Metadata\n  if (tags.containsKey(\"agent_fingerprint\")) {\n    ctx.agent_fingerprint = tags.get(\"agent_fingerprint\");\n  }\n  if (tags.containsKey(\"agent_role\")) {\n    ctx.agent_role = tags.get(\"agent_role\");\n  }\n  \n  // ==================== SMOLAGENTS FRAMEWORK ====================\n  if (tags.containsKey(\"smolagents.max_steps\")) {\n    ctx.smolagents_max_steps = tags.get(\"smolagents.max_steps\");\n  }\n  if (tags.containsKey(\"smolagents.tools_names\")) {\n    ctx.smolagents_tools_names = tags.get(\"smolagents.tools_names\");\n  }\n  \n  // ==================== LLM FUNCTION/TOOL CALLING (AUTOGEN) ====================\n  if (tags.containsKey(\"llm.tools\")) {\n    ctx.llm_tools = tags.get(\"llm.tools\");\n  }\n  if (tags.containsKey(\"llm.output_messages.0.message.tool_calls.0.tool_call.id\")) {\n    ctx.llm_tool_call_id = tags.get(\"llm.output_messages.0.message.tool_calls.0.tool_call.id\");\n  }\n  if (tags.containsKey(\"llm.output_messages.0.message.tool_calls.0.tool_call.function.name\")) {\n    ctx.llm_tool_call_function_name = tags.get(\"llm.output_messages.0.message.tool_calls.0.tool_call.function.name\");\n  }\n  if (tags.containsKey(\"llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments\")) {\n    ctx.llm_tool_call_function_arguments = tags.get(\"llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments\");\n  }\n  \n  // ==================== SERVICE INFORMATION ====================\n  if (tags.containsKey(\"service.name\")) {\n    ctx.service_name = tags.get(\"service.name\");\n  }\n  if (tags.containsKey(\"service.instance.id\")) {\n    ctx.service_instance_id = tags.get(\"service.instance.id\");\n  }\n  if (tags.containsKey(\"service.version\")) {\n    ctx.service_version = tags.get(\"service.version\");\n  }\n  \n  // ==================== ERROR HANDLING ====================\n  if (tags.containsKey(\"error\")) {\n    ctx.error = tags.get(\"error\");\n  }\n  if (tags.containsKey(\"exception.type\")) {\n    ctx.exception_type = tags.get(\"exception.type\");\n  }\n  if (tags.containsKey(\"exception.message\")) {\n    ctx.exception_message = tags.get(\"exception.message\");\n  }\n  if (tags.containsKey(\"exception.stacktrace\")) {\n    ctx.exception_stacktrace = tags.get(\"exception.stacktrace\");\n  }\n  \n  // ==================== HTTP FIELDS ====================\n  if (tags.containsKey(\"http.url\") || tags.containsKey(\"http.route\")) {\n    ctx.http_url = tags.get(\"http.url\") != null ? tags.get(\"http.url\") : tags.get(\"http.route\");\n  }\n  if (tags.containsKey(\"http.method\")) {\n    ctx.http_method = tags.get(\"http.method\");\n  }\n  if (tags.containsKey(\"http.status_code\")) {\n    ctx.http_status_code = tags.get(\"http.status_code\");\n  }\n  if (tags.containsKey(\"http.host\")) {\n    ctx.http_host = tags.get(\"http.host\");\n  }\n  \n  // ==================== EVALUATION METRICS ====================\n  // PII Detection - Prompt\n  if (tags.containsKey(\"evaluation.pii.prompt.detected\")) {\n    ctx.evaluation_pii_prompt_detected = tags.get(\"evaluation.pii.prompt.detected\");\n  }\n  if (tags.containsKey(\"evaluation.pii.prompt.entity_count\")) {\n    ctx.evaluation_pii_prompt_entity_count = tags.get(\"evaluation.pii.prompt.entity_count\");\n  }\n  if (tags.containsKey(\"evaluation.pii.prompt.score\")) {\n    ctx.evaluation_pii_prompt_score = tags.get(\"evaluation.pii.prompt.score\");\n  }\n  if (tags.containsKey(\"evaluation.pii.prompt.blocked\")) {\n    ctx.evaluation_pii_prompt_blocked = tags.get(\"evaluation.pii.prompt.blocked\");\n  }\n  if (tags.containsKey(\"evaluation.pii.prompt.redacted\")) {\n    ctx.evaluation_pii_prompt_redacted = tags.get(\"evaluation.pii.prompt.redacted\");\n  }\n  \n  // PII Detection - Response\n  if (tags.containsKey(\"evaluation.pii.response.detected\")) {\n    ctx.evaluation_pii_response_detected = tags.get(\"evaluation.pii.response.detected\");\n  }\n  if (tags.containsKey(\"evaluation.pii.response.entity_count\")) {\n    ctx.evaluation_pii_response_entity_count = tags.get(\"evaluation.pii.response.entity_count\");\n  }\n  if (tags.containsKey(\"evaluation.pii.response.score\")) {\n    ctx.evaluation_pii_response_score = tags.get(\"evaluation.pii.response.score\");\n  }\n  if (tags.containsKey(\"evaluation.pii.response.blocked\")) {\n    ctx.evaluation_pii_response_blocked = tags.get(\"evaluation.pii.response.blocked\");\n  }\n  if (tags.containsKey(\"evaluation.pii.response.redacted\")) {\n    ctx.evaluation_pii_response_redacted = tags.get(\"evaluation.pii.response.redacted\");\n  }\n  if (tags.containsKey(\"evaluation.pii.error\")) {\n    ctx.evaluation_pii_error = tags.get(\"evaluation.pii.error\");\n  }\n  \n  // Toxicity Detection - Prompt\n  if (tags.containsKey(\"evaluation.toxicity.prompt.detected\")) {\n    ctx.evaluation_toxicity_prompt_detected = tags.get(\"evaluation.toxicity.prompt.detected\");\n  }\n  if (tags.containsKey(\"evaluation.toxicity.prompt.max_score\")) {\n    ctx.evaluation_toxicity_prompt_max_score = tags.get(\"evaluation.toxicity.prompt.max_score\");\n  }\n  if (tags.containsKey(\"evaluation.toxicity.prompt.blocked\")) {\n    ctx.evaluation_toxicity_prompt_blocked = tags.get(\"evaluation.toxicity.prompt.blocked\");\n  }\n  \n  // Toxicity Detection - Response\n  if (tags.containsKey(\"evaluation.toxicity.response.detected\")) {\n    ctx.evaluation_toxicity_response_detected = tags.get(\"evaluation.toxicity.response.detected\");\n  }\n  if (tags.containsKey(\"evaluation.toxicity.response.max_score\")) {\n    ctx.evaluation_toxicity_response_max_score = tags.get(\"evaluation.toxicity.response.max_score\");\n  }\n  if (tags.containsKey(\"evaluation.toxicity.response.blocked\")) {\n    ctx.evaluation_toxicity_response_blocked = tags.get(\"evaluation.toxicity.response.blocked\");\n  }\n  if (tags.containsKey(\"evaluation.toxicity.error\")) {\n    ctx.evaluation_toxicity_error = tags.get(\"evaluation.toxicity.error\");\n  }\n  \n  // Bias Detection - Prompt\n  if (tags.containsKey(\"evaluation.bias.prompt.detected\")) {\n    ctx.evaluation_bias_prompt_detected = tags.get(\"evaluation.bias.prompt.detected\");\n  }\n  if (tags.containsKey(\"evaluation.bias.prompt.max_score\")) {\n    ctx.evaluation_bias_prompt_max_score = tags.get(\"evaluation.bias.prompt.max_score\");\n  }\n  if (tags.containsKey(\"evaluation.bias.prompt.blocked\")) {\n    ctx.evaluation_bias_prompt_blocked = tags.get(\"evaluation.bias.prompt.blocked\");\n  }\n  \n  // Bias Detection - Response\n  if (tags.containsKey(\"evaluation.bias.response.detected\")) {\n    ctx.evaluation_bias_response_detected = tags.get(\"evaluation.bias.response.detected\");\n  }\n  if (tags.containsKey(\"evaluation.bias.response.max_score\")) {\n    ctx.evaluation_bias_response_max_score = tags.get(\"evaluation.bias.response.max_score\");\n  }\n  if (tags.containsKey(\"evaluation.bias.response.blocked\")) {\n    ctx.evaluation_bias_response_blocked = tags.get(\"evaluation.bias.response.blocked\");\n  }\n  if (tags.containsKey(\"evaluation.bias.error\")) {\n    ctx.evaluation_bias_error = tags.get(\"evaluation.bias.error\");\n  }\n  \n  // Prompt Injection Detection\n  if (tags.containsKey(\"evaluation.prompt_injection.detected\")) {\n    ctx.evaluation_prompt_injection_detected = tags.get(\"evaluation.prompt_injection.detected\");\n  }\n  if (tags.containsKey(\"evaluation.prompt_injection.score\")) {\n    ctx.evaluation_prompt_injection_score = tags.get(\"evaluation.prompt_injection.score\");\n  }\n  if (tags.containsKey(\"evaluation.prompt_injection.types\")) {\n    ctx.evaluation_prompt_injection_types = tags.get(\"evaluation.prompt_injection.types\");\n  }\n  if (tags.containsKey(\"evaluation.prompt_injection.blocked\")) {\n    ctx.evaluation_prompt_injection_blocked = tags.get(\"evaluation.prompt_injection.blocked\");\n  }\n  if (tags.containsKey(\"evaluation.prompt_injection.error\")) {\n    ctx.evaluation_prompt_injection_error = tags.get(\"evaluation.prompt_injection.error\");\n  }\n  \n  // Restricted Topics - Prompt\n  if (tags.containsKey(\"evaluation.restricted_topics.prompt.detected\")) {\n    ctx.evaluation_restricted_topics_prompt_detected = tags.get(\"evaluation.restricted_topics.prompt.detected\");\n  }\n  if (tags.containsKey(\"evaluation.restricted_topics.prompt.blocked\")) {\n    ctx.evaluation_restricted_topics_prompt_blocked = tags.get(\"evaluation.restricted_topics.prompt.blocked\");\n  }\n  \n  // Restricted Topics - Response\n  if (tags.containsKey(\"evaluation.restricted_topics.response.detected\")) {\n    ctx.evaluation_restricted_topics_response_detected = tags.get(\"evaluation.restricted_topics.response.detected\");\n  }\n  if (tags.containsKey(\"evaluation.restricted_topics.response.blocked\")) {\n    ctx.evaluation_restricted_topics_response_blocked = tags.get(\"evaluation.restricted_topics.response.blocked\");\n  }\n  if (tags.containsKey(\"evaluation.restricted_topics.error\")) {\n    ctx.evaluation_restricted_topics_error = tags.get(\"evaluation.restricted_topics.error\");\n  }\n  \n  // Hallucination Detection - Response\n  if (tags.containsKey(\"evaluation.hallucination.response.detected\")) {\n    ctx.evaluation_hallucination_response_detected = tags.get(\"evaluation.hallucination.response.detected\");\n  }\n  if (tags.containsKey(\"evaluation.hallucination.response.score\")) {\n    ctx.evaluation_hallucination_response_score = tags.get(\"evaluation.hallucination.response.score\");\n  }\n  if (tags.containsKey(\"evaluation.hallucination.response.citations\")) {\n    ctx.evaluation_hallucination_response_citations = tags.get(\"evaluation.hallucination.response.citations\");\n  }\n  if (tags.containsKey(\"evaluation.hallucination.response.hedge_words\")) {\n    ctx.evaluation_hallucination_response_hedge_words = tags.get(\"evaluation.hallucination.response.hedge_words\");\n  }\n  if (tags.containsKey(\"evaluation.hallucination.response.claims\")) {\n    ctx.evaluation_hallucination_response_claims = tags.get(\"evaluation.hallucination.response.claims\");\n  }\n  if (tags.containsKey(\"evaluation.hallucination.response.indicators\")) {\n    ctx.evaluation_hallucination_response_indicators = tags.get(\"evaluation.hallucination.response.indicators\");\n  }\n  if (tags.containsKey(\"evaluation.hallucination.response.unsupported_claims\")) {\n    ctx.evaluation_hallucination_response_unsupported_claims = tags.get(\"evaluation.hallucination.response.unsupported_claims\");\n  }\n  if (tags.containsKey(\"evaluation.hallucination.error\")) {\n    ctx.evaluation_hallucination_error = tags.get(\"evaluation.hallucination.error\");\n  }\n}\n\n// Extract parent span ID\nif (ctx.references != null && ctx.references.length != 0) {\n  ctx.parent_spanID = ctx.references[0].spanID;\n}\n\n// Process tags\nif (ctx.process?.tags != null) {\n  def processTags = ctx.process.tags.stream().collect(Collectors.toMap(processTag -> processTag.key, processTag -> processTag.value));\n  if (processTags.containsKey(\"service.name\")) {\n    ctx.service_name = processTags.get(\"service.name\");\n  }\n  if (processTags.containsKey(\"telemetry.sdk.language\")) {\n    ctx.telemetry_sdk_language = processTags.get(\"telemetry.sdk.language\");\n  }\n}"
      }
    },
    {
      "script": {
        "lang": "painless",
        "ignore_failure": true,
        "source": "// Calculate span status based on GenAI context\nif (ctx?.duration != null) {\n  if (ctx?.gen_ai_system != null) {\n    // For GenAI operations\n    if (ctx?.error != null && ctx?.error == \"true\") {\n      ctx.span_status = \"ERROR\";\n      if (ctx?.references == null || ctx?.references.length == 0) {\n        ctx.trace_status = \"ERROR\";\n      }\n    } else if (ctx?.http_status_code != null && ctx?.http_status_code >= 400) {\n      ctx.span_status = \"ERROR\";\n      if (ctx?.references == null || ctx?.references.length == 0) {\n        ctx.trace_status = \"ERROR\";\n      }\n    } else if (ctx?.duration > 30000000) {\n      // >30 seconds is slow for LLM calls\n      ctx.span_status = \"SLOW\";\n      if (ctx?.references == null || ctx?.references.length == 0) {\n        ctx.trace_status = \"SLOW\";\n      }\n    } else {\n      ctx.span_status = \"OK\";\n      if (ctx?.references == null || ctx?.references.length == 0) {\n        ctx.trace_status = \"OK\";\n      }\n    }\n  } else {\n    // For non-GenAI operations, use standard logic\n    if (ctx?.http_status_code != null) {\n      if (ctx?.http_status_code >= 200 && ctx?.http_status_code < 400) {\n        if (ctx?.duration > 0 && ctx?.duration <= 5000000) {\n          ctx.span_status = \"OK\";\n          if (ctx?.references == null || ctx?.references.length == 0) {\n            ctx.trace_status = \"OK\";\n          }\n        } else if (ctx?.duration > 5000000) {\n          ctx.span_status = \"SLOW\";\n          if (ctx?.references == null || ctx?.references.length == 0) {\n            ctx.trace_status = \"SLOW\";\n          }\n        }\n      } else if (ctx?.http_status_code >= 400 && ctx?.http_status_code < 600) {\n        ctx.span_status = \"ERROR\";\n        if (ctx?.references == null || ctx?.references.length == 0) {\n          ctx.trace_status = \"ERROR\";\n        }\n      }\n    } else if (ctx?.error != null && ctx?.error == \"true\") {\n      ctx.span_status = \"ERROR\";\n      if (ctx?.references == null || ctx?.references.length == 0) {\n        ctx.trace_status = \"ERROR\";\n      }\n    } else {\n      if (ctx?.duration > 0 && ctx?.duration <= 5000000) {\n        ctx.span_status = \"OK\";\n        if (ctx?.references == null || ctx?.references.length == 0) {\n          ctx.trace_status = \"OK\";\n        }\n      } else if (ctx?.duration > 5000000) {\n        ctx.span_status = \"SLOW\";\n        if (ctx?.references == null || ctx?.references.length == 0) {\n          ctx.trace_status = \"SLOW\";\n        }\n      }\n    }\n  }\n}"
      }
    }
  ]
}'

echo -e "\n\nCreating index template for jaeger-span indices..."
curl -X PUT $CURL_AUTH "$OPENSEARCH_URL/_template/genai-jaeger-span-template" \
  -H 'Content-Type: application/json' \
  -d '{
  "order": 0,
  "index_patterns": ["*jaeger-span-*"],
  "settings": {
    "index": {
      "mapping": {
        "nested_fields": {
          "limit": "50"
        }
      },
      "requests": {
        "cache": {
          "enable": "true"
        }
      },
      "number_of_shards": "5",
      "number_of_replicas": "0",
      "default_pipeline": "genai-ingest-pipeline"
    }
  },
  "mappings": {
    "dynamic_templates": [
      {
        "span_tags_map": {
          "path_match": "tag.*",
          "mapping": {
            "ignore_above": 256,
            "type": "keyword"
          }
        }
      },
      {
        "process_tags_map": {
          "path_match": "process.tag.*",
          "mapping": {
            "ignore_above": 256,
            "type": "keyword"
          }
        }
      },
      {
        "strings_as_keyword": {
          "match_mapping_type": "string",
          "mapping": {
            "ignore_above": 256,
            "index": false,
            "type": "keyword"
          }
        }
      }
    ],
    "date_detection": true,
    "numeric_detection": false,
    "properties": {
      "traceID": { "type": "keyword", "ignore_above": 256 },
      "spanID": { "type": "keyword", "ignore_above": 256 },
      "parentSpanID": { "type": "keyword", "ignore_above": 256 },
      "parent_spanID": { "type": "keyword" },
      "operationName": { "type": "keyword", "ignore_above": 256 },
      "flags": { "type": "integer" },
      "duration": { "type": "long" },
      "startTime": { "type": "long" },
      "startTimeMillis": { "type": "date", "format": "epoch_millis" },
      "span_status": { "type": "keyword" },
      "trace_status": { "type": "keyword" },
      "process": {
        "properties": {
          "serviceName": { "type": "keyword", "ignore_above": 256 },
          "tag": { "type": "object" },
          "tags": {
            "type": "nested",
            "dynamic": false,
            "properties": {
              "key": { "type": "keyword", "ignore_above": 256 },
              "value": { "type": "keyword", "ignore_above": 256 },
              "tagType": { "type": "keyword", "ignore_above": 256 }
            }
          }
        }
      },
      "tags": {
        "type": "nested",
        "dynamic": false,
        "properties": {
          "key": { "type": "keyword", "ignore_above": 256 },
          "value": { "type": "keyword", "ignore_above": 256 },
          "tagType": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "tag": { "type": "object" },
      "references": {
        "type": "nested",
        "dynamic": false,
        "properties": {
          "refType": { "type": "keyword", "ignore_above": 256 },
          "traceID": { "type": "keyword", "ignore_above": 256 },
          "spanID": { "type": "keyword", "ignore_above": 256 }
        }
      },

        "__comment_otel": { "type": "text", "index": false, "doc_values": false },
        "otel_status_code": { "type": "keyword" },
        "otel_status_description": { "type": "text" },
        "otel_scope_name": { "type": "keyword" },
        "otel_scope_version": { "type": "keyword" },
        "telemetry_sdk_language": { "type": "keyword" },
        "span_kind": { "type": "keyword" },

        "__comment_session": { "type": "text", "index": false, "doc_values": false },
        "session_id": { "type": "keyword" },
        "user_id": { "type": "keyword" },

        "__comment_genai": { "type": "text", "index": false, "doc_values": false },
        "gen_ai_system": { "type": "keyword" },
        "gen_ai_request_model": { "type": "keyword" },
        "gen_ai_request_type": { "type": "keyword" },
        "gen_ai_operation_name": { "type": "keyword" },

        "gen_ai_request_message_count": { "type": "integer" },
        "gen_ai_request_first_message": { "type": "text", "index": false },
        "gen_ai_request_max_tokens": { "type": "integer" },
        "gen_ai_request_temperature": { "type": "double" },
        "gen_ai_request_top_p": { "type": "double" },

        "gen_ai_usage_prompt_tokens": { "type": "integer" },
        "gen_ai_usage_completion_tokens": { "type": "integer" },
        "gen_ai_usage_total_tokens": { "type": "integer" },
        "gen_ai_usage_input_tokens": { "type": "integer" },
        "gen_ai_usage_output_tokens": { "type": "integer" },

        "llm_token_count_prompt": { "type": "integer" },
        "llm_token_count_completion": { "type": "integer" },
        "llm_token_count_total": { "type": "integer" },

        "gen_ai_cost_amount": { "type": "double" },
        "gen_ai_cost_currency": { "type": "keyword" },
        "gen_ai_usage_cost_total": { "type": "double" },
        "gen_ai_usage_cost_prompt": { "type": "double" },
        "gen_ai_usage_cost_completion": { "type": "double" },

        "gen_ai_prompt": { "type": "text", "index": false },
        "gen_ai_response": { "type": "text", "index": false },
        "gen_ai_response_id": { "type": "keyword" },
        "gen_ai_response_model": { "type": "keyword" },
        "gen_ai_response_finish_reason": { "type": "keyword" },
        "gen_ai_response_finish_reasons": { "type": "keyword" },

        "gen_ai_server_ttft": { "type": "double" },
        "gen_ai_server_tbt": { "type": "double" },
        "gen_ai_streaming_token_count": { "type": "integer" },

        "gen_ai_gpu_utilization": { "type": "double" },
        "gen_ai_gpu_memory_used": { "type": "double" },
        "gen_ai_gpu_temperature": { "type": "double" },
        "gen_ai_gpu_power": { "type": "double" },
        "gpu_id": { "type": "keyword" },
        "gpu_name": { "type": "keyword" },

        "gen_ai_co2_emissions": { "type": "double" },

        "__comment_openinference": { "type": "text", "index": false, "doc_values": false },
        "openinference_span_kind": { "type": "keyword" },
        "input_value": { "type": "text", "index": false },
        "output_value": { "type": "text", "index": false },
        "output_mime_type": { "type": "keyword" },

        "__comment_tools": { "type": "text", "index": false, "doc_values": false },
        "tool_name": { "type": "keyword" },
        "tool_description": { "type": "text" },
        "tool_parameters": { "type": "text", "index": false },

        "__comment_crewai": { "type": "text", "index": false, "doc_values": false },
        "crewai_version": { "type": "keyword" },
        "python_version": { "type": "keyword" },
        "crew_key": { "type": "keyword" },
        "crew_id": { "type": "keyword" },
        "crew_fingerprint": { "type": "keyword" },
        "crew_process": { "type": "keyword" },
        "crew_memory": { "type": "boolean" },
        "crew_number_of_tasks": { "type": "integer" },
        "crew_number_of_agents": { "type": "integer" },
        "crew_fingerprint_created_at": { "type": "keyword" },
        "crew_agents": { "type": "text", "index": false },
        "crew_tasks": { "type": "text", "index": false },
        "task_key": { "type": "keyword" },
        "task_id": { "type": "keyword" },
        "task_fingerprint": { "type": "keyword" },
        "task_fingerprint_created_at": { "type": "keyword" },
        "agent_fingerprint": { "type": "keyword" },
        "agent_role": { "type": "keyword" },

        "__comment_smolagents": { "type": "text", "index": false, "doc_values": false },
        "smolagents_max_steps": { "type": "integer" },
        "smolagents_tools_names": { "type": "keyword" },

        "__comment_llm_function_calling": { "type": "text", "index": false, "doc_values": false },
        "llm_tools": { "type": "text", "index": false },
        "llm_tool_call_id": { "type": "keyword" },
        "llm_tool_call_function_name": { "type": "keyword" },
        "llm_tool_call_function_arguments": { "type": "text", "index": false },

        "__comment_service": { "type": "text", "index": false, "doc_values": false },
        "service_name": { "type": "keyword" },
        "service_instance_id": { "type": "keyword" },
        "service_version": { "type": "keyword" },

        "__comment_error": { "type": "text", "index": false, "doc_values": false },
        "error": { "type": "keyword" },
        "exception_type": { "type": "keyword" },
        "exception_message": { "type": "text" },
        "exception_stacktrace": { "type": "text", "index": false },

        "__comment_http": { "type": "text", "index": false, "doc_values": false },
        "http_url": { "type": "keyword" },
        "http_method": { "type": "keyword" },
        "http_status_code": { "type": "integer" },
        "http_host": { "type": "keyword" },

        "__comment_evaluation": { "type": "text", "index": false, "doc_values": false },

        "evaluation_pii_prompt_detected": { "type": "boolean" },
        "evaluation_pii_prompt_entity_count": { "type": "integer" },
        "evaluation_pii_prompt_score": { "type": "double" },
        "evaluation_pii_prompt_blocked": { "type": "boolean" },
        "evaluation_pii_prompt_redacted": { "type": "text", "index": false },

        "evaluation_pii_response_detected": { "type": "boolean" },
        "evaluation_pii_response_entity_count": { "type": "integer" },
        "evaluation_pii_response_score": { "type": "double" },
        "evaluation_pii_response_blocked": { "type": "boolean" },
        "evaluation_pii_response_redacted": { "type": "text", "index": false },
        "evaluation_pii_error": { "type": "text" },

        "evaluation_toxicity_prompt_detected": { "type": "boolean" },
        "evaluation_toxicity_prompt_max_score": { "type": "double" },
        "evaluation_toxicity_prompt_blocked": { "type": "boolean" },

        "evaluation_toxicity_response_detected": { "type": "boolean" },
        "evaluation_toxicity_response_max_score": { "type": "double" },
        "evaluation_toxicity_response_blocked": { "type": "boolean" },
        "evaluation_toxicity_error": { "type": "text" },

        "evaluation_bias_prompt_detected": { "type": "boolean" },
        "evaluation_bias_prompt_max_score": { "type": "double" },
        "evaluation_bias_prompt_blocked": { "type": "boolean" },

        "evaluation_bias_response_detected": { "type": "boolean" },
        "evaluation_bias_response_max_score": { "type": "double" },
        "evaluation_bias_response_blocked": { "type": "boolean" },
        "evaluation_bias_error": { "type": "text" },

        "evaluation_prompt_injection_detected": { "type": "boolean" },
        "evaluation_prompt_injection_score": { "type": "double" },
        "evaluation_prompt_injection_types": { "type": "keyword" },
        "evaluation_prompt_injection_blocked": { "type": "boolean" },
        "evaluation_prompt_injection_error": { "type": "text" },

        "evaluation_restricted_topics_prompt_detected": { "type": "boolean" },
        "evaluation_restricted_topics_prompt_blocked": { "type": "boolean" },

        "evaluation_restricted_topics_response_detected": { "type": "boolean" },
        "evaluation_restricted_topics_response_blocked": { "type": "boolean" },
        "evaluation_restricted_topics_error": { "type": "text" },

        "evaluation_hallucination_response_detected": { "type": "boolean" },
        "evaluation_hallucination_response_score": { "type": "double" },
        "evaluation_hallucination_response_citations": { "type": "integer" },
        "evaluation_hallucination_response_hedge_words": { "type": "integer" },
        "evaluation_hallucination_response_claims": { "type": "integer" },
        "evaluation_hallucination_response_indicators": { "type": "keyword" },
        "evaluation_hallucination_response_unsupported_claims": { "type": "text", "index": false },
        "evaluation_hallucination_error": { "type": "text" }
    }
  }
}'

echo -e "\n\n================================"
echo "OpenSearch setup COMPLETE v2.4"
echo "================================"
echo ""
echo "Pipeline: genai-ingest-pipeline (v2.4)"
echo "Template: genai-jaeger-span-template"
echo ""
echo "NEW in v2.4 - 4 additional LLM function calling fields:"
echo "  [AutoGen/Function Calling] llm.tools, tool_call.id, function.name, function.arguments"
echo ""
echo "Previous v2.3 additions (32 fields):"
echo "  [OTel Core] otel.scope.name, otel.scope.version (2)"
echo "  [CrewAI] Complete framework support (18 fields)"
echo "  [OpenInference] span.kind, input/output values (4)"
echo "  [Tools/MCP] tool.name, description, parameters (3)"
echo "  [Smolagents] max_steps, tools_names (2)"
echo "  [Token Counts] llm.token_count.* aliases (3)"
echo ""
echo "Total fields extracted: ~99 fields"
echo ""
echo "Framework Support:"
echo "  - GenAI Semantic Conventions (full)"
echo "  - CrewAI (complete)"
echo "  - OpenInference (complete)"
echo "  - Smolagents (complete)"
echo "  - AutoGen function calling (complete)"
echo "  - Tool/MCP instrumentation (complete)"
echo "  - Evaluation Metrics (PII, Toxicity, Bias, Hallucination)"
echo ""
echo "Verify the setup:"
echo "  curl $OPENSEARCH_URL/_ingest/pipeline/genai-ingest-pipeline"
echo "  curl $OPENSEARCH_URL/_template/genai-jaeger-span-template"
echo ""
echo "STORAGE NOTE: gen_ai.response, crew_agents/crew_tasks, input/output"
echo "values, and llm.tools will significantly increase storage. Consider"
echo "compression settings and data retention policies."
echo ""
