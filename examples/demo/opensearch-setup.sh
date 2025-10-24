#!/bin/bash

# OpenSearch Setup Script for GenAI Trace Instrumentation
# This script creates an ingest pipeline and index template for extracting GenAI fields from Jaeger spans

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
echo "Creating genai-ingest-pipeline..."
curl -X PUT $CURL_AUTH "$OPENSEARCH_URL/_ingest/pipeline/genai-ingest-pipeline" \
  -H 'Content-Type: application/json' \
  -d '{
  "description": "GenAI OTel Instrumentation Pipeline - Extracts GenAI semantic convention fields from tags and flattens them for analysis. Includes support for LLM providers, token usage, costs, GPU metrics, and CO2 tracking.",
  "version": 1,
  "processors": [
    {
      "script": {
        "lang": "painless",
        "ignore_failure": true,
        "source": "if (ctx.tags != null) {\n  def tags = ctx.tags.stream().collect(Collectors.toMap(tag -> tag.key, tag -> tag.value));\n  \n  // OpenTelemetry standard fields\n  if (tags.containsKey(\"otel.status_code\")) {\n    ctx.status_code = tags.get(\"otel.status_code\");\n  }\n  if (tags.containsKey(\"otel.status_description\")) {\n    ctx.status_description = tags.get(\"otel.status_description\");\n  }\n  if (tags.containsKey(\"telemetry.sdk.language\")) {\n    ctx.telemetry_sdk_language = tags.get(\"telemetry.sdk.language\");\n  }\n  if (tags.containsKey(\"span.kind\")) {\n    ctx.span_kind = tags.get(\"span.kind\");\n  }\n  \n  // GenAI Semantic Conventions - Core Fields\n  if (tags.containsKey(\"gen_ai.system\")) {\n    ctx.gen_ai_system = tags.get(\"gen_ai.system\");\n  }\n  if (tags.containsKey(\"gen_ai.request.model\")) {\n    ctx.gen_ai_request_model = tags.get(\"gen_ai.request.model\");\n  }\n  if (tags.containsKey(\"gen_ai.request.type\")) {\n    ctx.gen_ai_request_type = tags.get(\"gen_ai.request.type\");\n  }\n  if (tags.containsKey(\"gen_ai.operation.name\")) {\n    ctx.gen_ai_operation_name = tags.get(\"gen_ai.operation.name\");\n  }\n  \n  // Token Usage\n  if (tags.containsKey(\"gen_ai.usage.prompt_tokens\")) {\n    ctx.gen_ai_usage_prompt_tokens = Integer.parseInt(tags.get(\"gen_ai.usage.prompt_tokens\"));\n  }\n  if (tags.containsKey(\"gen_ai.usage.completion_tokens\")) {\n    ctx.gen_ai_usage_completion_tokens = Integer.parseInt(tags.get(\"gen_ai.usage.completion_tokens\"));\n  }\n  if (tags.containsKey(\"gen_ai.usage.total_tokens\")) {\n    ctx.gen_ai_usage_total_tokens = Integer.parseInt(tags.get(\"gen_ai.usage.total_tokens\"));\n  }\n  \n  // Cost Tracking\n  if (tags.containsKey(\"gen_ai.cost.amount\")) {\n    ctx.gen_ai_cost_amount = Double.parseDouble(tags.get(\"gen_ai.cost.amount\"));\n  }\n  if (tags.containsKey(\"gen_ai.cost.currency\")) {\n    ctx.gen_ai_cost_currency = tags.get(\"gen_ai.cost.currency\");\n  }\n  \n  // Prompt and Response\n  if (tags.containsKey(\"gen_ai.prompt\")) {\n    ctx.gen_ai_prompt = tags.get(\"gen_ai.prompt\");\n  }\n  if (tags.containsKey(\"gen_ai.response.model\")) {\n    ctx.gen_ai_response_model = tags.get(\"gen_ai.response.model\");\n  }\n  if (tags.containsKey(\"gen_ai.response.finish_reason\")) {\n    ctx.gen_ai_response_finish_reason = tags.get(\"gen_ai.response.finish_reason\");\n  }\n  \n  // Streaming metrics\n  if (tags.containsKey(\"gen_ai.server.ttft\")) {\n    ctx.gen_ai_server_ttft = Double.parseDouble(tags.get(\"gen_ai.server.ttft\"));\n  }\n  if (tags.containsKey(\"gen_ai.server.tbt\")) {\n    ctx.gen_ai_server_tbt = Double.parseDouble(tags.get(\"gen_ai.server.tbt\"));\n  }\n  \n  // GPU Metrics\n  if (tags.containsKey(\"gen_ai.gpu.utilization\")) {\n    ctx.gen_ai_gpu_utilization = Double.parseDouble(tags.get(\"gen_ai.gpu.utilization\"));\n  }\n  if (tags.containsKey(\"gen_ai.gpu.memory.used\")) {\n    ctx.gen_ai_gpu_memory_used = Double.parseDouble(tags.get(\"gen_ai.gpu.memory.used\"));\n  }\n  if (tags.containsKey(\"gen_ai.gpu.temperature\")) {\n    ctx.gen_ai_gpu_temperature = Double.parseDouble(tags.get(\"gen_ai.gpu.temperature\"));\n  }\n  if (tags.containsKey(\"gen_ai.gpu.power\")) {\n    ctx.gen_ai_gpu_power = Double.parseDouble(tags.get(\"gen_ai.gpu.power\"));\n  }\n  if (tags.containsKey(\"gpu_id\")) {\n    ctx.gpu_id = tags.get(\"gpu_id\");\n  }\n  if (tags.containsKey(\"gpu_name\")) {\n    ctx.gpu_name = tags.get(\"gpu_name\");\n  }\n  \n  // CO2 Tracking\n  if (tags.containsKey(\"gen_ai.co2.emissions\")) {\n    ctx.gen_ai_co2_emissions = Double.parseDouble(tags.get(\"gen_ai.co2.emissions\"));\n  }\n  \n  // Service Information\n  if (tags.containsKey(\"service.name\")) {\n    ctx.service_name = tags.get(\"service.name\");\n  }\n  if (tags.containsKey(\"service.instance.id\")) {\n    ctx.service_instance_id = tags.get(\"service.instance.id\");\n  }\n  if (tags.containsKey(\"service.version\")) {\n    ctx.service_version = tags.get(\"service.version\");\n  }\n  \n  // Error handling\n  if (tags.containsKey(\"error\")) {\n    ctx.error = tags.get(\"error\");\n  }\n  if (tags.containsKey(\"exception.type\")) {\n    ctx.exception_type = tags.get(\"exception.type\");\n  }\n  if (tags.containsKey(\"exception.message\")) {\n    ctx.exception_message = tags.get(\"exception.message\");\n  }\n  if (tags.containsKey(\"exception.stacktrace\")) {\n    ctx.exception_stacktrace = tags.get(\"exception.stacktrace\");\n  }\n  \n  // HTTP fields (for API calls)\n  if (tags.containsKey(\"http.url\") || tags.containsKey(\"http.route\")) {\n    ctx.http_url = tags.get(\"http.url\") != null ? tags.get(\"http.url\") : tags.get(\"http.route\");\n  }\n  if (tags.containsKey(\"http.method\")) {\n    ctx.http_method = tags.get(\"http.method\");\n  }\n  if (tags.containsKey(\"http.status_code\")) {\n    ctx.http_status_code = Integer.parseInt(tags.get(\"http.status_code\"));\n  }\n  if (tags.containsKey(\"http.host\")) {\n    ctx.http_host = tags.get(\"http.host\");\n  }\n}\n\n// Extract parent span ID\nif (ctx.references != null && ctx.references.length != 0) {\n  ctx.parent_spanID = ctx.references[0].spanID;\n}\n\n// Process tags\nif (ctx.process?.tags != null) {\n  def processTags = ctx.process.tags.stream().collect(Collectors.toMap(processTag -> processTag.key, processTag -> processTag.value));\n  if (processTags.containsKey(\"service.name\")) {\n    ctx.service_name = processTags.get(\"service.name\");\n  }\n  if (processTags.containsKey(\"telemetry.sdk.language\")) {\n    ctx.telemetry_sdk_language = processTags.get(\"telemetry.sdk.language\");\n  }\n}"
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
  "index_patterns": ["jaeger-span-*"],
  "template": {
    "settings": {
      "index": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "default_pipeline": "genai-ingest-pipeline"
      }
    },
    "mappings": {
      "properties": {
        "traceID": { "type": "keyword" },
        "spanID": { "type": "keyword" },
        "parent_spanID": { "type": "keyword" },
        "operationName": { "type": "keyword" },
        "duration": { "type": "long" },
        "startTime": { "type": "long" },
        "startTimeMillis": { "type": "date" },
        "span_status": { "type": "keyword" },
        "trace_status": { "type": "keyword" },

        "gen_ai_system": { "type": "keyword" },
        "gen_ai_request_model": { "type": "keyword" },
        "gen_ai_request_type": { "type": "keyword" },
        "gen_ai_operation_name": { "type": "keyword" },

        "gen_ai_usage_prompt_tokens": { "type": "integer" },
        "gen_ai_usage_completion_tokens": { "type": "integer" },
        "gen_ai_usage_total_tokens": { "type": "integer" },

        "gen_ai_cost_amount": { "type": "double" },
        "gen_ai_cost_currency": { "type": "keyword" },

        "gen_ai_prompt": { "type": "text", "index": false },
        "gen_ai_response_model": { "type": "keyword" },
        "gen_ai_response_finish_reason": { "type": "keyword" },

        "gen_ai_server_ttft": { "type": "double" },
        "gen_ai_server_tbt": { "type": "double" },

        "gen_ai_gpu_utilization": { "type": "double" },
        "gen_ai_gpu_memory_used": { "type": "double" },
        "gen_ai_gpu_temperature": { "type": "double" },
        "gen_ai_gpu_power": { "type": "double" },
        "gpu_id": { "type": "keyword" },
        "gpu_name": { "type": "keyword" },

        "gen_ai_co2_emissions": { "type": "double" },

        "service_name": { "type": "keyword" },
        "service_instance_id": { "type": "keyword" },
        "service_version": { "type": "keyword" },
        "telemetry_sdk_language": { "type": "keyword" },

        "status_code": { "type": "keyword" },
        "status_description": { "type": "text" },
        "span_kind": { "type": "keyword" },

        "error": { "type": "keyword" },
        "exception_type": { "type": "keyword" },
        "exception_message": { "type": "text" },
        "exception_stacktrace": { "type": "text", "index": false },

        "http_url": { "type": "keyword" },
        "http_method": { "type": "keyword" },
        "http_status_code": { "type": "integer" },
        "http_host": { "type": "keyword" }
      }
    }
  }
}'

echo -e "\n\n✅ OpenSearch setup complete!"
echo "Pipeline created: genai-ingest-pipeline"
echo "Index template created: genai-jaeger-span-template"
echo ""
echo "Verify the setup:"
echo "  Pipeline: curl $OPENSEARCH_URL/_ingest/pipeline/genai-ingest-pipeline"
echo "  Template: curl $OPENSEARCH_URL/_template/genai-jaeger-span-template"
