#!/bin/bash

echo "======================================================================"
echo "Fixing ALL OpenTelemetry Dependencies - Linux/macOS"
echo "======================================================================"

echo
echo "Step 1: Fixing protobuf version conflict..."
pip install "protobuf>=4.21.0,<6.0.0"

echo
echo "Step 2: Installing OpenTelemetry core packages (version 1.37.0)..."
pip install --upgrade \
    "opentelemetry-api==1.37.0" \
    "opentelemetry-sdk==1.37.0" \
    "opentelemetry-semantic-conventions==0.58b0"

echo
echo "Step 3: Installing OpenTelemetry OTLP exporters (version 1.37.0)..."
pip install --upgrade \
    "opentelemetry-proto==1.37.0" \
    "opentelemetry-exporter-otlp-proto-common==1.37.0" \
    "opentelemetry-exporter-otlp-proto-grpc==1.37.0" \
    "opentelemetry-exporter-otlp-proto-http==1.37.0" \
    "opentelemetry-exporter-otlp==1.37.0"

echo
echo "Step 4: Installing OpenTelemetry instrumentation (version 0.58b0)..."
pip install --upgrade \
    "opentelemetry-instrumentation==0.58b0" \
    "opentelemetry-instrumentation-httpx==0.58b0" \
    "opentelemetry-instrumentation-requests==0.58b0" \
    "opentelemetry-instrumentation-sqlalchemy==0.58b0" \
    "opentelemetry-instrumentation-psycopg2==0.58b0" \
    "opentelemetry-instrumentation-pymongo==0.58b0" \
    "opentelemetry-instrumentation-pymysql==0.58b0" \
    "opentelemetry-instrumentation-redis==0.58b0"

echo
echo "Step 5: Fixing pynvml warning..."
pip uninstall -y pynvml 2>/dev/null
pip install nvidia-ml-py

echo
echo "Step 6: Verifying installation..."
python -c "from opentelemetry.semconv._incubating.attributes import http_attributes; print('SUCCESS: Semantic conventions working!')" 2>/dev/null && (
    echo "[OK] Semantic conventions"
) || (
    echo "[FAILED] Semantic conventions"
)

echo
echo "======================================================================"
echo "Installed OpenTelemetry package versions:"
echo "======================================================================"
pip list | grep opentelemetry

echo
echo "======================================================================"
echo "Fix complete! Now run: python simple_test.py"
echo "======================================================================"
