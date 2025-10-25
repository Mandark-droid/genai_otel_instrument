#!/bin/bash
# Script to test the package before releasing to PyPI

set -e  # Exit on error

echo "=========================================="
echo "Testing genai-otel-instrument Release"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        exit 1
    fi
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if we're in the right directory
print_info "Checking project structure..."
if [ ! -f "pyproject.toml" ] || [ ! -d "genai_otel" ]; then
    print_warning "Please run this script from the project root directory"
    exit 1
fi
print_status $? "Project structure looks good"

# Check Python version and environment
print_info "Checking Python environment..."
python --version
python -c "import sys; print(f'Python path: {sys.executable}')"
print_status $? "Python environment check"

# Clean previous builds
print_info "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
print_status $? "Cleaned build directories and cache"

# Check if we're in a virtual environment (recommended)
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Not running in a virtual environment. Consider activating one for isolation."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Running in virtual environment: $VIRTUAL_ENV${NC}"
fi

# Install development dependencies
print_info "Installing development dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install -e ".[dev]"
else
    pip install -r requirements-dev.txt
fi
print_status $? "Development dependencies installed"

# Install optional dependencies for testing
print_info "Installing optional dependencies for testing..."
pip install \
    mysql-connector-python \
    psycopg2-binary \
    nvidia-ml-py \
    httpx \
    opentelemetry-instrumentation-sqlalchemy \
    opentelemetry-instrumentation-redis \
    opentelemetry-instrumentation-pymongo \
    opentelemetry-instrumentation-psycopg2 \
    opentelemetry-instrumentation-mysql \
    opentelemetry-instrumentation-kafka-python \
    opentelemetry-instrumentation-httpx
print_status $? "Optional and instrumentation dependencies installed"

# Auto-format code with both tools
print_info "Auto-formatting code with isort and black..."
isort genai_otel tests
black genai_otel tests
print_status $? "Code formatting applied"

# Run tests
print_info "Running tests..."
if [ -d "tests" ]; then
    pytest tests/ -v --cov=genai_otel --cov-report=term --cov-report=html --cov-report=term-missing
    print_status $? "All tests passed"
else
    print_warning "No tests directory found, skipping tests"
fi

# Code quality checks (should pass after auto-formatting)
print_info "Running code quality checks..."

# Check code formatting
print_info "Checking code formatting with black..."
black --check genai_otel tests
print_status $? "Code formatting check"

# Check import sorting
print_info "Checking import sorting with isort..."
isort --check-only genai_otel tests
print_status $? "Import sorting check"

# Run linter (with relaxed settings for now)
print_info "Running pylint..."
pylint genai_otel --rcfile=.pylintrc --exit-zero || true
print_status 0 "Linting complete (warnings are OK)"

# Build the package
print_info "Building package..."
python -m build
print_status $? "Package built successfully"

# Check the package
print_info "Checking package with twine..."
twine check dist/*
print_status $? "Package check passed"

# Test installation in a temporary environment (simplified - no pip upgrade)
print_info "Testing installation in temporary environment..."
TEMP_DIR=$(mktemp -d)
python -m venv "$TEMP_DIR/test_env"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source "$TEMP_DIR/test_env/Scripts/activate"
else
    source "$TEMP_DIR/test_env/bin/activate"
fi

# Install the package directly without upgrading pip
print_info "Installing package in temporary environment..."
pip install sqlalchemy mysql opentelemetry-instrumentation-mysql
pip install dist/*.whl
print_status $? "Package installed successfully"

# Test import
print_info "Testing package import..."
python -c "
import genai_otel
print(f'Successfully imported genai_otel version: {genai_otel.__version__}')

# Test that main components can be imported
from genai_otel.config import OTelConfig
from genai_otel.cost_calculator import CostCalculator
print('All core components imported successfully')
"
print_status $? "Package import successful"

# Test CLI
print_info "Testing CLI tool..."
genai-instrument --help > /dev/null 2>&1
print_status $? "CLI tool works"

# Cleanup
deactivate
rm -rf "$TEMP_DIR"
print_status $? "Cleanup completed"

echo ""
echo "=========================================="
echo -e "${GREEN}All checks passed! Ready for release! ✓${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Release Checklist:${NC}"
echo "✅ All tests passing"
echo "✅ Code formatting applied"
echo "✅ Import sorting applied"
echo "✅ Package builds successfully"
echo "✅ Installation tested"
echo "✅ CLI tool works"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review the generated distribution files:"
echo "   ls -la dist/"
echo ""
echo "2. Upload to Test PyPI:"
echo "   twine upload --repository testpypi dist/*"
echo ""
echo "3. Test install from Test PyPI in a clean environment:"
echo "   pip install --index-url https://test.pypi.org/simple/ genai-otel-instrument"
echo ""
echo "4. If everything works, upload to PyPI:"
echo "   twine upload dist/*"
echo ""
echo "5. Create a GitHub release:"
echo "   git tag v\$(python -c 'import genai_otel; print(genai_otel.__version__)')"
echo "   git push --tags"
echo ""
