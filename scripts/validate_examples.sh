#!/bin/bash
#
# Validate All Examples Script
#
# This script runs all PII, Toxicity, and Bias detection examples
# and validates that they execute without errors.
#
# Usage:
#   ./scripts/validate_examples.sh [options] [endpoint]
#
# Options:
#   -v, --verbose    Show detailed output from examples
#   -d, --dry-run    List examples without running them
#   -t, --timeout N  Set timeout in seconds (default: 90)
#   -h, --help       Show this help message
#
# Example:
#   ./scripts/validate_examples.sh http://192.168.206.128:55681
#   ./scripts/validate_examples.sh --verbose http://localhost:4318
#   ./scripts/validate_examples.sh --dry-run
#   ./scripts/validate_examples.sh --timeout 120 http://192.168.206.128:55681
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
VERBOSE=false
DRY_RUN=false
TIMEOUT=90
ENDPOINT="http://localhost:4318"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //'
            exit 0
            ;;
        http*)
            ENDPOINT="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

export OTEL_EXPORTER_OTLP_ENDPOINT="$ENDPOINT"

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Arrays to track results
declare -a FAILED_EXAMPLES
declare -a SKIPPED_EXAMPLES

echo "=========================================================================="
echo "GenAI OTEL Examples Validation"
echo "=========================================================================="
echo "OTEL Endpoint: $ENDPOINT"
echo "Timeout: ${TIMEOUT}s"
echo "Verbose: $VERBOSE"
echo "Dry Run: $DRY_RUN"
if [ -n "$OPENAI_API_KEY" ]; then
    echo "OpenAI API Key: ${OPENAI_API_KEY:0:10}..."
else
    echo "OpenAI API Key: Not set (examples will fail)"
fi
echo "=========================================================================="
echo ""

# Function to run an example
run_example() {
    local example_path=$1
    local example_name=$(basename "$example_path" .py)
    local category=$(basename $(dirname "$example_path"))

    TOTAL=$((TOTAL + 1))

    # Dry run mode - just list examples
    if [ "$DRY_RUN" = true ]; then
        echo "[$TOTAL] $category/$example_name"
        echo "    python $example_path"
        return
    fi

    echo -n "[$TOTAL] Testing $category/$example_name ... "

    # Skip env_var_config.py as it requires specific env vars
    if [[ "$example_name" == "env_var_config" ]]; then
        echo -e "${YELLOW}SKIPPED${NC} (requires env vars)"
        SKIPPED=$((SKIPPED + 1))
        SKIPPED_EXAMPLES+=("$category/$example_name")
        return
    fi

    # Skip perspective_api.py if no API key
    if [[ "$example_name" == "perspective_api" ]] && [[ -z "$PERSPECTIVE_API_KEY" ]]; then
        echo -e "${YELLOW}SKIPPED${NC} (requires PERSPECTIVE_API_KEY)"
        SKIPPED=$((SKIPPED + 1))
        SKIPPED_EXAMPLES+=("$category/$example_name")
        return
    fi

    # Run the example with configurable timeout
    if [ "$VERBOSE" = true ]; then
        # Verbose mode - show output
        if timeout ${TIMEOUT}s python "$example_path" 2>&1; then
            echo -e "${GREEN}PASSED${NC}"
            PASSED=$((PASSED + 1))
        else
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo -e "${RED}FAILED${NC} (timeout after ${TIMEOUT}s)"
            else
                echo -e "${RED}FAILED${NC} (exit code: $exit_code)"
            fi
            FAILED=$((FAILED + 1))
            FAILED_EXAMPLES+=("$category/$example_name")
        fi
    else
        # Silent mode - hide output
        if timeout ${TIMEOUT}s python "$example_path" > /dev/null 2>&1; then
            echo -e "${GREEN}PASSED${NC}"
            PASSED=$((PASSED + 1))
        else
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo -e "${RED}FAILED${NC} (timeout after ${TIMEOUT}s)"
            else
                echo -e "${RED}FAILED${NC} (exit code: $exit_code)"
            fi
            FAILED=$((FAILED + 1))
            FAILED_EXAMPLES+=("$category/$example_name")
        fi
    fi
}

# Test PII Detection Examples
echo -e "${BLUE}=== PII Detection Examples ===${NC}"
if [ -d "examples/pii_detection" ]; then
    for example in examples/pii_detection/*.py; do
        if [ -f "$example" ]; then
            run_example "$example"
        fi
    done
else
    echo -e "${RED}ERROR: examples/pii_detection directory not found${NC}"
    exit 1
fi
echo ""

# Test Toxicity Detection Examples
echo -e "${BLUE}=== Toxicity Detection Examples ===${NC}"
if [ -d "examples/toxicity_detection" ]; then
    for example in examples/toxicity_detection/*.py; do
        if [ -f "$example" ]; then
            run_example "$example"
        fi
    done
else
    echo -e "${RED}ERROR: examples/toxicity_detection directory not found${NC}"
    exit 1
fi
echo ""

# Test Bias Detection Examples
echo -e "${BLUE}=== Bias Detection Examples ===${NC}"
if [ -d "examples/bias_detection" ]; then
    for example in examples/bias_detection/*.py; do
        if [ -f "$example" ]; then
            run_example "$example"
        fi
    done
else
    echo -e "${YELLOW}WARNING: examples/bias_detection directory not found${NC}"
fi
echo ""

# Print Summary
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "=========================================================================="
    echo "DRY RUN SUMMARY"
    echo "=========================================================================="
    echo -e "Total Examples Found: $TOTAL"
    echo ""
    echo "To run all examples:"
    echo "  bash scripts/validate_examples.sh http://192.168.206.128:55681"
    echo ""
    echo "To run with verbose output:"
    echo "  bash scripts/validate_examples.sh --verbose http://192.168.206.128:55681"
    echo "=========================================================================="
    exit 0
fi

echo "=========================================================================="
echo "VALIDATION SUMMARY"
echo "=========================================================================="
echo -e "Total Examples:   $TOTAL"
echo -e "${GREEN}Passed:          $PASSED${NC}"
echo -e "${RED}Failed:          $FAILED${NC}"
echo -e "${YELLOW}Skipped:         $SKIPPED${NC}"
echo "=========================================================================="

# Print failed examples if any
if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed Examples:${NC}"
    for example in "${FAILED_EXAMPLES[@]}"; do
        echo "  - $example"
    done
fi

# Print skipped examples if any
if [ $SKIPPED -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Skipped Examples:${NC}"
    for example in "${SKIPPED_EXAMPLES[@]}"; do
        echo "  - $example"
    done
fi

echo ""

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Validation FAILED${NC}"
    echo ""
    echo "To see detailed output, run with --verbose flag"
    exit 1
else
    echo -e "${GREEN}Validation PASSED${NC}"
    exit 0
fi
