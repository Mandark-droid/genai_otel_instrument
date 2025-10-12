#!/bin/bash

# Script to merge all project files into a single text file
# Usage: ./merge_project.sh [output_file]

# Set output file name (default: project_merge.txt)


# Project root directory (adjust if needed)
PROJECT_ROOT="genai_otel_instrument"
OUTPUT_FILE="${1:-${PROJECT_ROOT}_project_merge.txt}"
# Directories to skip (separated by | for find command)
SKIP_DIRS="venv|__pycache__|htmlcov|.idea|.pytest_cache|.ruff_cache|.git|genai_otel_instrument.egg-info|dist|build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to process each file
process_file() {
    local file="$1"
    local relative_path="${file#$PROJECT_ROOT/}"

    echo "=== FILE: $relative_path ===" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Check if file is readable and text
    if file "$file" | grep -q "text"; then
        # Add line numbers to the content
        awk '{printf "%4d: %s\n", NR, $0}' "$file" >> "$OUTPUT_FILE"
    else
        echo "[BINARY FILE - CONTENT NOT SHOWN]" >> "$OUTPUT_FILE"
        echo "[File type: $(file -b "$file")]" >> "$OUTPUT_FILE"
    fi

    echo "" >> "$OUTPUT_FILE"
    echo "=== END OF: $relative_path ===" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
}

# Check if project directory exists
if [ ! -d "$PROJECT_ROOT" ]; then
    echo -e "${RED}Error: Project directory '$PROJECT_ROOT' not found${NC}"
    echo "Please run this script from the directory containing your project"
    exit 1
fi

# Clear output file
> "$OUTPUT_FILE"

# Add project header
echo "PROJECT MERGE: $PROJECT_ROOT" >> "$OUTPUT_FILE"
echo "Generated on: $(date)" >> "$OUTPUT_FILE"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'Not a git repository')" >> "$OUTPUT_FILE"
echo "Skipped directories: $SKIP_DIRS" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Counter for files processed
count=0
skipped=0

echo -e "${YELLOW}Starting project merge...${NC}"
echo -e "${YELLOW}Skipping directories: $SKIP_DIRS${NC}"
echo ""

# Process files using find with prune to skip directories
find "$PROJECT_ROOT" \
    -type d \( -name "venv" -o -name "__pycache__" -o -name "htmlcov" -o -name ".idea" \
             -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".git" \) -prune \
    -o -type f ! -name "$OUTPUT_FILE" -print0 | while IFS= read -r -d '' file; do

    # Skip common binary file extensions
    if [[ "$file" =~ \.(pyc|so|dll|exe|png|jpg|jpeg|gif|pdf|zip|tar|gz|ico|svg)$ ]] ||
       [[ "$(basename "$file")" == .* ]]; then
        echo -e "${YELLOW}Skipping: $file${NC}"
        ((skipped++))
        continue
    fi

    echo -e "${GREEN}Processing: $file${NC}"
    process_file "$file"
    ((count++))

done

# Add summary
echo "" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
echo "SUMMARY:" >> "$OUTPUT_FILE"
echo "Files processed: $count" >> "$OUTPUT_FILE"
echo "Files skipped: $skipped" >> "$OUTPUT_FILE"
echo "Total files found: $((count + skipped))" >> "$OUTPUT_FILE"

echo -e "\n${GREEN}Project merge completed!${NC}"
echo "Output file: $OUTPUT_FILE"
echo "Files processed: $count"
echo "Files skipped: $skipped"