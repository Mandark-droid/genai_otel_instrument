#!/bin/bash

# Ultra-simple script to combine all files
OUTPUT_FILE="${1:-combined_code.txt}"

echo "Creating combined code file: $OUTPUT_FILE"

rm -f "$OUTPUT_FILE"

for file in $(find . -type f -not -path '*/\.*' | grep -v "./$OUTPUT_FILE"); do
    echo "# $file" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "Complete!"