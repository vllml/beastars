#!/bin/bash
# Cleanup script to remove temporary =* files from project root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Cleaning up temporary files..."

# Remove =* files from project root
count=$(find "$PROJECT_ROOT" -maxdepth 1 -name "=*" -type f 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    find "$PROJECT_ROOT" -maxdepth 1 -name "=*" -type f -delete 2>/dev/null
    echo "Removed $count temporary file(s)"
else
    echo "No temporary files found"
fi
