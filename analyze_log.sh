#!/bin/bash
# ======================================
# 🔍 Log Analysis Script
# ======================================
# Analyzes test log files and extracts issues
# ======================================

if [ $# -eq 0 ]; then
    # Find the most recent log file
    LATEST_LOG=$(ls -t logs/test_run_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "❌ No log files found in logs/ directory"
        echo "   Run: bash run_test_with_logging.sh first"
        exit 1
    fi
    LOG_FILE="$LATEST_LOG"
    echo "📝 Using latest log: $LOG_FILE"
else
    LOG_FILE="$1"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

echo "======================================================================"
echo "🔍 Analyzing Log File: $LOG_FILE"
echo "======================================================================"
echo ""

# Extract different types of issues
echo "📊 Error Summary:"
echo "----------------------------------------------------------------------"
ERRORS=$(grep -i "error\|exception\|traceback\|failed" "$LOG_FILE" | wc -l)
echo "  Total errors/exceptions: $ERRORS"
if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "  First 10 errors:"
    grep -i "error\|exception\|traceback" "$LOG_FILE" | head -10 | sed 's/^/    /'
fi

echo ""
echo "⚠️  Warning Summary:"
echo "----------------------------------------------------------------------"
WARNINGS=$(grep -i "warning\|⚠️" "$LOG_FILE" | wc -l)
echo "  Total warnings: $WARNINGS"
if [ $WARNINGS -gt 0 ]; then
    echo ""
    echo "  First 10 warnings:"
    grep -i "warning\|⚠️" "$LOG_FILE" | head -10 | sed 's/^/    /'
fi

echo ""
echo "❌ Failed Operations:"
echo "----------------------------------------------------------------------"
FAILED=$(grep -i "failed\|❌" "$LOG_FILE" | wc -l)
echo "  Total failed operations: $FAILED"
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "  Failed operations:"
    grep -i "failed\|❌" "$LOG_FILE" | sed 's/^/    /'
fi

echo ""
echo "✅ Success Summary:"
echo "----------------------------------------------------------------------"
SUCCESS=$(grep -i "✅\|complete\|success" "$LOG_FILE" | wc -l)
echo "  Total success messages: $SUCCESS"

echo ""
echo "📋 Phase Completion Status:"
echo "----------------------------------------------------------------------"
for phase in "Phase 0" "Phase 1" "Phase 2" "Phase 3" "Phase 4" "Phase 5" "Phase 6"; do
    if grep -q "$phase" "$LOG_FILE"; then
        if grep "$phase" "$LOG_FILE" | grep -qi "complete\|✅\|success"; then
            echo "  ✅ $phase: Completed"
        else
            echo "  ⚠️  $phase: Found but status unclear"
        fi
    else
        echo "  ❌ $phase: Not found in log"
    fi
done

echo ""
echo "======================================================================"
echo "📁 Full log file: $LOG_FILE"
echo "======================================================================"

