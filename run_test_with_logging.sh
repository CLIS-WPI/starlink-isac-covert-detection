#!/bin/bash
# ======================================
# 🔧 Test Script with Full Logging
# ======================================
# Runs test_complete_system.sh and logs ALL output to a file
# ======================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/test_run_${TIMESTAMP}.log"

echo "======================================================================"
echo "🚀 Starting Complete System Test with Full Logging"
echo "======================================================================"
echo "📝 All output will be saved to: ${LOG_FILE}"
echo "======================================================================"
echo ""

# Run test script and capture ALL output (stdout + stderr)
# Use tee to both display and save to file
# Remove -x flag to reduce verbosity (use -x if you need command tracing)
bash test_complete_system.sh 2>&1 | tee "${LOG_FILE}"

# Get exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Test completed successfully!"
else
    echo "⚠️  Test completed with exit code: ${EXIT_CODE}"
fi
echo "📝 Full log saved to: ${LOG_FILE}"
echo "======================================================================"

# Also create a summary file with just errors and warnings
ERROR_LOG="logs/test_errors_${TIMESTAMP}.log"
grep -i "error\|warning\|failed\|exception\|traceback\|⚠️\|❌" "${LOG_FILE}" > "${ERROR_LOG}" 2>/dev/null || true

if [ -s "${ERROR_LOG}" ]; then
    echo "⚠️  Errors/Warnings extracted to: ${ERROR_LOG}"
    echo ""
    echo "📋 Quick Error Summary:"
    head -20 "${ERROR_LOG}"
else
    echo "✅ No errors found in log!"
fi

echo ""
echo "======================================================================"
echo "📊 Log Analysis:"
echo "  Full log: ${LOG_FILE}"
echo "  Errors only: ${ERROR_LOG}"
echo "======================================================================"

exit ${EXIT_CODE}

