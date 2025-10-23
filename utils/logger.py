# ======================================
# 📄 utils/logger.py
# Purpose: Initialize unified stdout/stderr logger to both console and file.
# This ensures all outputs are stored in a log file while still appearing on the terminal.
# ======================================

import sys
import atexit

class TeeLogger:
    """
    Logger that duplicates stdout/stderr to both console and a file.
    """
    def __init__(self, filename='output.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def init_logger(log_file='output.txt'):
    """
    Initialize logging by redirecting stdout/stderr to TeeLogger.
    """
    sys.stdout = TeeLogger(log_file)
    sys.stderr = sys.stdout
    atexit.register(lambda: sys.stdout.log.close())