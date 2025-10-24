.PHONY: test quick-test install-test clean

# Quick test without dependencies
quick-test:
	@python tests/quick_test.py

# Install test dependencies
install-test:
	@pip install pytest pytest-cov pytest-xdist --break-system-packages

# Run full test suite
test:
	@python tests/run_all_tests.py

# Run specific test file
test-detector:
	@pytest tests/test_detector.py -v

test-dataset:
	@pytest tests/test_dataset_generator.py -v

test-localization:
	@pytest tests/test_localization.py -v

# Clean test artifacts
clean:
	@rm -rf .pytest_cache __pycache__ tests/__pycache__
	@rm -rf htmlcov .coverage
	@find . -name "*.pyc" -delete

# Help
help:
	@echo "Available commands:"
	@echo "  make quick-test      - Run quick tests (no dependencies)"
	@echo "  make install-test    - Install test dependencies"
	@echo "  make test           - Run full test suite"
	@echo "  make test-detector  - Test detector only"
	@echo "  make clean          - Clean test artifacts"