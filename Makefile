.PHONY: help clean

# --- Variables ---

.DEFAULT_GOAL := help

# --- General Targets ---

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@printf "  \033[36m%-28s\033[0m %s\n" "help" "Show this help message"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| grep -v '^help:' \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'


clean: ## Remove cache files and generated local artifacts
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -not -path "*/.venv/*" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -not -path "*/.venv/*" -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -not -path "*/.venv/*" -exec rm -rf {} +
	@rm -rf .coverage htmlcov/
	@echo "Cleanup complete."

