.PHONY: help fix dev docs test clean demo tree

.DEFAULT_GOAL := help

help:
	@echo "Available developer commands:"
	@echo ""
	@echo "  make fix   : Auto-format and lint the code with Ruff."
	@echo "  make test  : Run the full pytest test suite."
	@echo "  make docs  : Serve the documentation site locally for live preview."
	@echo "  make demo  : Run the benchmark demo via the main CLI."
	@echo "  make dev   : Re-install library in editable mode with all dependencies."
	@echo "  make tree  : Display the clean project directory tree."
	@echo ""

fix:
	@echo "--- Formatting and linting with Ruff ---"
	ruff format .
	ruff check --fix .

dev:
	@echo "--- Re-installing in editable mode with [dev] extras ---"
	pip uninstall -y optpricing || true
	pip install -e '.[dev]'

docs:
	@echo "--- Starting MkDocs live server ---"
	mkdocs serve

test:
	@echo "--- Running pytest test suite ---"
	pytest

tree:
	@echo "--- Generating project tree ---"
	python devtools/project_tree.py

demo:
	@echo "--- Running demo via the 'optpricing' CLI ---"
	optpricing demo
