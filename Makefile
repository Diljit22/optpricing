.PHONY: help fix dev docs test clean demo tree

# This 'help' target is a common pattern to make the Makefile self-documenting.
help:
	@echo "Available commands:"
	@echo "  make fix       : Auto-format and lint the code with Ruff."
	@echo "  make dev       : Re-install the library in editable mode with all dependencies."
	@echo "  make docs      : Serve the documentation site locally for live preview."
	@echo "  make test      : Run the full pytest test suite."
	@echo "  make clean     : Remove all Python cache files."
	@echo "  make demo      : Run the benchmark/demo script."
	@echo "  make tree      : Display the clean project directory tree."

fix:
	@echo "Ruff format + fix"
	ruff format .
	ruff check --fix src/optpricing tests

dev:
	@echo "Re-install in editable mode (with [dev,app] extras)"
	-pip uninstall -y optPricing || true
	pip install -e '.[dev,app]'

docs:
	@echo "MkDocs live server"
	mkdocs serve

test:
	@echo "Test suite"
	pytest

demo:
	@echo "Running benchmark demo..."
	optpricing demo

tree:
	@echo "Project tree"
	python devtools/project_tree.py