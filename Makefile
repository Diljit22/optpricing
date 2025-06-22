.PHONY: fix dev docs test clean tree

fix:
	@echo "Ruff format + fix"
	ruff format .
	ruff check --fix src/quantfin tests scripts

dev:
	@echo "Re-install in editable mode (with [dev] extras)"
	-pip uninstall -y quantfin || true
	pip install -e '.[dev]'

docs:
	@echo "MkDocs live server"
	mkdocs serve

test:
	@echo "Test suite"
	pytest

clean:
	@echo "Cleaning Python byte-code caches"
	python scripts/devtools/clean_cache.py

tree:
	@echo "Project tree"
	python scripts/devtools/project_tree.py

price:
	@echo "All Models/Techs"
	python scripts/devtools/run_informal_test.py