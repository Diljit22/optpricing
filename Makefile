.PHONY: fix docs tree american european general

fix:
	@echo "--- Formatting and linting with Ruff ---"
	ruff format .
	ruff check --fix .

docs:
	@echo "--- Starting MkDocs live server ---"
	mkdocs serve

tree:
	python examples/project_tree.py

american:
	python examples/demo_american.py

general:
	python examples/demo_general.py

european:
	@echo "--- Running demo via the 'optpricing' CLI ---"
	optpricing demo
