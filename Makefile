.PHONY: install test run
install:
	pip install -r requirements.txt && pip install -e .
test:
	pytest -q
run:
	python -m rdio.pipeline --out reports
