PYTHON ?= .venv/bin/python
test:
	$(PYTHON) -m pytest -q
compile:
	PYTHONPYCACHEPREFIX=/tmp/rag-pycache $(PYTHON) -m compileall -q app
run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000
evaluation:
	$(PYTHON) -m app.evaluation.runner
benchmark:
	$(PYTHON) -m app.evaluation.benchmark
