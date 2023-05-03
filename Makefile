all: test wheel

test:
	PYTHONPATH=. pytest

wheel:
	python -m build --wheel .