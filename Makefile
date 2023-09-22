all: test wheel

test:
	PYTHONPATH=. pytest

wheel:
	pip wheel .