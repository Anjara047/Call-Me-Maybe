PYTHON = python3
PYTHON_VERSION = 3.13.1

GOINFRE_CACHE = $(HOME)/sgoinfre/call_me_maybe_cache
GOINFRE_VENV = $(HOME)/sgoinfre/call_me_maybe_venv
PYTHON_INSTALL_DIR = $(GOINFRE_CACHE)/python

.PHONY: all install run debug clean fclean lint

all: install run

install:
	@mkdir -p $(GOINFRE_CACHE)/huggingface
	@mkdir -p $(GOINFRE_CACHE)/uv
	@UV_PYTHON_INSTALL_DIR=$(PYTHON_INSTALL_DIR) \
		uv python install $(PYTHON_VERSION) >/dev/null 2>&1
	@HF_HOME=$(GOINFRE_CACHE)/huggingface \
		UV_CACHE_DIR=$(GOINFRE_CACHE)/uv \
		UV_PROJECT_ENVIRONMENT=$(GOINFRE_VENV) \
		UV_PYTHON_INSTALL_DIR=$(PYTHON_INSTALL_DIR) \
		uv sync --python $(PYTHON_VERSION)

run:
	@HF_HOME=$(GOINFRE_CACHE)/huggingface \
		UV_CACHE_DIR=$(GOINFRE_CACHE)/uv \
		UV_PROJECT_ENVIRONMENT=$(GOINFRE_VENV) \
		UV_PYTHON_INSTALL_DIR=$(PYTHON_INSTALL_DIR) \
		uv run --no-sync python -m src

debug:
	@HF_HOME=$(GOINFRE_CACHE)/huggingface \
		UV_CACHE_DIR=$(GOINFRE_CACHE)/uv \
		UV_PROJECT_ENVIRONMENT=$(GOINFRE_VENV) \
		UV_PYTHON_INSTALL_DIR=$(PYTHON_INSTALL_DIR) \
		uv run --no-sync python -m pdb -m src

clean:
	@rm -rf __pycache__
	@rm -rf .mypy_cache
	@rm -rf src/__pycache__
	@rm -rf src/.mypy_cache
	@rm -rf src/models/__pycache__
	@rm -rf src/models/.mypy_pycache

fclean: clean
	@rm -rf .venv
	@rm -rf $(GOINFRE_CACHE)
	@rm -rf $(GOINFRE_VENV)
	@rm -rf data/output

lint:
	-@flake8 src/
	-@mypy src/ --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs