SHELL := /bin/zsh

VENV ?= venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

MODEL ?=
PROMPT ?= What is the capital of France?
TEMPERATURE ?= 0
MAX_NEW_TOKENS ?= 80

DAY2_FILE ?= data/day2_questions.jsonl
DAY2_OUT ?= results/day2_results.jsonl

DAY3_DATASET ?= data/base_dataset.json
DAY3_OUT ?= results/consensus_dataset.json
CONSENSUS_IN ?= $(DAY3_OUT)

TRUTHFULQA_LIMIT ?= 50
TRUTHFULQA_DATASET ?= data/truthfulqa_balanced.json
TRUTHFULQA_OUT ?= results/truthfulqa_consensus.json

.PHONY: help venv install setup ensure-venv check-mps chat prompt day2 day2-question day3 analyze-layers truthfulqa-prepare truthfulqa-consensus

help:
	@echo "Available targets:"
	@echo "  make setup"
	@echo "  make check-mps"
	@echo "  make chat"
	@echo "  make prompt PROMPT='What is the capital of France?'"
	@echo "  make day2"
	@echo "  make day2-question PROMPT='What is the capital of France?'"
	@echo "  make day3"
	@echo "  make analyze-layers"
	@echo "  make truthfulqa-prepare"
	@echo "  make truthfulqa-consensus"
	@echo ""
	@echo "Optional variables:"
	@echo "  MODEL='microsoft/phi-2'"
	@echo "  TEMPERATURE=0"
	@echo "  MAX_NEW_TOKENS=80"
	@echo "  DAY2_OUT=results/day2_results.jsonl"
	@echo "  DAY3_OUT=results/consensus_dataset.json"
	@echo "  TRUTHFULQA_LIMIT=50"
	@echo "  TRUTHFULQA_OUT=results/truthfulqa_consensus.json"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup: install

ensure-venv:
	@test -x "$(PYTHON)" || (echo "Missing $(PYTHON). Run 'make setup' first." && exit 1)

check-mps: ensure-venv
	$(PYTHON) check_mps.py

chat: ensure-venv
	$(PYTHON) local_chat.py $(if $(MODEL),--model "$(MODEL)") --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

prompt: ensure-venv
	$(PYTHON) local_chat.py $(if $(MODEL),--model "$(MODEL)") --prompt "$(PROMPT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

day2: ensure-venv
	$(PYTHON) day2_pipeline.py $(if $(MODEL),--model "$(MODEL)") --questions-file "$(DAY2_FILE)" --out "$(DAY2_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

day2-question: ensure-venv
	$(PYTHON) day2_pipeline.py $(if $(MODEL),--model "$(MODEL)") --question "$(PROMPT)" --out "$(DAY2_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

day3: ensure-venv
	$(PYTHON) day3_consensus.py $(if $(MODEL),--model "$(MODEL)") --dataset "$(DAY3_DATASET)" --out "$(DAY3_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

analyze-layers: ensure-venv
	$(PYTHON) analyze_layer_locations.py --in "$(CONSENSUS_IN)"

truthfulqa-prepare: ensure-venv
	$(PYTHON) prepare_truthfulqa.py --out "$(TRUTHFULQA_DATASET)" --limit $(TRUTHFULQA_LIMIT)

truthfulqa-consensus: ensure-venv
	$(PYTHON) truthfulqa_consensus.py $(if $(MODEL),--model "$(MODEL)") --dataset "$(TRUTHFULQA_DATASET)" --out "$(TRUTHFULQA_OUT)" --limit $(TRUTHFULQA_LIMIT) --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)
