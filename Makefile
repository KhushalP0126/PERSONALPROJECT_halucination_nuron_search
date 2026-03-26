SHELL := /bin/zsh
.DEFAULT_GOAL := help

VENV ?= venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

MODEL ?=
QUESTION ?= What is the capital of France?
PROMPT ?=
TEMPERATURE ?= 0
MAX_NEW_TOKENS ?= 80

SEED_QUESTIONS ?= data/seed_questions.jsonl
SCORED_HIDDEN_OUT ?= results/scored_hidden_dataset.jsonl

CONSENSUS_SEED_DATASET ?= data/consensus_seed_dataset.json
CONSENSUS_DATASET_OUT ?= results/consensus_dataset.json
LAYER_SUPPORT_IN ?= $(CONSENSUS_DATASET_OUT)

TRUTHFULQA_LIMIT ?= 50
TRUTHFULQA_PAIRS ?= data/truthfulqa_pairs.json
TRUTHFULQA_BENCHMARK_OUT ?= results/truthfulqa_consensus_benchmark.json

VIS_INPUT ?= $(TRUTHFULQA_BENCHMARK_OUT)
VIS_OUTPUT_DIR ?= results/consensus_plots

NEURON_INPUT ?= $(CONSENSUS_DATASET_OUT)
NEURON_OUTPUT_DIR ?= results/neuron_contributions
NEURON_SAMPLE_INDEX ?= 0
NEURON_LAYER_MODE ?= auto
NEURON_LAYER ?=
NEURON_TOP_K ?= 20
QUESTION_CONTAINS ?=

LIMIT ?=
INPUT ?=
OUTPUT_DIR ?=
SAMPLE ?=
LAYER ?=
TOP_K ?=
QUESTION_MATCH ?=

RUN_QUESTION := $(if $(PROMPT),$(PROMPT),$(QUESTION))
RUN_LIMIT := $(if $(LIMIT),$(LIMIT),$(TRUTHFULQA_LIMIT))
RUN_VIS_INPUT := $(if $(INPUT),$(INPUT),$(VIS_INPUT))
RUN_VIS_OUTPUT_DIR := $(if $(OUTPUT_DIR),$(OUTPUT_DIR),$(VIS_OUTPUT_DIR))
RUN_NEURON_INPUT := $(if $(INPUT),$(INPUT),$(NEURON_INPUT))
RUN_NEURON_OUTPUT_DIR := $(if $(OUTPUT_DIR),$(OUTPUT_DIR),$(NEURON_OUTPUT_DIR))
RUN_SAMPLE := $(if $(SAMPLE),$(SAMPLE),$(NEURON_SAMPLE_INDEX))
RUN_LAYER := $(if $(LAYER),$(LAYER),$(NEURON_LAYER))
RUN_TOP_K := $(if $(TOP_K),$(TOP_K),$(NEURON_TOP_K))
RUN_QUESTION_MATCH := $(if $(QUESTION_MATCH),$(QUESTION_MATCH),$(QUESTION_CONTAINS))

.PHONY: help venv install setup ensure-venv check-mps \
	chat ask prompt \
	hidden-dataset hidden-one build-hidden-dataset build-hidden-dataset-one \
	consensus build-consensus-dataset \
	layers summarize-layer-support \
	truthfulqa prepare-truthfulqa \
	benchmark benchmark-truthfulqa \
	plots visualize-consensus \
	neurons analyze-neurons

help:
	@echo "Recommended targets:"
	@echo "  make setup"
	@echo "  make check-mps"
	@echo "  make ask QUESTION='What is the capital of France?'"
	@echo "  make hidden-dataset"
	@echo "  make consensus"
	@echo "  make layers"
	@echo "  make truthfulqa LIMIT=20"
	@echo "  make benchmark LIMIT=20"
	@echo "  make plots INPUT=results/truthfulqa_consensus_benchmark.json"
	@echo "  make neurons QUESTION_MATCH='capital of France' LAYER=19"
	@echo ""
	@echo "Legacy aliases still work:"
	@echo "  prompt build-hidden-dataset build-consensus-dataset summarize-layer-support"
	@echo "  prepare-truthfulqa benchmark-truthfulqa visualize-consensus analyze-neurons"
	@echo ""
	@echo "Useful variables:"
	@echo "  MODEL='microsoft/phi-2'"
	@echo "  QUESTION='What is the capital of Germany?'"
	@echo "  TEMPERATURE=0"
	@echo "  MAX_NEW_TOKENS=80"
	@echo "  LIMIT=50"
	@echo "  INPUT=results/consensus_dataset.json"
	@echo "  OUTPUT_DIR=results/consensus_plots"
	@echo "  SAMPLE=0"
	@echo "  LAYER=19"
	@echo "  TOP_K=20"
	@echo "  QUESTION_MATCH='capital of France'"

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

ask prompt: ensure-venv
	$(PYTHON) local_chat.py $(if $(MODEL),--model "$(MODEL)") --prompt "$(RUN_QUESTION)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

hidden-dataset build-hidden-dataset: ensure-venv
	$(PYTHON) build_scored_hidden_dataset.py $(if $(MODEL),--model "$(MODEL)") --questions-file "$(SEED_QUESTIONS)" --out "$(SCORED_HIDDEN_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

hidden-one build-hidden-dataset-one: ensure-venv
	$(PYTHON) build_scored_hidden_dataset.py $(if $(MODEL),--model "$(MODEL)") --question "$(RUN_QUESTION)" --out "$(SCORED_HIDDEN_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

consensus build-consensus-dataset: ensure-venv
	$(PYTHON) build_consensus_dataset.py $(if $(MODEL),--model "$(MODEL)") --dataset "$(CONSENSUS_SEED_DATASET)" --out "$(CONSENSUS_DATASET_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

layers summarize-layer-support: ensure-venv
	$(PYTHON) summarize_layer_support.py --in "$(LAYER_SUPPORT_IN)"

truthfulqa prepare-truthfulqa: ensure-venv
	$(PYTHON) prepare_truthfulqa_dataset.py --out "$(TRUTHFULQA_PAIRS)" --limit $(RUN_LIMIT)

benchmark benchmark-truthfulqa: ensure-venv
	$(PYTHON) benchmark_truthfulqa_consensus.py $(if $(MODEL),--model "$(MODEL)") --dataset "$(TRUTHFULQA_PAIRS)" --out "$(TRUTHFULQA_BENCHMARK_OUT)" --limit $(RUN_LIMIT) --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

plots visualize-consensus: ensure-venv
	$(PYTHON) visualize_consensus_patterns.py --in "$(RUN_VIS_INPUT)" --out-dir "$(RUN_VIS_OUTPUT_DIR)"

neurons analyze-neurons: ensure-venv
	$(PYTHON) analyze_neuron_contributions.py $(if $(MODEL),--model "$(MODEL)") --in "$(RUN_NEURON_INPUT)" --out-dir "$(RUN_NEURON_OUTPUT_DIR)" --sample-index $(RUN_SAMPLE) --layer-mode "$(NEURON_LAYER_MODE)" --top-k $(RUN_TOP_K) $(if $(RUN_LAYER),--layer $(RUN_LAYER)) $(if $(RUN_QUESTION_MATCH),--question-contains "$(RUN_QUESTION_MATCH)")
