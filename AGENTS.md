# Repository Guidelines

## Project Structure & Module Organization
- `src/vifact/`: core library
  - `retrieval/`, `rerank/`, `verify/`, `reasoner/`, `explainer/`, `prep/`
  - `pipeline.py`, `config.py`, `data.py`
- `scripts/`: CLI utilities (prepare data, build index, run pipeline, train/evaluate)
- `docs/`: dataset notes and usage guides
- `requirements.txt`: Python dependencies

## Build, Test, and Development Commands
- Install: `pip install -r requirements.txt`
- Add path (Colab/local): `python -c "import sys; sys.path.append('src')"`
- Prepare KV JSON dataset: `python scripts/prepare_ise_kvjson.py --input ise_train.json --out_dir ./prepared`
- Run pipeline (single claim):
  `python scripts/run_pipeline.py --corpus prepared/corpus.csv --format csv --claim "..." --bm25 prepared/bm25.json --topk 10`
- Train Reasoner: `python scripts/train_reasoner.py --train prepared/train.jsonl --valid prepared/valid.jsonl --out_dir ./ckpt/reasoner`
- Evaluate Reasoner: `python scripts/evaluate_reasoner.py --valid prepared/valid.jsonl --ckpt ./ckpt/reasoner`

## Coding Style & Naming Conventions
- Language: Python 3.10+
- Style: PEP 8, 4‑space indentation, type hints required for public APIs
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants
- Docstrings: short module/class/function docstrings; document CLI args
- Keep changes minimal, localized, and consistent with existing modules

## Testing Guidelines
- No formal suite included yet. Add targeted smoke tests or pytest when contributing features.
- Suggested layout: `tests/` mirrors `src/vifact/` (e.g., `tests/reasoner/test_model.py`).
- Run pytest (if added): `pytest -q`
- Validate with real samples: run `prepare_ise_kvjson.py`, then `train_reasoner.py` and `run_pipeline.py` on a few claims.

## Commit & Pull Request Guidelines
- Use descriptive commits; prefer Conventional Commits (e.g., `feat(retrieval): add RRF fusion`, `fix(pipeline): guard empty evidences`).
- Scope paths in subject if helpful (e.g., `scripts/train_reasoner.py`).
- PRs must include: summary, motivation, before/after behavior, run logs or examples (commands + key output), and any risks.
- Link issues when applicable; keep PRs focused and reviewable.

## Security & Configuration Tips
- Models download from Hugging Face; avoid committing checkpoints.
- Mind GPU memory (reduce `--max_evidences`, sequence lengths, batch size).
- Ensure dataset text is UTF‑8; chunk long contexts for retrieval and NLI.
