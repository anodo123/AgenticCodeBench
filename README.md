# Agentic Code Bench

A hands-on playground for building and evaluating agentic systems through progressively harder coding challenges.

## Why?

The project is built around a deliberate learning loop:

**WRITE CODE → RUN CHALLENGE → FAIL → UNDERSTAND → FIX → EVALUATE → REPEAT**

Agentic Code Bench is in early development. The repository currently contains only the initial project setup.

## Development

This project requires Python 3.12 or newer. Install the development dependencies with:

```bash
python -m pip install -e ".[dev]"
```

Run the available checks with:

```bash
pytest
ruff check .
mypy src
```
