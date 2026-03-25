# Contributing

Contributions are welcome. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/Mandark-droid/genai_otel_instrument.git
cd genai_otel_instrument
pip install -e ".[dev,all]"
```

## Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=genai_otel --cov-report=term

# Specific test file
pytest tests/test_config.py -v

# Specific instrumentor
pytest tests/instrumentors/test_openai_instrumentor.py -v
```

## Code Quality

```bash
black genai_otel tests          # Format (line length: 100)
isort genai_otel tests          # Sort imports
pylint genai_otel               # Lint
mypy genai_otel                 # Type check
```

## Adding a New Instrumentor

1. Create `genai_otel/instrumentors/{provider}_instrumentor.py`
2. Inherit from `BaseInstrumentor` and implement `instrument()` and `_extract_usage()`
3. Add to `INSTRUMENTORS` dict in `auto_instrument.py`
4. Add optional dependency to `pyproject.toml`
5. Create tests in `tests/instrumentors/test_{provider}_instrumentor.py`
6. Update `DEFAULT_INSTRUMENTORS` in `config.py` if enabled by default
7. Add pricing to `llm_pricing.json`

## Commit Messages

Use conventional commits:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation
- `test:` - Test changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance

## Community

- [Discord](https://discord.gg/6SVz6VKK)
- [GitHub Issues](https://github.com/Mandark-droid/genai_otel_instrument/issues)
