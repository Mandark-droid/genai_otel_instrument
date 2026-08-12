# Contributing to genai-otel-instrument

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and constructive in all interactions. We're here to build something useful together.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/Mandark-droid/genai_otel_instrument/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Relevant logs or error messages

### Suggesting Features

1. Check existing issues and discussions
2. Create a new issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Any alternatives you've considered

### Contributing Code

#### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Mandark-droid/genai_otel_instrument.git
cd genai_otel_instrument

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"
```

#### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Follow existing code style
   - Add tests for new functionality

3. **Run tests**
   ```bash
   pytest tests/ -v --cov=genai_otel
   ```

4. **Check code quality**
   ```bash
   # Format code
   black genai_otel tests
   isort genai_otel tests

   # Lint
   pylint genai_otel

   # Type check
   mypy genai_otel
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for refactoring
   - `chore:` for maintenance

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Pull Request Guidelines

- Keep PRs focused and atomic
- Update documentation as needed
- Add tests for new functionality
- Ensure all tests pass
- Update CHANGELOG.md
- Reference related issues

## Code Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Add type hints where possible
- Write docstrings for public APIs
- Keep functions focused and small

## Testing

- Write tests for all new code
- Aim for >80% code coverage
- Use pytest fixtures for setup
- Mock external dependencies
- Test edge cases and error conditions

### Running Specific Tests

```bash
# Run specific test file
pytest tests/test_config.py -v

# Run specific test
pytest tests/test_config.py::TestOTelConfig::test_default_config -v

# Run with coverage
pytest tests/ --cov=genai_otel --cov-report=html
```

## Adding New Instrumentors

When adding support for a new LLM provider:

1. Create a new file in `genai_otel/instrumentors/`
2. Inherit from `BaseInstrumentor`
3. Implement required methods:
   - `instrument(config)`
   - `_extract_usage(result)`
4. Add conditional imports
5. Add to `auto_instrument.py`
6. Add optional dependency to `setup.py`
7. Write tests
8. Update documentation

Example structure:
```python
"""Instrumentor for XYZ Provider."""

import logging
from typing import Dict, Optional
from .base import BaseInstrumentor
from ..config import OTelConfig

logger = logging.getLogger(__name__)

class XYZInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()
        self._xyz_available = False
        self._check_availability()

    def _check_availability(self):
        try:
            import xyz
            self._xyz_available = True
        except ImportError:
            pass

    def instrument(self, config: OTelConfig):
        if not self._xyz_available:
            return
        # Implementation

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        # Implementation
```

## Renaming a span attribute (read this before you do)

The OpenTelemetry GenAI conventions are still `Status: Development` and have already
renamed attributes more than once. When you follow one of those renames, **emit both
spellings for at least one major version. Never replace.**

This is not tidiness. An attribute rename fails in a uniquely bad way:

- Nothing raises. The old name is simply absent.
- A consumer reading it gets *zero*, not an error — zero tokens, zero cost, no provider.
- Many consumers treat a missing provider or missing token count as **"not a GenAI
  span"** and drop the record entirely rather than showing it unlabelled.

So the failure surfaces as a confident, plausible number on someone's dashboard. That
survives review in a way an exception never would, and it can run for weeks.

We have shipped this bug twice, in both directions:

- Emitting only the *superseded* token names meant backends consuming the current
  conventions read zero tokens and derived zero cost (fixed in 1.9.0).
- Then emitting only the *current* names meant consumers still reading the superseded
  ones read zero (fixed in 1.10.0 by making dual emission the default).

**The rules:**

1. Put the naming policy in **one place** — `genai_otel/semconv.py` for the constants,
   a central helper in `instrumentors/base.py` for emission. Instrumentors must not
   decide naming individually; there are ~29 of them and they will drift.
2. Default to **dual emission**. Consumers include long-lived on-prem deployments that
   cannot be upgraded in lockstep with a library release.
3. Parse `OTEL_SEMCONV_STABILITY_OPT_IN` with `genai_semconv_modes()`. It is a
   **comma-separated list shared by every instrumentation area** — `"dup" in raw` is
   true for `http/dup`, which has nothing to do with GenAI.
4. Drop a superseded name only in a **major** release, and say so in the CHANGELOG
   under an explicit *Upgrading* heading.
5. Assume you cannot see your consumers. This library is published; the code that reads
   these attributes generally lives in another repository, is not in your context, and
   will not fail loudly. Compatibility is your responsibility here, not theirs.

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update CHANGELOG.md
- Add examples for new features

## Questions?

- Open an issue for questions
- Check existing documentation
- Review closed issues and PRs

Thank you for contributing! 🎉
