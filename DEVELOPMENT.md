# Development Guide

## Pre-Commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to automatically enforce code quality standards before every commit.

### What Pre-Commit Does

Pre-commit hooks automatically run on every `git commit` to:

1. **Format Code**: Applies `black` formatting (line length: 100)
2. **Sort Imports**: Organizes imports with `isort` (black-compatible)
3. **Fix Basic Issues**:
   - Remove trailing whitespace
   - Fix end-of-file issues
   - Normalize line endings to LF
   - Check YAML syntax
   - Prevent large files from being committed
4. **Security Checks**: Runs `bandit` to detect common security issues

### Setup

Pre-commit hooks are already installed if you cloned this repo. If not:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
```

### Usage

**Automatic (Recommended)**
- Hooks run automatically on `git commit`
- If issues are found and auto-fixed, the commit will fail
- Simply run `git add` and `git commit` again

**Manual Run**
```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files genai_otel/config.py

# Run a specific hook
pre-commit run black --all-files
```

**Skip Hooks (Emergency Only)**
```bash
# Skip pre-commit hooks (NOT RECOMMENDED)
git commit --no-verify

# Skip pre-push hooks
git push --no-verify
```

### Adding New Hooks

Edit `.pre-commit-config.yaml` to add new hooks. See [pre-commit.com](https://pre-commit.com/hooks.html) for available hooks.

## Code Quality Standards

### Formatting
- **Black**: Line length 100, default settings
- **isort**: Black-compatible profile

```bash
# Format code
black genai_otel tests --line-length 100

# Sort imports
isort genai_otel tests --profile black --line-length 100
```

### Linting
- **Pylint**: Target score 9.0+/10.0

```bash
# Run pylint
pylint genai_otel --disable=C0301,C0103,R0913,R0914,R0915,R0912,W0718,W0719

# Or use the shorter command
pylint genai_otel --exit-zero
```

### Testing
- **Coverage Target**: 85%+
- **All tests must pass** before committing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=genai_otel --cov-report=html --cov-report=term

# Quick test
pytest tests/ --tb=no -q
```

## CI/CD Integration

The pre-commit hooks ensure that:
1. ✅ CI/CD pipelines won't fail due to formatting issues
2. ✅ Code style is consistent across all contributors
3. ✅ Security issues are caught early
4. ✅ Common mistakes are prevented

## Troubleshooting

### Hook Fails After Fixing

If a hook auto-fixes files, the commit will fail. This is expected:

```bash
# Files were auto-fixed by pre-commit
git add .
git commit -m "your message"  # This will now succeed
```

### Disable Specific Hook

Edit `.pre-commit-config.yaml` and comment out the hook you want to disable.

### Update Hook Versions

```bash
# Update all hooks to latest versions
pre-commit autoupdate
```

### Clean Hook Cache

```bash
# Clear pre-commit cache
pre-commit clean
```

## Best Practices

1. **Run pre-commit frequently** during development:
   ```bash
   pre-commit run --all-files
   ```

2. **Fix issues incrementally** rather than at commit time

3. **Don't skip hooks** unless absolutely necessary

4. **Keep hook configuration updated** with latest versions

5. **Add new hooks** when adopting new tools/standards

## Related Documentation

- [Pre-commit Official Docs](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [Bandit Security](https://bandit.readthedocs.io/)
