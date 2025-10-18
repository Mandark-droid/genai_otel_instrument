# CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing, validation, and publishing.

## Workflows

### 1. `test.yml` - Continuous Integration Tests

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Jobs:**

#### `test` job
- **Matrix:** Runs on Ubuntu, macOS, and Windows with Python 3.8-3.12
- **Steps:**
  1. Lint with pylint
  2. Check formatting with black
  3. Check import sorting with isort
  4. Type check with mypy
  5. Run pytest with coverage
  6. Upload coverage to Codecov

#### `build-and-install-test` job
- **Matrix:** Runs on Ubuntu and Windows with Python 3.9 and 3.12
- **Purpose:** Validates that the package can be built and installed correctly
- **Steps:**
  1. Build the package using `python -m build`
  2. Check package with `twine check`
  3. Install in isolated virtual environment
  4. Test core imports
  5. Verify CLI tool works (`genai-instrument --help`)

#### `security` job
- **Platform:** Ubuntu with Python 3.11
- **Steps:**
  1. Run safety check for dependency vulnerabilities
  2. Run bandit security scan

### 2. `publish.yml` - PyPI Publication

**Triggers:**
- Release published on GitHub
- Manual workflow dispatch (for testing)

**Steps:**
1. **Test Suite:** Runs full pytest suite to ensure all tests pass
2. **Code Quality:** Validates formatting and import sorting
3. **Build:** Creates wheel and sdist packages
4. **Validation:** Checks package with twine
5. **Installation Test:** Tests package installation in clean environment
6. **Publish to Test PyPI:** Uploads to Test PyPI (if release event)
7. **Publish to PyPI:** Uploads to production PyPI (if release event)

**Required Secrets:**
- `TEST_PYPI_API_TOKEN`: API token for Test PyPI
- `PYPI_API_TOKEN`: API token for production PyPI

### 3. `pre-release-check.yml` - Pre-Release Validation

**Triggers:**
- Manual workflow dispatch
- Push of version tags (`v*`)

**Purpose:** Comprehensive validation before creating a release (mimics `scripts/test_release.sh`)

**Matrix:** Runs on Ubuntu, Windows, and macOS with Python 3.9 and 3.12

**Steps:**
1. Display environment information
2. Install development dependencies
3. Auto-format code (isort + black)
4. Run full test suite with coverage
5. Perform code quality checks
6. Build package
7. Validate package with twine
8. Test installation in isolated environment
9. Verify CLI functionality
10. Upload build artifacts (Ubuntu + Python 3.12 only)

**Use this workflow:**
- Before creating a new release
- To validate your changes work across all platforms
- As a final check that everything is ready for publication

## Usage Examples

### Running Pre-Release Validation Manually

1. Go to "Actions" tab in GitHub
2. Select "Pre-Release Validation" workflow
3. Click "Run workflow"
4. Select branch and click "Run workflow"

This will run all validation checks across all platforms.

### Creating a Release

1. **Run Pre-Release Validation:**
   ```bash
   # Locally, you can use:
   ./scripts/test_release.sh

   # Or trigger the GitHub Action manually
   ```

2. **Ensure all checks pass**

3. **Create and push a version tag:**
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
   This triggers the pre-release-check workflow.

4. **Create GitHub Release:**
   - Go to "Releases" → "Draft a new release"
   - Select the tag you created
   - Fill in release notes
   - Click "Publish release"

   This triggers the publish workflow which will:
   - Run tests
   - Build package
   - Publish to Test PyPI
   - Publish to PyPI

### Manual Testing Before Release

Use the local script for faster iteration:

```bash
# From project root
./scripts/test_release.sh
```

This script provides colored output and validates:
- Project structure
- Python environment
- Code formatting (auto-applies)
- Full test suite
- Package build
- Package validation
- Installation in isolated environment
- CLI functionality

## Workflow Dependencies

All workflows require these tools (automatically installed):
- `pytest` - Testing framework
- `pytest-cov` - Coverage plugin
- `black` - Code formatter
- `isort` - Import sorter
- `pylint` - Linter
- `mypy` - Type checker
- `build` - Package builder
- `twine` - Package uploader

## Best Practices

1. **Before committing:**
   - Run `black genai_otel tests`
   - Run `isort genai_otel tests`
   - Run `pytest tests/`

2. **Before creating a PR:**
   - Ensure all CI checks pass
   - Review coverage report

3. **Before releasing:**
   - Run `./scripts/test_release.sh` locally
   - Trigger "Pre-Release Validation" workflow
   - Verify all platforms pass
   - Update CHANGELOG.md
   - Update version in `genai_otel/__version__.py`

4. **After releasing:**
   - Test installation from PyPI: `pip install genai-otel-instrument`
   - Verify functionality with example code
   - Monitor for any user-reported issues

## Troubleshooting

### Tests fail only on specific OS
- Check for platform-specific path handling
- Review file permission issues (especially on Windows)
- Check line ending differences (CRLF vs LF)

### Package build fails
- Ensure `pyproject.toml` is valid
- Check MANIFEST.in includes all necessary files
- Verify dependencies are correctly specified

### Installation test fails
- Check entry points in `pyproject.toml`
- Verify all required files are included in the package
- Test locally with: `pip install dist/*.whl`

### CLI tool not found after installation
- Verify `[project.scripts]` section in `pyproject.toml`
- Check that console_scripts entry point is correct
- Test with: `python -m genai_otel.cli`
