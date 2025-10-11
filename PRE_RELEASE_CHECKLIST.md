# Pre-Release Checklist for genai-otel-instrument

Complete this checklist before publishing to PyPI.

## Documentation
- [ ] README.md is complete and accurate
- [ ] CHANGELOG.md is updated with version info
- [ ] All code has docstrings
- [ ] Usage examples are provided
- [ ] Installation instructions are clear

## Code Quality
- [ ] All tests pass locally
- [ ] Code coverage is >80%
- [ ] No pylint errors (warnings acceptable)
- [ ] Black formatting applied
- [ ] isort applied
- [ ] Type hints added where possible

## Package Structure
- [x] pyproject.toml is complete
- [x] setup.py is correct
- [x] MANIFEST.in includes all necessary files
- [x] LICENSE file exists
- [x] __version__.py exists
- [x] py.typed marker file exists

## Dependencies
- [x] Core dependencies are minimal
- [x] Optional dependencies are properly configured
- [x] All import statements are conditional
- [x] No hardcoded library imports at module level

## Testing
- [ ] Run tests on Python 3.8, 3.9, 3.10, 3.11, 3.12
- [ ] Test on Linux, macOS, Windows
- [ ] Test clean install: `pip install -e .`
- [ ] Test with optional deps: `pip install -e ".[all]"`
- [ ] Test CLI tool works: `genai-instrument --help`

## Security
- [ ] No hardcoded secrets or API keys
- [ ] Sensitive data not logged
- [ ] Run `safety check`
- [ ] Run `bandit -r genai_otel`
- [ ] Dependencies reviewed for vulnerabilities

## Git & GitHub
- [ ] All changes committed
- [ ] Version tag created (e.g., v0.1.0)
- [ ] GitHub repository is public
- [ ] .gitignore is properly configured
- [ ] CI/CD workflows are set up
- [ ] Branch protection rules configured

## PyPI Preparation
- [ ] PyPI account created
- [ ] API token generated
- [ ] Test PyPI account created
- [ ] Test PyPI API token generated
- [ ] GitHub secrets configured:
  - [ ] PYPI_API_TOKEN
  - [ ] TEST_PYPI_API_TOKEN

## Build & Test Release
- [ ] Build package: `python -m build`
- [ ] Check package: `twine check dist/*`
- [ ] Test upload to Test PyPI:
  ```bash
  twine upload --repository testpypi dist/*
  ```
- [ ] Test install from Test PyPI:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ genai-otel-instrument
  ```
- [ ] Verify installation works
- [ ] Test basic functionality

## Final Checks
- [ ] Package name is available on PyPI
- [ ] Version number follows semantic versioning
- [ ] All URLs in setup.py are correct
- [ ] Author email is correct
- [ ] License is correctly specified
- [ ] Keywords are relevant
- [ ] Classifiers are appropriate

## Release Process
1. [ ] Update version in `__version__.py`
2. [ ] Update CHANGELOG.md
3. [ ] Commit changes
4. [ ] Create git tag: `git tag v0.1.0`
5. [ ] Push tag: `git push origin v0.1.0`
6. [ ] Create GitHub release
7. [ ] Publish to PyPI (manual or via GitHub Actions)
8. [ ] Verify package on PyPI
9. [ ] Test installation: `pip install genai-otel-instrument`
10. [ ] Announce release (if applicable)

## Post-Release
- [ ] Monitor PyPI download stats
- [ ] Watch for bug reports
- [ ] Respond to GitHub issues
- [ ] Update documentation if needed
- [ ] Plan next release