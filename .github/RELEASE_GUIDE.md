# Release Guide

This guide explains how to publish new versions of `genai-otel-instrument` to PyPI.

## Automated Publishing Workflow

The repository is configured to automatically publish to PyPI when you create a GitHub release. The workflow:

1. ✅ Runs full test suite
2. ✅ Checks code quality (black, isort)
3. ✅ Builds the package
4. ✅ Tests package installation
5. 📦 Publishes to Test PyPI
6. ✅ Verifies Test PyPI upload
7. 📦 Publishes to production PyPI
8. 📋 Creates release summary

## Prerequisites

Before creating a release, ensure:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is properly formatted: `black genai_otel tests && isort genai_otel tests`
- [ ] CHANGELOG.md is updated with release notes
- [ ] Version is bumped (handled by `setuptools_scm` based on git tags)
- [ ] All changes are committed and pushed to `main` branch

## Creating a Release

### Method 1: GitHub UI (Recommended)

1. Navigate to https://github.com/YOUR_ORG/genai-otel-instrument/releases/new

2. Click "Choose a tag" and create a new tag:
   - Format: `v{MAJOR}.{MINOR}.{PATCH}` (e.g., `v0.1.14`)
   - Target: `main` branch

3. Fill in release details:
   - **Release title**: Same as tag (e.g., `v0.1.14`)
   - **Description**: Copy relevant section from CHANGELOG.md

4. Click "Publish release"

5. Monitor the workflow:
   - Go to Actions tab
   - Watch "Publish to PyPI" workflow

### Method 2: GitHub CLI

```bash
# Create and push a tag
git tag -a v0.1.14 -m "Release v0.1.14: Added LangChain chat model support"
git push origin v0.1.14

# Create release from tag
gh release create v0.1.14 \
  --title "v0.1.14" \
  --notes-file CHANGELOG.md \
  --target main
```

### Method 3: Manual Trigger (Testing)

For testing the workflow without creating a release:

1. Go to Actions tab
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. Select branch and click "Run workflow"

**Note**: Manual triggers will run tests but won't publish to PyPI (only releases trigger publishing).

## Versioning Strategy

This project uses `setuptools_scm` for automatic versioning:

- **Version source**: Git tags
- **Format**: Semantic Versioning (MAJOR.MINOR.PATCH)
- **Pre-releases**: Add suffix like `-alpha.1`, `-beta.1`, `-rc.1`

### Version Bumping Guidelines

- **MAJOR**: Breaking changes, incompatible API changes
- **MINOR**: New features, backward-compatible additions
- **PATCH**: Bug fixes, backward-compatible improvements

### Examples

```bash
# Patch release (bug fixes)
git tag v0.1.14

# Minor release (new features)
git tag v0.2.0

# Major release (breaking changes)
git tag v1.0.0

# Pre-release
git tag v0.2.0-rc.1
```

## Secrets Configuration

The workflow uses these GitHub secrets (already configured):

- `TEST_PYPI_API_TOKEN`: Token for https://test.pypi.org
- `PYPI_API_TOKEN`: Token for https://pypi.org

### Rotating Tokens

If tokens need rotation:

1. Generate new tokens:
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

2. Update GitHub secrets:
   - Go to repository Settings → Secrets and variables → Actions
   - Update `TEST_PYPI_API_TOKEN` and `PYPI_API_TOKEN`

## Manual Publishing (Fallback)

If automated workflow fails, publish manually:

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Publish to Test PyPI
python -m twine upload --verbose --repository testpypi --username __token__ --password YOUR_TEST_PYPI_TOKEN dist/*

# Verify Test PyPI installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ genai-otel-instrument

# Publish to PyPI
python -m twine upload --verbose dist/* --username __token__ --password YOUR_PYPI_TOKEN
```

## Post-Release Checklist

After publishing:

- [ ] Verify package on PyPI: https://pypi.org/project/genai-otel-instrument/
- [ ] Test installation: `pip install genai-otel-instrument --upgrade`
- [ ] Update documentation if needed
- [ ] Announce release (if major version)
- [ ] Monitor GitHub issues for installation problems

## Troubleshooting

### Workflow fails on test step
- Review test failures in Actions logs
- Fix issues locally and create new release

### "File already exists" error on PyPI
- Package version already published
- Bump version number and create new release
- **Note**: Cannot overwrite existing PyPI versions

### Test PyPI upload succeeds but PyPI fails
- Check PyPI token is valid
- Verify package metadata
- Check for naming conflicts

### Secrets not working
- Verify secrets are set in repository settings
- Check secret names match workflow file exactly
- Ensure tokens have correct permissions

## References

- [Python Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)
