# Quick Release Guide

## TL;DR - Release Checklist

```bash
# 1. Update CHANGELOG.md with release notes
# 2. Commit changes
git add .
git commit -m "chore: Prepare release v0.1.14"
git push origin main

# 3. Create and push tag
git tag v0.1.14
git push origin v0.1.14

# 4. Create GitHub release (triggers auto-publish)
gh release create v0.1.14 --title "v0.1.14" --notes-file <(sed -n '/## \[0.1.14\]/,/## \[/p' CHANGELOG.md | head -n -1)
```

## What Happens Automatically?

When you create a GitHub release:

1. ✅ **Tests run** - Full pytest suite with coverage
2. ✅ **Quality checks** - black, isort formatting validation
3. ✅ **Package builds** - Creates wheel and source distribution
4. ✅ **Installation test** - Verifies package can be installed
5. 📦 **Publishes to TestPyPI** - Test environment first
6. ⏳ **Waits 30s** - Allows TestPyPI to process
7. ✅ **Verifies TestPyPI** - Attempts installation from test server
8. 📦 **Publishes to PyPI** - Production release
9. 📋 **Creates summary** - Release report in GitHub Actions

## Version Format

```
v{MAJOR}.{MINOR}.{PATCH}[-{PRE-RELEASE}]

Examples:
  v0.1.14          # Patch release
  v0.2.0           # Minor release
  v1.0.0           # Major release
  v0.2.0-rc.1      # Release candidate
  v0.2.0-beta.2    # Beta release
```

## Common Commands

```bash
# Check current version
git describe --tags --abbrev=0

# List all tags
git tag -l

# Delete local tag (if mistake)
git tag -d v0.1.14

# Delete remote tag (if mistake)
git push origin :refs/tags/v0.1.14

# View workflow status
gh run list --workflow=publish.yml

# View workflow logs
gh run view --log

# Manual workflow trigger (for testing)
gh workflow run publish.yml
```

## Quick Fixes

### "Tests failed" in workflow
```bash
# Run tests locally first
pytest tests/ -v

# Fix issues and recommit
git add .
git commit -m "fix: Address test failures"
git push

# Delete and recreate tag
git tag -d v0.1.14
git push origin :refs/tags/v0.1.14
git tag v0.1.14
git push origin v0.1.14
```

### "Package already exists" error
```bash
# Can't overwrite published versions
# Must increment version
git tag v0.1.15
git push origin v0.1.15
gh release create v0.1.15 --title "v0.1.15" --notes "Hotfix release"
```

### View published packages
- TestPyPI: https://test.pypi.org/project/genai-otel-instrument/
- PyPI: https://pypi.org/project/genai-otel-instrument/

## Secrets Location

GitHub Repository Settings → Secrets and variables → Actions:
- `TEST_PYPI_API_TOKEN` - For test.pypi.org
- `PYPI_API_TOKEN` - For pypi.org

---

**For detailed information, see [RELEASE_GUIDE.md](RELEASE_GUIDE.md)**
