# Pre-Release Checklist for genai-otel-instrument

Complete this checklist before publishing to PyPI.

## Documentation
- [ ] README.md is complete and accurate
- [x] CHANGELOG.md is updated with version info
- [x] All code has docstrings
- [x] Usage examples are provided
- [x] Create an examples folder with one example for each supported instrumented library and also one demo folder
- [x] Installation instructions are clear

## Code Quality
- [x] All tests pass locally
- [x] Code coverage is >80% (currently at 98%)
- [ ] No pylint errors (warnings acceptable)
- [ ] Black formatting applied
- [ ] isort applied
- [x] Type hints added where possible

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
- [ ] Test on Linux, macOS, Windows (CI workflows need to pass)
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
- [x] .gitignore is properly configured
- [x] CI/CD workflows are set up
- [ ] CI/CD workflows passing (black, isort, tests)
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

## Examples & Demo Validation

### Individual Example Testing
Test each example runs without instrumentation errors:
- [ ] `examples/anthropic/example.py`
- [ ] `examples/aws_bedrock/example.py`
- [ ] `examples/azure_openai/example.py`
- [ ] `examples/cohere/example.py`
- [ ] `examples/google_ai/example.py`
- [ ] `examples/groq/example.py`
- [ ] `examples/huggingface/example.py`
- [ ] `examples/langchain/example.py`
- [ ] `examples/litellm/example.py`
- [ ] `examples/llamaindex/example.py`
- [ ] `examples/mistralai/example.py`
- [ ] `examples/ollama/example.py`
- [ ] `examples/openai/example.py`
- [ ] `examples/replicate/example.py`
- [ ] `examples/smolagents/example.py`
- [ ] `examples/togetherai/example.py`
- [ ] `examples/vertexai/example.py`

### Demo Validation
- [ ] Run `demo/` folder examples successfully
- [ ] Verify demo showcases complete observability pipeline
- [ ] Test demo with OTLP backend (Jaeger/Grafana/etc.)

### Documentation Validation
- [ ] Confirm all examples have proper README.md files
- [ ] Verify README includes API key setup instructions
- [ ] Check README shows expected output format
- [ ] Validate examples match documentation

### Functional Validation
- [ ] Verify all examples show telemetry output (traces/metrics)
- [ ] Check that examples handle missing API keys gracefully
- [ ] Ensure all examples use correct import pattern: `import genai_otel; genai_otel.instrument()`
- [ ] Confirm spans contain proper attributes (gen_ai.system, model, tokens)
- [ ] Validate metrics are recorded (requests, tokens, cost, latency)

### Integration Testing
- [ ] Test examples with Console exporter (default)
- [ ] Test examples with OTLP HTTP exporter
- [ ] Test examples with OTLP gRPC exporter (if configured)
- [ ] Verify GPU metrics collection (if GPU available)
- [ ] Test cost tracking accuracy across providers
- [ ] Validate MCP instrumentation (databases, vector DBs, Redis, Kafka)
- [ ] Verify HTTP instrumentation is disabled by default
- [ ] Test with `enable_http_instrumentation=True` flag

## Post-Release
- [ ] Monitor PyPI download stats
- [ ] Watch for bug reports
- [ ] Respond to GitHub issues
- [ ] Update documentation if needed
- [ ] Plan next release
