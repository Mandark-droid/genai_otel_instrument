"""Script to automatically fix all instrumentors to use conditional imports."""

import os
import re
from pathlib import Path

# Mapping of instrumentor files to their import statements and library names
INSTRUMENTOR_FIXES = {
    "google_ai_instrumentor.py": {
        "imports": ["import google.generativeai as genai"],
        "library": "google.generativeai",
        "check_name": "_google_available",
    },
    "aws_bedrock_instrumentor.py": {
        "imports": ["import boto3"],
        "library": "boto3",
        "check_name": "_boto3_available",
    },
    "cohere_instrumentor.py": {
        "imports": ["import cohere"],
        "library": "cohere",
        "check_name": "_cohere_available",
    },
    "ollama_instrumentor.py": {
        "imports": ["import ollama"],
        "library": "ollama",
        "check_name": "_ollama_available",
    },
    "langchain_instrumentor.py": {
        "imports": [
            "from langchain.chains.base import Chain",
            "from langchain.agents.agent import AgentExecutor",
        ],
        "library": "langchain",
        "check_name": "_langchain_available",
    },
    "huggingface_instrumentor.py": {
        "imports": ["import transformers"],
        "library": "transformers",
        "check_name": "_transformers_available",
    },
}


def remove_top_level_imports(content, imports_to_remove):
    """Remove specified import statements from module level."""
    for imp in imports_to_remove:
        # Remove the import line and any "# Moved to top" comment
        pattern = rf"^{re.escape(imp)}.*?\n"
        content = re.sub(pattern, "", content, flags=re.MULTILINE)
    return content


def add_availability_check(class_name, library, check_name):
    """Generate the availability check code."""
    return f'''
    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self.{check_name} = False
        self._check_availability()

    def _check_availability(self):
        """Check if {library} library is available."""
        try:
            import {library}
            self.{check_name} = True
            logger.debug("{library} library detected and available for instrumentation")
        except ImportError:
            logger.debug("{library} library not installed, instrumentation will be skipped")
            self.{check_name} = False
'''


def add_early_return_to_instrument(content, check_name):
    """Add early return to instrument method."""
    # Find the instrument method and add the check
    pattern = r"(def instrument\(self, config: OTelConfig\):.*?\n)"
    replacement = rf'\1        """Instrument {check_name.replace("_", " ")} if available."""\n        if not self.{check_name}:\n            logger.debug("Skipping instrumentation - library not available")\n            return\n\n'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL, count=1)
    return content


def fix_instrumentor_file(filepath, imports_to_remove, library, check_name):
    """Fix a single instrumentor file."""
    print(f"Fixing {filepath.name}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if already fixed
    if check_name in content:
        print(f"  ✓ Already fixed")
        return

    # Remove top-level imports
    content = remove_top_level_imports(content, imports_to_remove)

    # Find the class definition
    class_match = re.search(r"class (\w+Instrumentor)\(BaseInstrumentor\):", content)
    if not class_match:
        print(f"  ✗ Could not find class definition")
        return

    class_name = class_match.group(1)

    # Add __init__ and _check_availability after class definition
    class_def_line = class_match.group(0)
    init_code = add_availability_check(class_name, library, check_name)
    content = content.replace(
        f'{class_def_line}\n    """',
        f'{class_def_line}\n    """',
    )

    # Insert after the docstring
    docstring_end = content.find('"""', content.find(class_def_line) + len(class_def_line) + 10)
    if docstring_end != -1:
        insert_pos = content.find("\n", docstring_end) + 1
        content = content[:insert_pos] + init_code + content[insert_pos:]

    # Add early return to instrument method
    content = add_early_return_to_instrument(content, check_name)

    # Move imports inside methods
    for imp in imports_to_remove:
        # Add import statement at the beginning of instrument method after early return
        if "def instrument(self, config: OTelConfig):" in content:
            pattern = r"(def instrument\(self, config: OTelConfig\):.*?return\n\n)"
            replacement = rf"\1        try:\n            {imp}\n"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL, count=1)

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✓ Fixed successfully")


def main():
    """Main function to fix all instrumentors."""
    # Get the project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    instrumentors_dir = project_root / "genai_otel" / "instrumentors"

    if not instrumentors_dir.exists():
        print(f"Error: Instrumentors directory not found at {instrumentors_dir}")
        return

    print("=" * 60)
    print("Fixing Instrumentors - Adding Conditional Imports")
    print("=" * 60)
    print()

    for filename, config in INSTRUMENTOR_FIXES.items():
        filepath = instrumentors_dir / filename
        if filepath.exists():
            fix_instrumentor_file(
                filepath, config["imports"], config["library"], config["check_name"]
            )
        else:
            print(f"Warning: {filename} not found")

    print()
    print("=" * 60)
    print("✓ All instrumentors have been fixed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review the changes in genai_otel/instrumentors/")
    print("2. Run tests: pytest tests/ -v")
    print("3. Commit the changes")


if __name__ == "__main__":
    main()
