# FFMeshSplat

A Python project with modern tooling

[![CI](https://github.com/erenovic/ffmeshsplat/workflows/CI/badge.svg)](https://github.com/erenovic/ffmeshsplat/actions)
[![Coverage](https://codecov.io/gh/erenovic/ffmeshsplat/branch/main/graph/badge.svg)](https://codecov.io/gh/erenovic/ffmeshsplat)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install ffmeshsplat
```

## Quick Start

### Python API

```python
from ffmeshsplat.main import hello_world

# Basic usage
result = hello_world()
print(result)  # "Hello from FFMeshSplat!"

# With a name
result = hello_world("Alice")
print(result)  # "Hello, Alice, from FFMeshSplat!"
```

### Command Line

```bash
python -m ffmeshsplat
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/erenovic/ffmeshsplat.git
cd ffmeshsplat
```

2. Set up development environment:
```bash
make all  # Install dependencies, pre-commit hooks, and run all checks
```

### Development Commands

```bash
make help          # Show all available commands
make install-dev   # Install development dependencies
make fmt           # Format code with ruff
make lint          # Lint code with ruff
make typecheck     # Type check with mypy
make test          # Run tests
make test-cov      # Run tests with coverage
make ci            # Run all CI checks locally
make clean         # Clean build artifacts
```


## Features

- ✅ Modern Python packaging with `pyproject.toml`
- ✅ Code formatting and linting with Ruff
- ✅ Testing with pytest and coverage reporting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.