# AGENTS.md - Agent Coding Guidelines

This document provides guidelines for agentic coding agents working in this repository.

## Project Overview

Nebula Graph LangChain Integration - A Python project for integrating LangChain with Nebula Graph database.

## Build/Lint/Test Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Examples
```bash
python examples/basic_usage.py
python examples/langchain_integration.py
python examples/setup_nebula.py
python examples/setup_nebula.py --action reset
```

### Running Tests
No formal test suite exists yet. To run a single test manually:
```bash
python -m pytest tests/test_file.py::test_function -v
```
Or if using unittest:
```bash
python -m unittest tests.test_module.TestClass.test_method -v
```

### Linting
Install and run ruff (recommended):
```bash
pip install ruff
ruff check src/
ruff check src/ --fix  # Auto-fix issues
```

Or use pylint:
```bash
pip install pylint
pylint src/
```

### Type Checking
```bash
pip install mypy
mypy src/ --ignore-missing-imports
```

### Formatting
```bash
pip install black
black src/
```

## Code Style Guidelines

### General Principles
- Follow PEP 8 style guide for Python
- Use 4 spaces for indentation (not tabs)
- Maximum line length: 120 characters
- Use explicit imports (no wildcard imports)

### Imports
```python
# Standard library first, then third-party, then local
from typing import Any, Dict, List, Optional, Tuple
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from langchain.chains.base import Chain
from langchain.pydantic_v1 import BaseModel, Field
from .nebula_graph_store import NebulaGraphStore  # Relative imports for internal modules
```

### Naming Conventions
- **Classes**: `CamelCase` (e.g., `NebulaGraphStore`)
- **Functions/Methods**: `snake_case` (e.g., `execute_query`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **Private methods**: Prefix with underscore (e.g., `_initialize_connection`)
- **Instance variables**: `snake_case` with optional underscore prefix for private

### Type Hints
- Use type hints for all function arguments and return values
- Use `Optional[X]` instead of `X | None` for Python 3.9 compatibility
- Use `Union[X, Y]` for multiple types

```python
def execute_query(self, query: str) -> List[Dict[str, Any]]:
    ...
```

### Docstrings
Use Google-style docstrings (Chinese comments as per project convention):

```python
def method_name(self, param1: str, param2: Optional[int] = None) -> bool:
    """
    方法的简要描述

    Args:
        param1: 参数1的描述
        param2: 参数2的描述，可选

    Returns:
        返回值的描述

    Raises:
        Exception: 异常情况的描述
    """
```

### Error Handling
- Use try-except blocks for operations that may fail
- Raise specific exceptions with meaningful messages
- Log errors appropriately (use `print` for simple cases, logging module for production)

```python
try:
    result = self.session.execute(query)
    if not result.is_succeeded():
        raise Exception(f"查询执行失败: {result.error_msg()}")
except Exception as e:
    raise ConnectionError(f"连接Nebula Graph失败: {str(e)}")
```

### Pydantic Models
- Use `pydantic_v1` for LangChain compatibility
- Use `Field` for field definitions with descriptions
- Set `arbitrary_types_allowed = True` in Config class when needed

```python
class NebulaConfig(BaseModel):
    addresses: List[str] = Field(default=["127.0.0.1:9669"], description="地址列表")
    user: str = Field(default="root", description="用户名")

    class Config:
        arbitrary_types_allowed = True
```

### LangChain Integration
- Implement proper Chain interface with `input_keys` and `output_keys` properties
- Use `_call` method for main chain logic
- Support callback managers for verbose output
- Use `BaseModel` from `langchain.pydantic_v1` for LangChain-compatible models

### Class Structure
```python
class MyClass:
    """类的简要描述

    类的详细描述（可选）
    """

    def __init__(self, param: str):
        """初始化方法"""
        self.param = param

    def public_method(self) -> None:
        """公开方法"""
        pass

    def _private_method(self) -> None:
        """私有方法"""
        pass
```

### File Organization
- One public class per module when possible
- Keep related functionality together
- Use `__all__` to explicitly define public API
- Use relative imports for internal module imports

### Database Connection
- Always close connections properly (use `__del__` or context managers)
- Handle connection errors gracefully
- Use connection pools when available

### Best Practices
1. Keep functions small and focused (single responsibility)
2. Avoid deep nesting (max 3-4 levels)
3. Use early returns when possible
4. Prefer explicit over implicit
5. Write meaningful variable and function names
6. Add comments only when necessary (code should be self-documenting)
7. Test edge cases and error conditions

### Git Conventions (if applicable)
- Write concise commit messages
- Use present tense: "Add feature" not "Added feature"
- Reference issues in commit messages when applicable

## Project Structure
```
nebula/
├── src/                      # Source code
│   ├── __init__.py
│   ├── nebula_graph_store.py # Graph database operations
│   └── nebula_chain.py       # LangChain integration
├── examples/                 # Example scripts
├── requirements.txt          # Dependencies
└── README.md                # Documentation
```

## Environment Setup
Create a `.env` file (see `.env.example`):
```
NEBULA_ADDRESSES=127.0.0.1:9669
NEBULA_USER=root
NEBULA_PASSWORD=nebula
NEBULA_SPACE=test
OPENAI_API_KEY=your_key
```
