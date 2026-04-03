# File Manager Module

The **file-manager** module provides a set of utilities for interacting with files in a project repository. It is designed to be lightweight, easy to use, and fully integrated with the project's file system.

## Overview

The module exposes the following functions:

| Function | Description |
|----------|-------------|
| `fm_read_file(path)` | Read the contents of a file relative to the project root. |
| `fm_list_files(path='.', pattern='**/*')` | List files in the project matching a glob pattern. |
| `fm_search_in_files(query, path='.', pattern='**/*.py', is_regex=False)` | Search for a string or regex in files and return matches with context. |
| `fm_write_file(path, content)` | Create or overwrite a file; returns a diff if the file already existed. |
| `fm_patch_file(path, old_string, new_string)` | Replace the first occurrence of `old_string` with `new_string` in a file. |
| `fm_check_invariants(path='.')` | Validate files against rules defined in `rules.json`. |

These utilities are used throughout the project, for example in the `api_server.py` to handle file operations via HTTP endpoints.

## Installation

The module is part of the repository, so no external installation is required. Ensure that the environment variable `FILE_MANAGER_ROOT` points to the project root.

## Usage

Below are examples of how to use each function.

### Read a File
```python
from file_manager import fm_read_file

result = fm_read_file('path/to/file.txt')
print(result['content'])
```

### List Files
```python
from file_manager import fm_list_files

files = fm_list_files(pattern='**/*.py')
for f in files:
    print(f"{f['path']} ({f['size']} bytes)")
```

### Search in Files
```python
from file_manager import fm_search_in_files

matches = fm_search_in_files(query='TODO', pattern='**/*.py')
for m in matches:
    print(f"{m['file']}:{m['line_number']}: {m['line_content']}")
```

### Write a File
```python
from file_manager import fm_write_file

fm_write_file('new_file.txt', 'Hello, world!')
```

### Patch a File
```python
from file_manager import fm_patch_file

fm_patch_file('config.yaml', 'old_value', 'new_value')
```

### Check Invariants
```python
from file_manager import fm_check_invariants

violations = fm_check_invariants()
for v in violations:
    print(f"Rule {v['rule_id']} violated in {v['file']} at line {v['line_number']}")
```

## Example: Using fm_search_in_files in API
In `api_server.py`, the following snippet demonstrates how `fm_search_in_files` is used to find all usages of `FastMCP`:

```python
# Example: find all usages of FastMCP
fm_search_in_files(query='FastMCP', pattern='**/*.py')
```

## Contributing

Feel free to submit pull requests to improve the module. Ensure that any new code is covered by tests and adheres to the project's coding standards.

## License

This project is licensed under the MIT License.
