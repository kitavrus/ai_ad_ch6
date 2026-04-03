# CHANGELOG

## [Unreleased]
- Added comprehensive file management utilities including read, list, search, write, patch, and invariant checks.
- Implemented strict generation workflow: SEARCH → LIST → SEARCH → WRITE.
- Updated API server to expose POST /pr-review endpoint for automated PR review.
- Added README with usage examples and contribution guidelines.
- Fixed minor bugs in file search and write operations.

## [1.0.0] - 2026-04-03
- Initial release of the file-manager module.
- Included core functions: `fm_read_file`, `fm_list_files`, `fm_search_in_files`, `fm_write_file`, `fm_patch_file`, `fm_check_invariants`.
- Provided example usage in `api_server.py`.
- Added `CHANGELOG.md` and `README.md`.
