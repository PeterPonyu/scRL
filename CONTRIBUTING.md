# Contributing to scRL

Contributions to scRL are welcome. This document outlines the process for contributing to this project.

## How to Contribute

### Reporting Issues

If you encounter any bugs or have feature requests, please open an issue on [GitHub Issues](https://github.com/PeterPonyu/scRL/issues). When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Your environment information (Python version, OS, package versions)
- Error messages or traceback if applicable
- A minimal code example to reproduce the issue

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`.
2. **Install development dependencies**:
   ```bash
   git clone https://github.com/your-username/scRL.git
   cd scRL
   pip install -e .
   ```
3. **Make your changes** and ensure they follow the existing code style.
4. **Test your changes** thoroughly.
5. **Update documentation** if needed (docstrings, notebooks, README).
6. **Submit a pull request** with a clear description of your changes.

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions following NumPy style
- Keep functions focused and modular

## API Naming Conventions

When adding new functions, please follow our naming conventions:

- **`d_*`** prefix: Functions for **discrete** features (categorical data like cluster labels, lineage identities). Example: `d_rewards`
- **`c_*`** prefix: Functions for **continuous** features (numerical data like gene expression values). Example: `c_rewards`
- Use clear, descriptive names that indicate the function's purpose

### Backward Compatibility

When renaming functions, please:
1. Add an alias in `__init__.py` for the old name
2. Update the `__all__` list to include both names
3. Document the change in release notes

## Documentation

- All public functions should have docstrings
- Update the Sphinx documentation if adding new features
- Keep notebook tutorials up-to-date with API changes

## Questions?

Feel free to reach out via [GitHub Issues](https://github.com/PeterPonyu/scRL/issues) or contact [fuzeyu99@126.com](mailto:fuzeyu99@126.com).

Thank you for contributing!
