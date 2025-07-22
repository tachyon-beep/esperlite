# Contributing to Esper

Thank you for your interest in contributing to the Esper Morphogenetic Training Platform!

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Check if the issue already exists
- Include steps to reproduce
- Include system information
- Provide error messages and logs

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Check code quality (`black src tests && ruff check src tests && pytype`)
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/esperlite.git
cd esperlite

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest
```

### Code Style

- Follow PEP 8
- Use Black for formatting
- Use type hints
- Write docstrings for public APIs
- Keep line length under 100 characters

### Testing

- Write tests for new features
- Maintain >90% test coverage
- Run full test suite before submitting
- Include both unit and integration tests

### Documentation

- Update docstrings
- Update README if needed
- Add examples for new features
- Document breaking changes

## Development Workflow

1. Check out a new branch from `main`
2. Make your changes
3. Add tests
4. Update documentation
5. Run quality checks
6. Submit PR

## Questions?

Feel free to open an issue for discussion or reach out to the maintainers.
