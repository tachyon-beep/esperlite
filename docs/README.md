# Esper Morphogen Documentation

This directory contains the comprehensive documentation for the Esper Morphogen platform, built using Sphinx with automatic API documentation generation.

## Documentation Overview

The documentation includes:

- **Main Documentation**: Overview, quick start guide, and project description
- **API Reference**: Complete API documentation automatically generated from docstrings
- **Module Coverage**: All modules in `src/esper/` are documented, including:
  - Core modules (esper, configs, core, execution)
  - Services (tamiyo, tezzeret, tolaria, urza)
  - Contracts and utilities

## Building Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

This installs:
- Sphinx (documentation generator)
- sphinx-autodoc-typehints (type hints support)
- sphinx-rtd-theme (ReadTheDocs theme)
- myst-parser (Markdown support)

### Building HTML Documentation

From the project root directory:

```bash
cd docs
make html
```

The generated HTML documentation will be available in `docs/build/html/`.

Open `docs/build/html/index.html` in your browser to view the documentation.

### Other Build Options

- **Clean build**: `make clean && make html`
- **PDF documentation**: `make latexpdf` (requires LaTeX)
- **Check for broken links**: `make linkcheck`
- **View all options**: `make help`

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation index
│   └── api/                 # API documentation
│       ├── index.rst        # API reference index
│       ├── esper.rst        # Main esper package docs
│       ├── configs.rst      # Configuration models
│       ├── contracts.rst    # Data structures
│       ├── core.rst         # Core functionality
│       ├── execution.rst    # Execution engine
│       ├── services.rst     # Service implementations
│       └── utils.rst        # Utility functions
├── build/                   # Generated documentation
│   └── html/               # HTML output
├── Makefile                # Build automation
└── README.md               # This file
```

## Configuration

The documentation is configured in `docs/source/conf.py` with:

- **Autodoc**: Automatic API documentation from docstrings
- **ReadTheDocs theme**: Professional appearance
- **Napoleon**: Google/NumPy style docstring support
- **Type hints**: Automatic type annotation documentation
- **Cross-references**: Links between modules and external libraries
- **Search functionality**: Full-text search capability

## Docstring Style

The project uses **Google-style docstrings** with type hints:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is invalid.
    """
```

## Development

### Adding New Modules

When adding new modules to `src/esper/`, the documentation will automatically discover them during the build process due to the autodoc configuration.

### Updating Documentation

1. Update docstrings in the source code
2. Run `make html` to regenerate documentation
3. Check the output in `docs/build/html/`

### ReadTheDocs Integration

The documentation is configured for ReadTheDocs hosting:

- Uses `requirements.txt` or `pyproject.toml` for dependencies
- Configured with `docs/source/conf.py`
- Includes `.nojekyll` file for GitHub Pages compatibility

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed with `pip install -e ".[docs]"`
2. **Module not found**: Check that `src/esper/` is in the Python path (configured in `conf.py`)
3. **Build warnings**: Most warnings are non-critical and don't affect functionality
4. **Missing modules**: Run `make clean && make html` to rebuild from scratch

### Build Verification

To verify the documentation build:

```bash
cd docs
make html
ls -la build/html/
```

You should see:
- `index.html` (main documentation)
- `api/` directory with all module documentation
- `_static/` directory with CSS and JavaScript assets
- `search.html` with search functionality

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [ReadTheDocs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Autodoc Extension](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)