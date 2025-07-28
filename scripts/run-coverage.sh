#!/bin/bash

# Run tests with coverage
echo "Running tests with coverage..."
pytest --cov=src/esper --cov-report=term-missing --cov-report=html --cov-report=xml

# Display summary
echo ""
echo "Coverage report generated:"
echo "  - Terminal output above"
echo "  - HTML report: htmlcov/index.html"
echo "  - XML report: coverage.xml"

# Open HTML report if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    open htmlcov/index.html
elif command -v xdg-open > /dev/null; then
    xdg-open htmlcov/index.html
fi