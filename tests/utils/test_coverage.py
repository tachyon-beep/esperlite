"""
Test coverage utilities and quality gates for Esper platform.

This module provides tools for measuring, reporting, and enforcing
test coverage standards across the codebase.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pytest


class CoverageAnalyzer:
    """Analyze and report test coverage metrics."""

    def __init__(self, source_dir: str = "src/esper", min_coverage: float = 90.0):
        self.source_dir = Path(source_dir)
        self.min_coverage = min_coverage
        self.coverage_data = {}

    def run_coverage_analysis(
        self, test_paths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Run comprehensive coverage analysis.

        Args:
            test_paths: Specific test paths to run, or None for all tests

        Returns:
            Dictionary mapping module names to coverage percentages
        """
        if test_paths is None:
            test_paths = ["tests/"]

        # Run tests with coverage
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=" + str(self.source_dir),
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-report=html:htmlcov",
            "--cov-branch",  # Include branch coverage
        ] + test_paths

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

            if result.returncode != 0:
                print(f"Tests failed with return code {result.returncode}")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return {}

            return self._parse_coverage_results()

        except Exception as e:
            print(f"Error running coverage analysis: {e}")
            return {}

    def _parse_coverage_results(self) -> Dict[str, float]:
        """Parse coverage results from generated files."""
        import json

        coverage_file = Path("coverage.json")
        if not coverage_file.exists():
            return {}

        try:
            with open(coverage_file) as f:
                data = json.load(f)

            coverage_by_file = {}
            files = data.get("files", {})

            for file_path, file_data in files.items():
                # Convert absolute path to relative module name
                if self.source_dir.name in file_path:
                    rel_path = Path(file_path).relative_to(Path.cwd())
                    module_name = str(rel_path).replace("/", ".").replace(".py", "")

                    summary = file_data.get("summary", {})
                    covered_lines = summary.get("covered_lines", 0)
                    num_statements = summary.get("num_statements", 1)

                    coverage_percent = (
                        (covered_lines / num_statements) * 100
                        if num_statements > 0
                        else 0
                    )
                    coverage_by_file[module_name] = coverage_percent

            self.coverage_data = coverage_by_file
            return coverage_by_file

        except Exception as e:
            print(f"Error parsing coverage results: {e}")
            return {}

    def generate_coverage_report(self) -> str:
        """Generate a detailed coverage report."""
        if not self.coverage_data:
            return "No coverage data available"

        report_lines = ["=" * 80, "ESPER PLATFORM TEST COVERAGE REPORT", "=" * 80, ""]

        # Overall statistics
        all_coverages = list(self.coverage_data.values())
        overall_coverage = (
            sum(all_coverages) / len(all_coverages) if all_coverages else 0
        )

        report_lines.extend(
            [
                f"Overall Coverage: {overall_coverage:.1f}%",
                f"Minimum Required: {self.min_coverage:.1f}%",
                f"Status: {'✅ PASS' if overall_coverage >= self.min_coverage else '❌ FAIL'}",
                "",
                "Coverage by Module:",
                "-" * 40,
            ]
        )

        # Sort modules by coverage (lowest first)
        sorted_modules = sorted(self.coverage_data.items(), key=lambda x: x[1])

        for module_name, coverage in sorted_modules:
            status = "✅" if coverage >= self.min_coverage else "❌"
            report_lines.append(f"{status} {module_name:<50} {coverage:>6.1f}%")

        # Identify problem areas
        low_coverage_modules = [
            (name, cov)
            for name, cov in self.coverage_data.items()
            if cov < self.min_coverage
        ]

        if low_coverage_modules:
            report_lines.extend(["", "Modules Below Minimum Coverage:", "-" * 40])

            for module_name, coverage in sorted(
                low_coverage_modules, key=lambda x: x[1]
            ):
                deficit = self.min_coverage - coverage
                report_lines.append(
                    f"❌ {module_name:<50} {coverage:>6.1f}% (deficit: {deficit:.1f}%)"
                )

        # Coverage goals and recommendations
        report_lines.extend(
            [
                "",
                "Coverage Goals:",
                "-" * 20,
                "• Core execution modules: >95% (critical for reliability)",
                "• Service modules: >90% (important for integration)",
                "• Utility modules: >85% (good for maintainability)",
                "• Contract modules: >80% (basic validation)",
                "",
            ]
        )

        return "\n".join(report_lines)

    def check_coverage_gates(self) -> Tuple[bool, List[str]]:
        """
        Check if coverage meets quality gates.

        Returns:
            Tuple of (passed, list of failures)
        """
        if not self.coverage_data:
            return False, ["No coverage data available"]

        failures = []

        # Overall coverage gate
        all_coverages = list(self.coverage_data.values())
        overall_coverage = (
            sum(all_coverages) / len(all_coverages) if all_coverages else 0
        )

        if overall_coverage < self.min_coverage:
            failures.append(
                f"Overall coverage {overall_coverage:.1f}% below minimum {self.min_coverage:.1f}%"
            )

        # Module-specific gates
        critical_modules = [
            "src.esper.execution.kasmina_layer",
            "src.esper.core.model_wrapper",
            "src.esper.execution.state_layout",
        ]

        for module in critical_modules:
            coverage = self.coverage_data.get(module, 0)
            if coverage < 95.0:  # Higher standard for critical modules
                failures.append(
                    f"Critical module {module} coverage {coverage:.1f}% below 95% requirement"
                )

        # Check for modules with zero coverage
        zero_coverage_modules = [
            name for name, cov in self.coverage_data.items() if cov == 0
        ]

        if zero_coverage_modules:
            failures.append(
                f"Modules with zero coverage: {', '.join(zero_coverage_modules)}"
            )

        return len(failures) == 0, failures


class TestQualityGates:
    """Quality gates for test infrastructure."""

    @staticmethod
    def check_test_file_structure() -> Tuple[bool, List[str]]:
        """Check that test file structure follows conventions."""
        issues = []

        test_dir = Path("tests")
        if not test_dir.exists():
            return False, ["tests/ directory not found"]

        # Check for required test directories
        required_dirs = ["core", "execution", "integration", "performance", "utils"]
        for req_dir in required_dirs:
            dir_path = test_dir / req_dir
            if not dir_path.exists():
                issues.append(f"Missing required test directory: tests/{req_dir}/")

        # Check that test files follow naming convention
        for test_file in test_dir.rglob("test_*.py"):
            # Verify test file has corresponding source file
            rel_path = test_file.relative_to(test_dir)
            if rel_path.name != "test_infrastructure.py":  # Skip infrastructure tests
                source_path = (
                    Path("src")
                    / "esper"
                    / rel_path.parent
                    / rel_path.name.replace("test_", "").replace(".py", ".py")
                )
                if not source_path.exists() and "test_" in rel_path.name:
                    # Check for alternative naming patterns
                    alt_source = (
                        Path("src")
                        / "esper"
                        / rel_path.parent
                        / rel_path.name.replace("test_", "")
                    )
                    if not alt_source.exists():
                        issues.append(
                            f"Test file {test_file} has no corresponding source file"
                        )

        return len(issues) == 0, issues

    @staticmethod
    def check_test_documentation() -> Tuple[bool, List[str]]:
        """Check that tests are properly documented."""
        issues = []

        test_files = list(Path("tests").rglob("test_*.py"))

        for test_file in test_files:
            try:
                with open(test_file) as f:
                    content = f.read()

                # Check for module docstring
                if not content.strip().startswith(
                    '"""'
                ) and not content.strip().startswith("'''"):
                    issues.append(f"Test file {test_file} missing module docstring")

                # Check for class docstrings in test classes
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("class Test") and ":" in line:
                        # Check if next non-empty line is a docstring
                        for j in range(i + 1, min(i + 5, len(lines))):
                            next_line = lines[j].strip()
                            if (
                                next_line
                                and not next_line.startswith('"""')
                                and not next_line.startswith("'''")
                            ):
                                issues.append(
                                    f"Test class in {test_file} line {i+1} missing docstring"
                                )
                                break
                            elif next_line.startswith('"""') or next_line.startswith(
                                "'''"
                            ):
                                break

            except Exception as e:
                issues.append(f"Error reading test file {test_file}: {e}")

        return len(issues) == 0, issues

    @staticmethod
    def check_performance_test_coverage() -> Tuple[bool, List[str]]:
        """Check that performance tests cover critical paths."""
        issues = []

        perf_test_dir = Path("tests/performance")
        if not perf_test_dir.exists():
            return False, ["Performance test directory missing"]

        # Check for required performance test files
        required_perf_tests = [
            "test_kasmina_performance.py",
        ]

        for req_test in required_perf_tests:
            test_file = perf_test_dir / req_test
            if not test_file.exists():
                issues.append(f"Missing required performance test: {req_test}")
            else:
                # Check test file contains key performance tests
                try:
                    with open(test_file) as f:
                        content = f.read()

                    required_test_methods = [
                        "test_dormant_seed_overhead",
                        "test_single_forward_pass_latency",
                    ]

                    for method in required_test_methods:
                        if method not in content:
                            issues.append(
                                f"Performance test {req_test} missing {method}"
                            )

                except Exception as e:
                    issues.append(f"Error reading performance test {req_test}: {e}")

        return len(issues) == 0, issues


@pytest.mark.slow
class TestCoverageIntegration:
    """Integration tests for coverage analysis system."""

    def test_coverage_analyzer_initialization(self):
        """Test coverage analyzer initialization."""
        analyzer = CoverageAnalyzer(source_dir="src/esper", min_coverage=85.0)

        assert analyzer.source_dir == Path("src/esper")
        assert analyzer.min_coverage == 85.0
        assert analyzer.coverage_data == {}

    def test_coverage_report_generation(self):
        """Test coverage report generation with mock data."""
        analyzer = CoverageAnalyzer(min_coverage=90.0)

        # Mock some coverage data
        analyzer.coverage_data = {
            "src.esper.core.model_wrapper": 95.5,
            "src.esper.execution.kasmina_layer": 88.2,
            "src.esper.utils.config": 92.1,
            "src.esper.services.urza.main": 45.0,  # Below threshold
        }

        report = analyzer.generate_coverage_report()

        # Verify report contains expected sections
        assert "ESPER PLATFORM TEST COVERAGE REPORT" in report
        assert "Overall Coverage:" in report
        assert "Coverage by Module:" in report
        assert "Modules Below Minimum Coverage:" in report

        # Verify specific data appears
        assert "95.5%" in report  # High coverage module
        assert "45.0%" in report  # Low coverage module

        print("Sample coverage report:")
        print(report)

    def test_quality_gates_check(self):
        """Test quality gates checking."""
        analyzer = CoverageAnalyzer(min_coverage=90.0)

        # Test with good coverage
        analyzer.coverage_data = {
            "src.esper.core.model_wrapper": 95.5,
            "src.esper.execution.kasmina_layer": 96.2,
            "src.esper.execution.state_layout": 97.1,
        }

        passed, failures = analyzer.check_coverage_gates()
        assert passed
        assert len(failures) == 0

        # Test with poor coverage
        analyzer.coverage_data = {
            "src.esper.core.model_wrapper": 85.0,  # Below critical threshold
            "src.esper.execution.kasmina_layer": 80.0,  # Below critical threshold
            "src.esper.services.new_service": 0.0,  # Zero coverage
        }

        passed, failures = analyzer.check_coverage_gates()
        assert not passed
        assert len(failures) > 0

        # Verify specific failure types
        failure_text = " ".join(failures)
        assert "Critical module" in failure_text
        assert "zero coverage" in failure_text

    def test_test_file_structure_check(self):
        """Test test file structure validation."""
        passed, issues = TestQualityGates.check_test_file_structure()

        # Should pass since we have proper structure
        if not passed:
            print("Test structure issues:", issues)

        # At minimum, should not have critical missing directories
        critical_missing = [
            issue for issue in issues if "Missing required test directory" in issue
        ]
        assert (
            len(critical_missing) == 0
        ), f"Critical test directories missing: {critical_missing}"


def run_comprehensive_coverage_check(min_coverage: float = 90.0) -> bool:
    """
    Run comprehensive coverage check with quality gates.

    Args:
        min_coverage: Minimum required coverage percentage

    Returns:
        True if all quality gates pass
    """
    print("Running comprehensive coverage analysis...")

    # Initialize analyzer
    analyzer = CoverageAnalyzer(min_coverage=min_coverage)

    # Run coverage analysis
    coverage_data = analyzer.run_coverage_analysis()

    if not coverage_data:
        print("❌ Failed to generate coverage data")
        return False

    # Generate report
    report = analyzer.generate_coverage_report()
    print(report)

    # Check quality gates
    coverage_passed, coverage_failures = analyzer.check_coverage_gates()
    structure_passed, structure_issues = TestQualityGates.check_test_file_structure()
    doc_passed, doc_issues = TestQualityGates.check_test_documentation()
    perf_passed, perf_issues = TestQualityGates.check_performance_test_coverage()

    # Report results
    all_passed = coverage_passed and structure_passed and doc_passed and perf_passed

    print("\n" + "=" * 80)
    print("QUALITY GATES SUMMARY")
    print("=" * 80)
    print(f"Coverage Gates:    {'✅ PASS' if coverage_passed else '❌ FAIL'}")
    print(f"Structure Check:   {'✅ PASS' if structure_passed else '❌ FAIL'}")
    print(f"Documentation:     {'✅ PASS' if doc_passed else '❌ FAIL'}")
    print(f"Performance Tests: {'✅ PASS' if perf_passed else '❌ FAIL'}")
    print(f"Overall Result:    {'✅ PASS' if all_passed else '❌ FAIL'}")

    # Report failures
    if not all_passed:
        print("\nFailures:")
        for failure in coverage_failures + structure_issues + doc_issues + perf_issues:
            print(f"❌ {failure}")

    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_coverage_check()
    sys.exit(0 if success else 1)
