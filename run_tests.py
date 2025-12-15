"""Test runner and coverage report."""

import sys
import pytest


def run_tests():
    """Run all tests with coverage."""
    args = [
        'tests/',
        '-v',
        '--cov=.',
        '--cov-report=html',
        '--cov-report=term-missing',
        '--cov-exclude=tests/*',
        '--cov-exclude=setup.py',
    ]
    
    return pytest.main(args)


if __name__ == '__main__':
    sys.exit(run_tests())
