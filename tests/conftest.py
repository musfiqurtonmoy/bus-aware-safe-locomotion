"""Release gate (paper/release_checklist.md): tests marked `release` run only with --release.
They check the compiled PDFs for placeholders, which are pending by design until submission, so
they would fail every routine run; `python -m pytest -q tests --release` is the submission check."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--release", action="store_true", default=False,
                     help="also run the release checks on the compiled PDFs")


def pytest_configure(config):
    config.addinivalue_line("markers", "release: release gate on the compiled PDFs (needs --release)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--release"):
        return
    skip = pytest.mark.skip(reason="release gate: run with --release before submission")
    for item in items:
        if "release" in item.keywords:
            item.add_marker(skip)
