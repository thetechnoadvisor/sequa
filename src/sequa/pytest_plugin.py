"""Pytest plugin for Sequa LLM cassette recording and replaying."""

from __future__ import annotations

import os
from typing import Any, Generator
import pytest

from sequa.cassette import cassette


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add Sequa command-line options to pytest."""
    group = parser.getgroup("sequa", "Sequa LLM Cassette Testing")
    group.addoption(
        "--sequa-mode",
        action="store",
        default=None,
        choices=["auto", "record", "replay", "live"],
        help="Global override for Sequa execution mode (auto, record, replay, live).",
    )
    group.addoption(
        "--sequa-path",
        action="store",
        default="tests/cassettes",
        help="Base directory path for saving/loading Sequa cassette recordings (default: tests/cassettes).",
    )
    group.addoption(
        "--sequa-mask-pii",
        action="store_true",
        default=False,
        help="Enable automatic PII masking for all recorded cassettes.",
    )
    group.addoption(
        "--disable-sequa",
        action="store_true",
        default=False,
        help="Disable Sequa LLM interception and recording during test execution.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register Sequa custom markers in pytest."""
    config.addinivalue_line(
        "markers",
        "sequa(**kwargs): Wrap test function in a Sequa cassette context manager. "
        "Supports path, mode, ignore_fields, normalizer, adapter, mask_pii, guardrails, storage.",
    )
    config.addinivalue_line(
        "markers",
        "cassette(**kwargs): Alias for pytest.mark.sequa.",
    )


def _get_marker_kwargs(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Extract marker options from @pytest.mark.sequa or @pytest.mark.cassette."""
    marker = request.node.get_closest_marker("sequa") or request.node.get_closest_marker("cassette")
    if marker:
        return dict(marker.kwargs)
    return {}


def _resolve_cassette_kwargs(
    request: pytest.FixtureRequest,
    marker_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Resolve full cassette initialization parameters based on CLI options, markers, and test name."""
    config = request.config

    if config.getoption("--disable-sequa", default=False):
        return {}

    cli_mode = config.getoption("--sequa-mode", default=None)
    mode = cli_mode or marker_kwargs.get("mode", "auto")

    if "path" in marker_kwargs:
        cassette_path = marker_kwargs["path"]
    else:
        base_path = config.getoption("--sequa-path", default="tests/cassettes")
        module_name = request.module.__name__ if request.module else "test"
        module_stem = module_name.split(".")[-1]
        test_name = request.node.name
        test_stem = test_name.replace("[", "_").replace("]", "").replace("/", "_")
        cassette_path = os.path.join(base_path, module_stem, f"{test_stem}.json")

    cli_mask_pii = config.getoption("--sequa-mask-pii", default=False)
    mask_pii = cli_mask_pii or marker_kwargs.get("mask_pii", False)

    kwargs: dict[str, Any] = {
        "path": cassette_path,
        "mode": mode,
        "mask_pii": mask_pii,
    }

    for key in ("ignore_fields", "normalizer", "adapter", "guardrails", "storage"):
        if key in marker_kwargs:
            kwargs[key] = marker_kwargs[key]

    return kwargs


@pytest.fixture
def sequa_cassette(request: pytest.FixtureRequest) -> Generator[cassette | None, None, None]:
    """Pytest fixture providing active Sequa cassette context manager.

    Usage
    -----
    >>> def test_llm_call(sequa_cassette):
    ...     res = model.invoke("hello")
    """
    if request.config.getoption("--disable-sequa", default=False):
        yield None
        return

    marker_kwargs = _get_marker_kwargs(request)
    cassette_kwargs = _resolve_cassette_kwargs(request, marker_kwargs)

    with cassette(**cassette_kwargs) as cas:
        yield cas


@pytest.fixture
def cassette_fixture(sequa_cassette: cassette | None) -> cassette | None:
    """Alias fixture for `sequa_cassette`."""
    return sequa_cassette


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    """Autouse wrapper for tests annotated with @pytest.mark.sequa or @pytest.mark.cassette.

    If a test function or test class has @pytest.mark.sequa / @pytest.mark.cassette
    and does not explicitly request the `sequa_cassette` fixture, this hook automatically
    wraps the test call within a cassette context.
    """
    marker = item.get_closest_marker("sequa") or item.get_closest_marker("cassette")
    disabled = item.config.getoption("--disable-sequa", default=False)

    fixture_requested = (
        "sequa_cassette" in getattr(item, "fixturenames", [])
        or "cassette_fixture" in getattr(item, "fixturenames", [])
    )

    if marker and not disabled and not fixture_requested:
        marker_kwargs = dict(marker.kwargs)
        cli_mode = item.config.getoption("--sequa-mode", default=None)
        mode = cli_mode or marker_kwargs.get("mode", "auto")

        if "path" in marker_kwargs:
            cassette_path = marker_kwargs["path"]
        else:
            base_path = item.config.getoption("--sequa-path", default="tests/cassettes")
            module_name = item.module.__name__ if item.module else "test"
            module_stem = module_name.split(".")[-1]
            test_name = item.name
            test_stem = test_name.replace("[", "_").replace("]", "").replace("/", "_")
            cassette_path = os.path.join(base_path, module_stem, f"{test_stem}.json")

        cli_mask_pii = item.config.getoption("--sequa-mask-pii", default=False)
        mask_pii = cli_mask_pii or marker_kwargs.get("mask_pii", False)

        kwargs: dict[str, Any] = {
            "path": cassette_path,
            "mode": mode,
            "mask_pii": mask_pii,
        }

        for key in ("ignore_fields", "normalizer", "adapter", "guardrails", "storage"):
            if key in marker_kwargs:
                kwargs[key] = marker_kwargs[key]

        with cassette(**kwargs):
            yield
    else:
        yield
