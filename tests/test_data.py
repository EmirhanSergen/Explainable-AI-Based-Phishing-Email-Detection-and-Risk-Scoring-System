"""Tests for phishing_ai.data module."""

import pytest


def test_clean_text():
    from phishing_ai.data import clean_text

    assert clean_text("  hello  ") == "hello"
    assert clean_text("") == ""
    assert clean_text(None) == ""
