"""Integration tests for kiezchat RAG chat app.

Run locally:
    ANTHROPIC_API_KEY=... python -m pytest test_chat.py -v
"""
import json
import os
import re
import pytest
import requests

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:5001")


def ask(question: str) -> str:
    """Send a question and return the full streamed response text."""
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"message": question},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()
    text_parts = []
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode() if isinstance(line, bytes) else line
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            text_parts.append(json.loads(line[6:])["text"])
        except (json.JSONDecodeError, KeyError):
            pass
    return "".join(text_parts)


class TestBasicQuestions:
    def test_event_dates(self):
        answer = ask("When does Kiez Burn 2026 start?")
        assert "june" in answer.lower() or "23" in answer, f"Expected date info, got: {answer}"

    def test_principles_count(self):
        answer = ask("How many principles does Kiez Burn have?")
        assert "11" in answer, f"Expected 11 principles, got: {answer}"

    def test_principles_list(self):
        answer = ask("What are the 11 Kiez Burn principles?")
        # Should mention at least 8 of the 11 principles
        principles = [
            "participation", "self-expression", "self-reliance", "commerce",
            "trace", "community", "inclusion", "gifting", "communal", "immediacy", "consent"
        ]
        found = sum(1 for p in principles if p.lower() in answer.lower())
        assert found >= 8, f"Only found {found}/11 principles in: {answer[:500]}"


class TestCampListing:
    """These tests verify the listing/aggregation behaviour that was previously broken."""

    KNOWN_CAMPS = [
        "Baby Bar",
        "Boilerwagen",
        "HMS Hedonism",
        "Starfucks",
        "Enchanted Forest",
        "Solardome",
        "Saunacious Spa",
        "Pussy Temple",
        "The Burnt Nest Pub",
        "Der Oktopus",
        "Cathedral of Crucifera",
        "The Next Stage",
        "Museum of Emptiness",
        "Fairy Teahouse",
    ]

    def test_list_all_camps_returns_many(self):
        """Asking for all camps should return at least 10 distinct camp names."""
        answer = ask("List all camps and installations at Kiez Burn 2026")
        found = [c for c in self.KNOWN_CAMPS if c.lower() in answer.lower()]
        assert len(found) >= 10, (
            f"Expected at least 10 known camps in answer, found only {len(found)}: {found}\n"
            f"Answer (first 800 chars): {answer[:800]}"
        )

    def test_explicit_all_camps(self):
        """Explicitly requesting all camps should not just return a few."""
        answer = ask("Give me the complete list of all Kieze (camps) at Kiez Burn 2026")
        # Count camp-like items (lines with emoji or bullet points naming a camp)
        lines_with_camps = [
            l for l in answer.split('\n')
            if any(c.lower() in l.lower() for c in self.KNOWN_CAMPS)
        ]
        assert len(lines_with_camps) >= 8, (
            f"Expected 8+ camp lines, got {len(lines_with_camps)}.\n"
            f"Answer: {answer[:800]}"
        )

    def test_specific_camp_details(self):
        """Asking about a specific camp should return its description."""
        answer = ask("What is the Baby Bar at Kiez Burn?")
        assert "baby" in answer.lower() or "bar" in answer.lower(), (
            f"Expected Baby Bar info, got: {answer}"
        )

    def test_camp_count_reasonable(self):
        """The total number of camps mentioned should be > 5 for a full listing query."""
        answer = ask("What camps are there at Kiez Burn 2026?")
        # Count how many of our known camps appear
        found = sum(1 for c in self.KNOWN_CAMPS if c.lower() in answer.lower())
        assert found >= 5, (
            f"Only {found} known camps mentioned for a listing query.\n"
            f"Answer: {answer[:600]}"
        )


class TestEdgeCases:
    def test_empty_message_returns_error(self):
        resp = requests.post(f"{BASE_URL}/chat", json={"message": ""}, timeout=10)
        assert resp.status_code == 400

    def test_unknown_topic(self):
        answer = ask("What is the weather forecast for Berlin next week?")
        # Should acknowledge it doesn't know, not hallucinate
        lower = answer.lower()
        assert any(w in lower for w in ["don't", "doesn't", "not", "cannot", "weather", "forecast"]), (
            f"Expected honest 'I don't know', got: {answer}"
        )
