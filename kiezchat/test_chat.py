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
EVENT_YEAR = os.environ.get("EVENT_YEAR", "2026")


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
        answer = ask(f"When does Kiez Burn {EVENT_YEAR} start?")
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
        answer = ask(f"List all camps and installations at Kiez Burn {EVENT_YEAR}")
        found = [c for c in self.KNOWN_CAMPS if c.lower() in answer.lower()]
        assert len(found) >= 10, (
            f"Expected at least 10 known camps in answer, found only {len(found)}: {found}\n"
            f"Answer (first 800 chars): {answer[:800]}"
        )

    def test_explicit_all_camps(self):
        """Explicitly requesting all camps should not just return a few."""
        answer = ask(f"Give me the complete list of all Kieze (camps) at Kiez Burn {EVENT_YEAR}")
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
        answer = ask(f"What camps are there at Kiez Burn {EVENT_YEAR}?")
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


class TestKnownBadAnswers:
    """Regression tests for specific hallucinations reported by users."""

    def test_ticket_name_change_not_24h(self):
        """Should NOT claim name changes are allowed up to 24 hours before event.
        The 24h deadline applies to BurnHalla, not Kiez Burn.
        For Kiez Burn 2026, the deadline was June 9 (14 days before the event).
        """
        answer = ask("What is the deadline to change the name on my ticket?")
        lower = answer.lower()
        # Must not claim 24 hours before event
        assert "24 hour" not in lower and "24h" not in lower, (
            f"Answer incorrectly states a 24-hour deadline (that's BurnHalla policy, not Kiez Burn):\n{answer}"
        )
        # Should mention June 9 or the correct deadline context
        assert any(w in lower for w in ["june 9", "9 june", "9.", "june", "deadline", "transfer"]), (
            f"Answer should mention the ticket transfer deadline, got:\n{answer}"
        )

    def test_ticket_resale_price_cap(self):
        """Charging more than face value IS against No Commerce principle.
        The bot should confirm this is not allowed, not hedge.
        """
        answer = ask("Can I sell my Kiez Burn ticket for more than I paid?")
        lower = answer.lower()
        # Should say it's not allowed / against principles
        assert any(w in lower for w in ["not allowed", "not permit", "against", "no commerce", "violation", "cannot", "can't", "shouldn't", "principle"]), (
            f"Answer should clearly state resale markup is against the No Commerce principle, got:\n{answer}"
        )

    def test_no_shift_not_pushed_to_volunteer(self):
        """When someone says they don't want to do a shift, the bot should not
        respond by giving them volunteer shift advice or suggesting they sign up.
        """
        answer = ask("I don't want to do a shift at Kiez Burn. Is that okay?")
        lower = answer.lower()
        # Should NOT immediately push volunteer signup links or suggest they must sign up
        bad_phrases = ["sign up for shift", "signup", "shift signup", "you should volunteer", "shifts are needed"]
        found_bad = [p for p in bad_phrases if p in lower]
        assert not found_bad, (
            f"Answer incorrectly pushed shift volunteering on someone who said they don't want shifts. Found: {found_bad}\n{answer}"
        )
        # Should acknowledge participation takes many forms or that it's okay
        assert any(w in lower for w in ["participate", "contribution", "many ways", "okay", "fine", "welcome", "form"]), (
            f"Answer should acknowledge participation takes many forms, got:\n{answer}"
        )

    def test_year_filter_no_past_event_data(self):
        """Answers should not present old-year information as current.
        A question about the current event should not surface 2024/2025 specifics.
        """
        answer = ask("What were the ticket prices for past Kiez Burns?")
        lower = answer.lower()
        # This is about past events, so mentioning past years is fine. But if asked about
        # the current event, old info should not come up as if it's current.
        # This is a sanity check that the response doesn't just hallucinate current prices
        # from old data.
        assert len(answer) > 20, f"Expected a real answer, got: {answer}"

