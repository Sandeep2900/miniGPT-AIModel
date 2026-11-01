"""
tests/test_session_memory.py
Comprehensive test suite for SessionMemory.
Verifies name extraction, stop word filtering, name query responses,
history buffer trimming, and session reset.
"""

import pytest
from session_memory import SessionMemory


@pytest.fixture
def memory():
    """Provides a fresh SessionMemory instance for each test."""
    return SessionMemory(max_history=6)


def test_extract_english_name(memory):
    """Verifies standard English name declaration: 'my name is sandeep'."""
    extracted = memory.extract_name("my name is sandeep")
    assert extracted == "Sandeep"
    assert memory.user_name == "Sandeep"


def test_extract_hindi_name_with_hai(memory):
    """Verifies standard Hindi name declaration: 'mera naam Rahul hai'."""
    extracted = memory.extract_name("mera naam Rahul hai")
    assert extracted == "Rahul"
    assert memory.user_name == "Rahul"


def test_extract_i_am_format(memory):
    """Verifies contraction/phrase: 'I am Priya'."""
    extracted = memory.extract_name("I am Priya")
    assert extracted == "Priya"
    assert memory.user_name == "Priya"


def test_reject_student_role(memory):
    """Verifies role descriptor 'main ek student hoon' must NOT be extracted as a name."""
    extracted = memory.extract_name("main ek student hoon")
    assert extracted is None
    assert memory.user_name is None


def test_reject_adjective_stop_word(memory):
    """Verifies adjective 'my name is good' must NOT be extracted as a name."""
    extracted = memory.extract_name("my name is good")
    assert extracted is None
    assert memory.user_name is None


def test_reject_fine_stop_word(memory):
    """Verifies 'i am fine' must NOT be extracted as a name."""
    extracted = memory.extract_name("i am fine")
    assert extracted is None
    assert memory.user_name is None


def test_empty_and_whitespace_input(memory):
    """Verifies empty strings and whitespace produce None safely."""
    assert memory.extract_name("") is None
    assert memory.extract_name("   ") is None
    assert memory.answer_name_question("") is None
    assert memory.answer_name_question("   ") is None


def test_mixed_case_handling(memory):
    """Verifies mixed case input properly normalizes and capitalizes name."""
    extracted = memory.extract_name("mErA nAaM aMiT hAi")
    assert extracted == "Amit"
    assert memory.user_name == "Amit"


def test_name_question_when_set(memory):
    """Verifies answering 'what is my name' when name is known."""
    memory.user_name = "Sandeep"
    response = memory.answer_name_question("what is my name")
    assert response == "Aapka naam Sandeep hai."


def test_name_question_when_unset(memory):
    """Verifies answering 'what is my name' when name is not yet provided."""
    response = memory.answer_name_question("what is my name")
    assert response == "Aapne abhi tak apna naam nahi bataya."


def test_hinglish_name_questions(memory):
    """Verifies Hindi/Hinglish variants: 'mera naam kya hai', 'mera naam batao'."""
    memory.user_name = "Rahul"
    assert memory.answer_name_question("mera naam kya hai") == "Aapka naam Rahul hai."
    assert memory.answer_name_question("mera naam batao") == "Aapka naam Rahul hai."
    assert memory.answer_name_question("do you know my name") == "Aapka naam Rahul hai."


def test_non_name_question_returns_none(memory):
    """Verifies normal conversation queries return None from answer_name_question."""
    assert memory.answer_name_question("hello how are you") is None
    assert memory.answer_name_question("what is the weather today") is None


def test_history_buffer_max_turns(memory):
    """Verifies history buffer retains strictly at most 6 recent turns."""
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        memory.add_turn(role, f"Message {i}")

    assert len(memory.history) == 6
    # Oldest retained message should be Message 2 (0 and 1 trimmed)
    assert memory.history[0]["text"] == "Message 2"
    assert memory.history[-1]["text"] == "Message 7"


def test_memory_reset(memory):
    """Verifies reset clears both user_name and conversation history."""
    memory.extract_name("my name is Sandeep")
    memory.add_turn("user", "Hello")
    memory.add_turn("assistant", "Hi there")

    assert memory.user_name == "Sandeep"
    assert len(memory.history) == 2

    memory.reset()
    assert memory.user_name is None
    assert memory.history == []
