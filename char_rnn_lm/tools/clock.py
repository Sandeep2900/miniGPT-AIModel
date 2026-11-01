"""
tools/clock.py
System clock provider for real-time datetime awareness.
Provides the exact current date and time without network calls.
"""

from datetime import datetime


def get_current_time() -> str:
    """
    Returns current system date and time in format: YYYY-MM-DD HH:MM
    Example: '2026-11-01 20:30'
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    print(f"Current time: {get_current_time()}")
