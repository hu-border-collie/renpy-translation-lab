"""Shared duration formatting helpers for GUI progress labels."""
from __future__ import annotations


def format_remaining_duration_zh(seconds: int) -> str:
    if seconds <= 0:
        return "即将完成"
    if seconds < 60:
        return f"约剩 {seconds} 秒"
    minutes, secs = divmod(seconds, 60)
    if seconds < 3600:
        if secs >= 30:
            minutes += 1
            secs = 0
        if secs:
            return f"约剩 {minutes} 分 {secs} 秒"
        return f"约剩 {minutes} 分"
    hours, minutes = divmod(minutes, 60)
    if minutes >= 30:
        hours += 1
        minutes = 0
    if minutes:
        return f"约剩 {hours} 小时 {minutes} 分"
    return f"约剩 {hours} 小时"