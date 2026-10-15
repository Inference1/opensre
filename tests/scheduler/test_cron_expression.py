"""Cron lines are read the crontab way, although APScheduler 3.x counts weekdays from 0 = Monday."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from infrastructure.scheduling.scheduler.cron_expression import build_cron_trigger
from infrastructure.scheduling.scheduler.runner import compute_next_run
from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind

_SUNDAY_NOON = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _fire_days(cron: str) -> set[str]:
    """The weekdays ``cron`` fires on during the two weeks after a Sunday noon."""
    trigger = build_cron_trigger(cron, "UTC")
    days: set[str] = set()
    fire = trigger.get_next_fire_time(None, _SUNDAY_NOON)
    while fire is not None and fire < _SUNDAY_NOON + timedelta(days=14):
        days.add(f"{fire:%a}")
        fire = trigger.get_next_fire_time(fire, fire)
    return days


@pytest.mark.parametrize(
    ("day_of_week", "fires_on"),
    [
        ("1-5", {"Mon", "Tue", "Wed", "Thu", "Fri"}),
        ("0", {"Sun"}),
        ("7", {"Sun"}),
        ("*/2", {"Sun", "Tue", "Thu", "Sat"}),
        # A step after a name range is ignored by APScheduler, so steps become lists.
        ("1-5/2", {"Mon", "Wed", "Fri"}),
        ("mon-fri", {"Mon", "Tue", "Wed", "Thu", "Fri"}),
        ("mon-fri/2", {"Mon", "Wed", "Fri"}),
        ("MON-SUN/2", {"Mon", "Wed", "Fri", "Sun"}),
        ("tue-fri/2", {"Tue", "Thu"}),
        ("mon-fri/2,sun", {"Mon", "Wed", "Fri", "Sun"}),
    ],
)
def test_a_weekday_field_fires_on_its_crontab_days(day_of_week: str, fires_on: set[str]) -> None:
    assert _fire_days(f"0 10 * * {day_of_week}") == fires_on


def test_a_weekday_number_above_7_is_rejected_before_it_is_stored() -> None:
    with pytest.raises(ValueError, match="0-7"):
        build_cron_trigger("0 10 * * 8", "UTC")


def test_the_stored_next_run_skips_the_weekend() -> None:
    """``next_run`` is what the store keeps and the loop list shows; it must agree with the trigger."""
    # Arrange: a weekday loop, asked after Friday's slot has passed
    task = ScheduledTask(
        kind=TaskKind.MANUAL_LOOP,
        cron="0 8 * * 1-5",
        timezone="UTC",
        provider=Provider.INTERACTIVE_SHELL,
    )
    friday_after_slot = datetime(2026, 10, 9, 9, 0, tzinfo=UTC)

    # Act
    next_run = compute_next_run(task, friday_after_slot)

    # Assert: Monday, not Saturday
    assert next_run == "2026-10-12T08:00:00+00:00"


@pytest.mark.parametrize("day_of_week", ["mon-fri/0", "mon-fri/-1", "fri-mon/2"])
def test_invalid_named_weekday_steps_are_rejected(day_of_week: str) -> None:
    with pytest.raises(ValueError):
        build_cron_trigger(f"0 9 * * {day_of_week}", "UTC")


def test_next_run_honors_a_named_weekday_step() -> None:
    task = ScheduledTask(
        kind=TaskKind.MANUAL_LOOP,
        cron="0 9 * * mon-fri/2",
        timezone="UTC",
        provider=Provider.INTERACTIVE_SHELL,
    )
    tuesday_before_slot = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)

    assert compute_next_run(task, tuesday_before_slot) == "2026-10-21T09:00:00+00:00"
