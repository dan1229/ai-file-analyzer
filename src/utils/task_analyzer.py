from typing import List, Dict, Any, Optional
import os


from ..parsers.date_parser import (
    extract_top_line_info,
    extract_date_from_tags,
    build_day_to_date_map,
)
from ..parsers.task_parser import extract_tasks


def process_task(
    task_stack: List[Dict[str, str]], date: Optional[str], stats: Dict[str, Any]
) -> None:
    """Process a task and its subtasks, updating the statistics dictionary."""
    if not date:
        return

    # Initialize date stats if not exists
    if date not in stats["tasks_by_date"]:
        stats["tasks_by_date"][date] = {
            "tasks_total": 0,
            "tasks_completed": 0,
            "workouts": [],
            "habits": {},
        }

    date_stats = stats["tasks_by_date"][date]
    task_text = task_stack[-1]["text"].lower()
    status = task_stack[-1]["status"]

    # Update overall stats
    stats["tasks_total"] += 1
    if status == "completed":
        stats["tasks_completed"] += 1

    # Update per-date stats
    date_stats["tasks_total"] += 1
    if status == "completed":
        date_stats["tasks_completed"] += 1

    # Process workouts
    if "workout" in task_text or "🏋️" in task_text:
        if len(task_stack) > 1:
            # Add each sub-task as a workout variant
            for sub_task in task_stack[1:]:
                date_stats["workouts"].append(
                    {"type": sub_task["text"], "status": sub_task["status"]}
                )
        else:
            date_stats["workouts"].append({"type": task_text, "status": status})

    # Process habits
    habits_list = [
        "study hebrew",
        "massage scalp",
        "red light mask",
        "take out trash",
        "check plants water",
        "weekly planning",
        "fantasy waivers",
        "set fantasy line ups",
    ]

    for habit in habits_list:
        if habit in task_text:
            if habit not in date_stats["habits"]:
                date_stats["habits"][habit] = []
            date_stats["habits"][habit].append({"status": status})


def process_input_file(
    file_path: str, input_year: str, stats: Dict[str, Any], use_ai: bool = False
) -> Optional[str]:
    """
    Process a single input file and update the statistics.
    Returns potentially the filename where the output / analysis for this file is stored.
    """
    # Only import AI components if needed
    if use_ai:
        from ..ai.integrator import AIIntegrator

        ai = AIIntegrator(os.path.dirname(file_path))
    else:
        ai = None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract date information
    year, week_num, day_of_year = extract_top_line_info(content)
    date_from_tags = extract_date_from_tags(content)

    # Skip if year doesn't match input year
    # TODO what to do here?
    # if input_year and year and str(year) != input_year:
    #     return None

    # Process with AI if enabled and we have a valid date
    day_to_date_map = {}

    if ai:
        return ai.process_file(file_path)
    else:
        # Handle weekly vs daily files
        if year is not None and day_of_year is not None:
            day_to_date_map = build_day_to_date_map(year, day_of_year, content)
        else:
            if date_from_tags and date_from_tags.endswith(input_year):
                day_to_date_map = build_day_to_date_map(
                    None, None, content, fallback_date=date_from_tags
                )

        if day_to_date_map:
            extract_tasks(content, day_to_date_map, stats)
        else:
            if date_from_tags and date_from_tags.endswith(input_year):
                single_day_map = {"All": date_from_tags}
                extract_tasks(content, single_day_map, stats)

    return None
