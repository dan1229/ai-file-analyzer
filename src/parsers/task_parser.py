from typing import List, Optional, Dict, Any


def parse_task_line(line: str) -> Optional[Dict[str, str]]:
    """Parse a single task line and return its status and text."""
    if line.startswith("- [x]") or line.startswith("- [ ]"):
        status = "completed" if line[3] == "x" else "not completed"
        text = line[5:].strip()
        return {"status": status, "text": text}
    return None


def get_indent_level(line: str) -> int:
    """Calculate the indentation level of a line."""
    stripped_line = line.lstrip(" \t")
    indent_level = len(line) - len(stripped_line)
    return indent_level


def extract_tasks(
    content: str, day_to_date_map: Dict[str, str], stats: Dict[str, Any]
) -> None:
    """Extract tasks from content and update stats."""
    from ..utils.task_analyzer import (
        process_task,
    )  # Import here to avoid circular dependency

    lines: List[str] = content.splitlines()
    task_stack: List[Dict[str, str]] = []
    prev_indent_level = 0
    current_date = None

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("#### ") or stripped_line.startswith("### "):
            heading_text = (
                stripped_line[4:].strip()
                if stripped_line.startswith("#### ")
                else stripped_line[3:].strip()
            )
            day_name = heading_text.split(" ")[0].title()

            if day_name in day_to_date_map:
                current_date = day_to_date_map[day_name]
            else:
                current_date = None
            task_stack = []
            prev_indent_level = 0
            continue

        if stripped_line.startswith("- ["):
            task_info = parse_task_line(stripped_line)
            if task_info:
                indent_level = get_indent_level(line)

                if indent_level > prev_indent_level:
                    task_stack.append(task_info)
                elif indent_level == prev_indent_level:
                    if task_stack:
                        task_stack[-1] = task_info
                    else:
                        task_stack.append(task_info)
                else:
                    levels_up = prev_indent_level - indent_level
                    task_stack = task_stack[:-levels_up]
                    if task_stack:
                        task_stack[-1] = task_info
                    else:
                        task_stack.append(task_info)

                prev_indent_level = indent_level
                process_task(task_stack.copy(), current_date, stats)
        else:
            prev_indent_level = 0
            task_stack = []
