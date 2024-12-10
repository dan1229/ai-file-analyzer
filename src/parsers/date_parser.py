from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
import re


def is_valid_date(date_str: str) -> bool:
    """Check if a string represents a valid date in supported formats."""
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue
    return False


def extract_top_line_info(
    content: str,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract year, week number, and day of year from the top line of a note."""
    lines = content.splitlines()
    top_line_pattern = (
        r"\[\[([\d]{4}) Daily TODO\]\]\s*-\s*\[Week\]\s*(\d+)\s*/\s*\[Day\]\s*(\d+)"
    )

    for line in lines:
        match = re.search(top_line_pattern, line)
        if match:
            year = int(match.group(1))
            week_num = int(match.group(2))
            day_of_year = int(match.group(3))
            return year, week_num, day_of_year
    return None, None, None


def get_base_date(year: int, day_of_year: int) -> datetime:
    """Convert year and day of year to a datetime object."""
    return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)


def extract_date_from_tags(content: str) -> Optional[str]:
    """Extract date from the tags section of a note."""
    lines = content.splitlines()
    date = None
    tags_section = False

    for line in lines:
        if line.strip() == "## Tags":
            tags_section = True
            continue
        if tags_section:
            tags = line.strip().split()
            for tag in tags:
                if tag.startswith("#") and len(tag) > 1:
                    tag_content = tag[1:]
                    if is_valid_date(tag_content):
                        date = tag_content
                        break
            break
    return date


def build_day_to_date_map(
    year: Optional[int],
    day_of_year: Optional[int],
    content: str,
    fallback_date: Optional[str] = None,
) -> Dict[str, str]:
    """Build a mapping from day names to dates."""
    lines = content.splitlines()

    day_map = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
    }

    if year is not None and day_of_year is not None:
        base_date = get_base_date(year, day_of_year)

        def python_to_template_day(python_wd: int) -> int:
            return (python_wd + 1) % 7

        base_template_day = python_to_template_day(base_date.weekday())
    elif fallback_date:
        base_date = datetime.strptime(fallback_date, "%m-%d-%Y")
        base_template_day = 0
    else:
        return {}

    found_days = []
    for line in lines:
        if line.startswith("### ") or line.startswith("#### "):
            heading_text = (
                line[4:].strip() if line.startswith("#### ") else line[3:].strip()
            )
            dname = heading_text.split(" ")[0].title()
            if dname in day_map:
                found_days.append(dname)

    if not found_days:
        return {}

    day_to_date_map = {}
    for dname in found_days:
        offset = day_map[dname] - base_template_day
        day_to_date_map[dname] = (base_date + timedelta(days=offset)).strftime(
            "%m-%d-%Y"
        )

    return day_to_date_map
