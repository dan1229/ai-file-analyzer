from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime


class DocumentProcessor:
    """Simple document processor for extracting structured information"""

    def __init__(self, file_paths: List[str], config_path: str):
        self.file_paths = file_paths
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def process_content(self, content: str) -> Dict[str, Any]:
        """Extract basic information from content"""
        return {
            "content_length": len(content),
            "line_count": len(content.splitlines()),
            "word_count": len(content.split()),
            "has_content": bool(content.strip()),
        }


def process_task(task_data: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """Process file information and update statistics."""
    date = datetime.now().strftime("%Y-%m-%d")

    if date not in stats["tasks_by_date"]:
        stats["tasks_by_date"][date] = {
            "files_processed": 0,
            "total_lines": 0,
            "total_words": 0,
            "files": [],
        }

    current_date_stats = stats["tasks_by_date"][date]
    current_date_stats["files_processed"] += 1
    current_date_stats["total_lines"] += task_data.get("line_count", 0)
    current_date_stats["total_words"] += task_data.get("word_count", 0)

    if "path" in task_data:
        current_date_stats["files"].append(
            {
                "path": task_data["path"],
                "size": task_data.get("content_length", 0),
                "lines": task_data.get("line_count", 0),
                "words": task_data.get("word_count", 0),
            }
        )


def process_input_file(
    file_path: str,
    config_path: str,
    stats: Dict[str, Any],
    batch_size: int = 32,  # kept for compatibility
) -> Optional[str]:
    """
    Process files using simple text analysis.
    """
    try:
        # Skip binary files and very large files
        if not os.path.isfile(file_path):
            return None

        file_size = os.path.getsize(file_path)
        if file_size > 1024 * 1024:  # Skip files larger than 1MB
            print(f"Skipping large file {file_path}: {file_size / 1024 / 1024:.2f}MB")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Skipping binary file {file_path}")
            return None

        processor = DocumentProcessor([file_path], config_path)
        task_data = processor.process_content(content)
        task_data["path"] = file_path

        process_task(task_data, stats)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

    return None
