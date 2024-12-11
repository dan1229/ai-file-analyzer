from typing import Any, List, Union, Dict
import logging
import json

logger = logging.getLogger(__name__)


class OutputAnalyzer:
    """Analyzes file statistics and provides summaries."""

    _instance = None
    _initialized = False

    def __new__(cls) -> "OutputAnalyzer":
        """Implement singleton pattern to ensure only one instance."""
        if cls._instance is None:
            cls._instance = super(OutputAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the analyzer settings."""
        if self._initialized:
            return
        self._initialized = True

    def analyze_output(self, stats: Dict[str, Any]) -> str:
        """
        Analyze the file statistics and produce a readable report.
        """
        # Add debug logging to inspect the stats
        logger.debug(f"Received stats for analysis: {json.dumps(stats, indent=2)}")

        if not stats:
            logger.warning("Empty stats received for analysis")
            return "No data available for analysis."

        tasks_by_date = stats.get("tasks_by_date")
        if not tasks_by_date:
            logger.warning("No tasks_by_date found in stats")
            return "No data available for analysis."

        analysis = ["# File Analysis Report\n"]

        # Overall Statistics
        total_files = 0
        total_lines = 0
        total_words = 0

        for date, date_stats in tasks_by_date.items():
            total_files += date_stats.get("files_processed", 0)
            total_lines += date_stats.get("total_lines", 0)
            total_words += date_stats.get("total_words", 0)

        analysis.append("## Overall Statistics\n")
        analysis.append(f"- Total files processed: {total_files}\n")
        analysis.append(f"- Total lines analyzed: {total_lines}\n")
        analysis.append(f"- Total words counted: {total_words}\n")
        if total_files > 0:
            analysis.append(
                f"- Average lines per file: {total_lines/total_files:.1f}\n"
            )
            analysis.append(
                f"- Average words per file: {total_words/total_files:.1f}\n\n"
            )

        # Statistics by Date
        analysis.append("## Statistics by Date\n")
        for date, date_stats in sorted(tasks_by_date.items()):
            analysis.append(f"\n### {date}\n")
            analysis.append(
                f"- Files processed: {date_stats.get('files_processed', 0)}\n"
            )
            analysis.append(f"- Total lines: {date_stats.get('total_lines', 0)}\n")
            analysis.append(f"- Total words: {date_stats.get('total_words', 0)}\n")

            # File Details
            if date_stats.get("files"):
                analysis.append("\nFile Details:\n")
                for file_info in date_stats["files"]:
                    analysis.append(f"- {file_info['path']}:\n")
                    analysis.append(f"  - Size: {file_info['size']} bytes\n")
                    analysis.append(f"  - Lines: {file_info['lines']}\n")
                    analysis.append(f"  - Words: {file_info['words']}\n")

        return "".join(analysis)
