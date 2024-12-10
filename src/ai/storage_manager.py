from datetime import datetime
import json
import os
from typing import Dict, Any, Optional
import logging

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger(__name__)


class AnalysisStorage:
    """Manages storage and retrieval of AI analysis results."""

    def __init__(self, base_dir: str):
        """
        Initialize storage manager.

        Args:
            base_dir: Base directory for storing analysis results
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.analysis_dir = os.path.join(self.base_dir, "analysis", current_datetime)
        self.analysis_file = os.path.join(self.analysis_dir, "analyses.json")
        self._ensure_storage_exists()
        self._ensure_file_exists()

    def _ensure_storage_exists(self) -> None:
        """Ensure the storage directory exists."""
        os.makedirs(self.analysis_dir, exist_ok=True)

    def _ensure_file_exists(self) -> None:
        """Ensure the analysis file exists with an empty list."""
        if not os.path.exists(self.analysis_file):
            with open(self.analysis_file, "w") as f:
                json.dump([], f)

    def store_analysis(self, date: Optional[str], analysis: Dict[str, Any]) -> str:
        """
        Store analysis results by appending to the main analysis file.

        Args:
            date: Date string in format MM-DD-YYYY
            analysis: Analysis results to store
        """
        try:
            current_analyses = []
            if os.path.exists(self.analysis_file):
                with open(self.analysis_file, "r") as f:
                    current_analyses = json.load(f)

            analysis_entry = {"date": date, "data": analysis}
            current_analyses.append(analysis_entry)

            with open(self.analysis_file, "w") as f:
                json.dump(current_analyses, f, indent=2)
            logger.debug("Successfully stored analysis!")
        except Exception as e:
            logger.error(f"Error storing analysis for {date}: {str(e)}")
            raise

        return self.analysis_file
