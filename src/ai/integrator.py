from typing import Optional
from .page_analyzer import PageAnalyzer
from .storage_manager import AnalysisStorage
import logging


class AIIntegrator:
    """Handles integration between the main app and AI components."""

    def __init__(self, base_dir: str):
        """
        Initialize AI integration.

        Args:
            base_dir: Base directory for the application
        """
        self.analyzer = PageAnalyzer()
        self.storage = AnalysisStorage(base_dir)
        self.logger = logging.getLogger(__name__)

    def process_file(self, file_path: str, date: Optional[str] = None) -> str:
        """
        Process a single file with AI analysis.

        Args:
            file_path: Path to the file to analyze
            date: Optional date string (MM-DD-YYYY)
        """
        try:
            self.logger.debug(f"Processing file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            extracted_date = date or self.analyzer.extract_date(content)
            self.logger.debug(f"Extracted date: {extracted_date}")

            analysis = self.analyzer.analyze_page(content, extracted_date)
            filepath = self.storage.store_analysis(extracted_date, analysis)
            self.logger.debug(
                f"Successfully processed and stored analysis for date: {extracted_date}"
            )
            return filepath
        except Exception as e:
            self.logger.error(
                f"Error processing file {file_path}: {str(e)}", exc_info=True
            )
            return ""
