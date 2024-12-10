import os
import logging
import argparse

from src.ai.output_analyzer import OutputAnalyzer
from .utils.file_utils import scan_directory
from .utils.task_analyzer import process_input_file


logger = logging.getLogger(__name__)


def get_validated_directory(default_dir: str, autorun: bool = False) -> str:
    if autorun:
        logger.info(f"Autorun enabled, using default directory: {default_dir}")
        return default_dir

    logger.info("Starting directory validation")
    logger.info("=== Directory Selection ===")
    logger.info(f"Default directory: {default_dir}")
    directory = input(
        "\nEnter the directory path to scan [Press Enter for default]: "
    ).strip()

    if not directory:
        logger.debug("Using default directory")
        return default_dir

    if not os.path.isdir(directory):
        logger.error(f"Invalid directory path provided: {directory}")
        return get_validated_directory(default_dir)

    logger.info(f"Directory validated: {directory}")
    return directory


def get_validated_year(default_year: str, autorun: bool = False) -> str:
    if autorun:
        logger.info(f"Autorun enabled, using default year: {default_year}")
        return default_year

    logger.info("Starting year validation")
    logger.info("=== Year Selection ===")
    logger.info(f"Default year: {default_year}")
    year = input("\nEnter the year to analyze [Press Enter for default]: ").strip()

    if not year:
        logger.debug("Using default year")
        return default_year

    if not year.isdigit() or len(year) != 4:
        logger.error(f"Invalid year format provided: {year}")
        logger.error("Please enter a valid 4-digit year.")
        return get_validated_year(default_year)

    logger.info(f"Year validated: {year}")
    return year


def get_analysis_settings(autorun: bool = False) -> dict:
    if autorun:
        logger.info("Autorun enabled, using AI analysis")
    else:
        logger.info("=== Analysis Type ===")
        logger.info("Using AI-Enhanced analysis")

    return {"ai_enabled": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autorun",
        action="store_true",
        help="Run with default values without prompting",
    )
    args = parser.parse_args()

    logger.info("Starting Task Analyzer")
    logger.info("=" * 30)

    DEFAULT_DIR = os.getcwd()
    DEFAULT_YEAR = "2024"
    DEFAULT_FILE_TYPES = ["md", "txt", "py", "js", "html", "css", "json", "yaml", "yml"]

    directory = get_validated_directory(DEFAULT_DIR, args.autorun)
    year = get_validated_year(DEFAULT_YEAR, args.autorun)
    file_types = DEFAULT_FILE_TYPES

    logger.info("Beginning file scan")
    logger.info("=== Scanning Files ===")
    files = scan_directory(directory, file_types)
    total_files = len(files)
    logger.info(f"Found {total_files} files in {directory}")

    logger.info("=== Processing Tasks ===")
    stats = {"tasks_total": 0, "tasks_completed": 0, "tasks_by_date": {}}

    logger.info("AI Analysis enabled")

    output_files = {}
    for i, file_path in enumerate(files, 1):
        if i % 50 == 0 or i == total_files:
            progress = (i / total_files) * 100
            logger.info(
                f"Processing progress: {progress:.1f}% ({i}/{total_files} files)"
            )

        output_file = process_input_file(file_path, year, stats, True)  # Always use AI
        if output_file:
            output_files[output_file] = output_file

    logger.info("Processing completed, preparing results")
    logger.info("Analyzing results with AI")
    logger.info(OutputAnalyzer().analyze_output(output_files))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s\t| %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
