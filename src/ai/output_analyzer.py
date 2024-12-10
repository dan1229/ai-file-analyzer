from typing import List, Union, Dict
import logging
import json
import gc

import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class OutputAnalyzer:
    """
    Analyzes output files and provides summaries or statistics.

    This class uses a local LLM (via transformers) to summarize the content
    of JSON files
    """

    _instance = None
    _initialized = False
    SUMMARIZER = None

    def __new__(cls) -> "OutputAnalyzer":
        """Implement singleton pattern to ensure only one instance."""
        if cls._instance is None:
            cls._instance = super(OutputAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the analyzer settings."""
        if self._initialized:
            return
        self.has_ml_capabilities = False
        self.SUMMARIZER = None
        self._initialized = True

    def _initialize_ml(self) -> None:
        """Lazy load the ML models for summarization."""

        try:
            logger.info("Initializing ML summarization model...")

            # Clear GPU/CPU memory before loading model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_name = "sshleifer/distilbart-cnn-6-6"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
            ).cpu()

            self.SUMMARIZER = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device="cpu",
                framework="pt",
            )
            self.has_ml_capabilities = True
            logger.info("ML summarization model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ML summarization model: {e}")
            logger.info("Falling back to basic analysis mode.")
            self.has_ml_capabilities = False

    def _read_and_combine_data(
        self, output_files: Union[List[str], Dict[str, str]]
    ) -> List[Dict]:
        """
        Read and combine JSON data from provided output files.

        :param output_files: A dictionary mapping file keys to file paths,
                             or a list of file paths.
        :return: A combined list of all JSON entries from all files.
        """
        if isinstance(output_files, dict):
            files_to_process = list(output_files.values())
        else:
            files_to_process = output_files

        combined_data = []
        for file_path in files_to_process:
            try:
                logger.info(f"Reading file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    combined_data.extend(data)
                logger.info(f"Successfully read {len(data)} entries from {file_path}")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        logger.info(f"Total entries combined: {len(combined_data)}")
        return combined_data

    def _chunk_text(self, text: str, max_length: int = 1024) -> List[str]:
        """
        Split text into chunks of specified max_length characters.

        :param text: The text to chunk.
        :param max_length: Maximum length of each chunk.
        :return: A list of text chunks.
        """
        return [text[i : i + max_length] for i in range(0, len(text), max_length)]

    def analyze_output(
        self, output_files: Union[List[str], Dict[str, str]], max_chunks: int = 3
    ) -> str:
        """
        Analyze the output files using local LLM or basic stats. Attempts to produce a structured
        and meaningful summary with clearly defined sections.
        """
        logger.info(f"Analyzing output files: {output_files}")
        combined_data = self._read_and_combine_data(output_files)

        # Initialize ML if needed
        self._initialize_ml()

        # If ML is not available, fallback to basic analysis.
        if not self.has_ml_capabilities or not self.SUMMARIZER:
            return self._basic_analysis(combined_data)

        try:
            # Use only a subset of data for summarization to avoid large prompt issues
            data_sample = json.dumps(combined_data[:100], indent=2)

            # Well-structured prompts that ask the summarizer to present the findings
            # in a clear, bulleted format.
            prompts: List[str] = [
                (
                    "You are a data analyst. You have been given the following JSON data sample. "
                    "First, identify the main insights and present them as a short list of bullet points:\n\n"
                    f"{data_sample}"
                    "\n\nPlease provide a concise list of key insights (bulleted) without extra commentary."
                ),
                (
                    "Now, examine the data again and identify recurring patterns and significant trends. "
                    "Present them as a short list of bullet points:\n\n"
                    f"{data_sample}"
                    "\n\nPlease provide a concise list of patterns and trends (bulleted) without extra commentary."
                ),
                (
                    "Finally, summarize the most important findings and notable observations. "
                    "Give a short bullet-pointed list focusing on crucial takeaways:\n\n"
                    f"{data_sample}"
                    "\n\nPlease provide the key findings (bulleted), focusing on the essential conclusions."
                ),
            ]

            summaries: List[str] = []
            total_chunks = len(prompts)

            for i, prompt in enumerate(prompts, 1):
                logger.info(f"Processing analysis section {i}/{total_chunks}")
                summary = self.SUMMARIZER(
                    prompt,
                    max_length=200,
                    min_length=50,
                    do_sample=False,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )
                # Expecting the summarizer to return a list with one dictionary: [{"summary_text": "..."}]
                summaries.append(summary[0]["summary_text"].strip())

            # Construct the final analysis report
            analysis: str = "# Detailed Analysis Report\n\n"

            analysis += "## Key Insights\n"
            analysis += summaries[0] + "\n\n"

            analysis += "## Patterns and Trends\n"
            analysis += summaries[1] + "\n\n"

            analysis += "## Important Findings\n"
            analysis += summaries[2] + "\n\n"

            # Add statistical overview
            analysis += "## Statistical Overview\n"
            analysis += self._get_statistical_highlights(combined_data)

            return analysis

        except Exception as e:
            logger.error(f"ML-based summarization failed: {e}")
            return self._basic_analysis(combined_data)

    def _get_statistical_highlights(self, data: List[Dict]) -> str:
        """
        Generate statistical highlights from the data.
        """
        highlights = []

        # Total records
        highlights.append(f"- Total records analyzed: {len(data)}")

        # Field completeness
        for field in data[0].keys():
            non_null = sum(1 for item in data if item.get(field) is not None)
            completeness = (non_null / len(data)) * 100
            highlights.append(f"- {field} field completeness: {completeness:.1f}%")

        # Unique values per field
        for field in data[0].keys():
            unique_values = len(
                {str(item.get(field)) for item in data if item.get(field) is not None}
            )
            highlights.append(f"- Unique values in {field}: {unique_values}")

        return "\n".join(highlights) + "\n"

    def _basic_analysis(self, data: List[Dict]) -> str:
        """
        Perform basic statistical analysis on the data.

        :param data: The data to analyze.
        :return: A formatted analysis string with key statistics.
        """
        if not data:
            return "No data available for analysis."

        analysis = ["# Basic Analysis Report\n"]

        # Data Overview
        analysis.append("## Overview\n")
        analysis.append(f"- Total records analyzed: {len(data)}\n")
        analysis.append(f"- Fields present: {', '.join(data[0].keys())}\n\n")

        # Field Statistics
        analysis.append("## Field Statistics\n")
        for field in data[0].keys():
            # Count non-null values
            non_null_count = sum(1 for item in data if item.get(field) is not None)
            null_percentage = ((len(data) - non_null_count) / len(data)) * 100

            # Determine field type and collect relevant stats
            values = [item.get(field) for item in data if item.get(field) is not None]
            if values:
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric field statistics
                    avg = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    analysis.append(f"### {field}\n")
                    analysis.append("- Type: Numeric\n")
                    analysis.append(f"- Average: {avg:.2f}\n")
                    analysis.append(f"- Min: {min_val}\n")
                    analysis.append(f"- Max: {max_val}\n")
                elif all(isinstance(v, str) for v in values):
                    # String field statistics
                    unique_values = len(set(values))
                    most_common = max(set(values), key=values.count)
                    avg_length = sum(len(str(v)) for v in values) / len(values)
                    analysis.append(f"### {field}\n")
                    analysis.append("- Type: Text\n")
                    analysis.append(f"- Unique values: {unique_values}\n")
                    analysis.append(f"- Most common: {most_common}\n")
                    analysis.append(f"- Average length: {avg_length:.1f} characters\n")

            analysis.append(f"- Null percentage: {null_percentage:.1f}%\n\n")

        # Value Distribution (for fields with manageable unique values)
        analysis.append("## Value Distribution\n")
        for field in data[0].keys():
            values = [
                str(item.get(field)) for item in data if item.get(field) is not None
            ]
            unique_values = set(values)
            if (
                len(unique_values) <= 10
            ):  # Only show distribution for fields with few unique values
                analysis.append(f"\n### {field} Distribution\n")
                for value in unique_values:
                    count = values.count(value)
                    percentage = (count / len(values)) * 100
                    analysis.append(f"- {value}: {count} ({percentage:.1f}%)\n")

        return "".join(analysis)
