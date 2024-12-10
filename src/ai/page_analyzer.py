import re
from typing import Dict, Any, List, Optional
from transformers import pipeline
from collections import Counter
import dateparser
import logging

logger = logging.getLogger(__name__)


class PageAnalyzer:
    """Analyzes page content with optional ML capabilities."""

    _instance = None
    _initialized = False
    MAX_TOKEN_LENGTH = 128
    SENTIMENT_ANALYZER: Optional[pipeline] = None

    def __new__(cls) -> "PageAnalyzer":
        """Ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super(PageAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize basic analyzer settings."""
        if self._initialized:
            return

        self.has_ml_capabilities = False
        self.SENTIMENT_ANALYZER = None
        self._initialized = True

    def _initialize_ml(self) -> None:
        """Lazy load ML models only when needed."""
        if self.SENTIMENT_ANALYZER is not None:
            return

        try:
            import torch
            import gc

            logger.info("Initializing ML models (this may take a moment)...")

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Initialize on CPU with minimal memory footprint
            self.SENTIMENT_ANALYZER = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=1,
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": True},
            )

            self.has_ml_capabilities = True
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"ML capabilities disabled: {str(e)}")
            logger.info("Running in basic analysis mode")
            self.has_ml_capabilities = False

    def analyze_page(self, content: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single page and return structured data."""
        try:
            # Basic stats (always available)
            word_count = len(content.split())
            habits = self._extract_habits(content)
            projects = self._extract_projects(content)

            analysis: Dict[str, Any] = {
                "stats": {
                    "word_count": word_count,
                    "habits": habits,
                    "projects": projects,
                },
                "summary": "",  # Initialize with empty string
            }

            # Add ML-based analysis if available
            if self.has_ml_capabilities:
                try:
                    analysis["summary"] = self._generate_summary(
                        content[:10000]
                    )  # Now correctly typed
                    analysis["stats"]["sentiment"] = self._analyze_sentiment(
                        content[:10000]
                    )
                except Exception as e:
                    logger.warning(f"ML analysis failed for {date} - {str(e)}")
                    analysis["summary"] = self._basic_summary(content)
                    analysis["stats"]["sentiment"] = self._basic_sentiment(content)
            else:
                analysis["summary"] = self._basic_summary(content)
                analysis["stats"]["sentiment"] = self._basic_sentiment(content)

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing page for {date}: {str(e)}")
            return {
                "summary": "Analysis failed",
                "stats": {
                    "word_count": 0,
                    "sentiment": "neutral",
                    "habits": [],
                    "projects": [],
                },
            }

    def _chunk_text(self, text: str, max_length: int = 200) -> List[str]:
        """Split text into chunks of approximately max_length words."""
        try:
            words: List[str] = text.split()
            chunks: List[str] = []
            current_chunk: List[str] = []
            current_length = 0

            for word in words:
                current_length += len(word) + 1  # +1 for space
                if current_length > max_length and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)

            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks
        except Exception as e:
            logger.warning(f"Text chunking failed - {str(e)}")
            return [text[:max_length]]

    def _analyze_sentiment(self, content: str) -> str:
        """Analyze sentiment using ML if available."""
        try:
            if not self.has_ml_capabilities:
                return self._basic_sentiment(content)

            # Initialize ML models if needed
            self._initialize_ml()

            # Split content into smaller chunks
            chunks = self._chunk_text(content, max_length=100)  # Reduced chunk size

            if self.SENTIMENT_ANALYZER is None:
                raise Exception("ML model not initialized")

            # Analyze sentiment for each chunk
            sentiments: List[str] = []
            for chunk in chunks[:3]:  # Reduced from 5 to 3 chunks
                if not chunk.strip():
                    continue
                try:
                    result = self.SENTIMENT_ANALYZER(chunk)[0]
                    sentiments.append(result["label"].lower())
                except Exception:
                    continue

            if not sentiments:
                return self._basic_sentiment(content)

            # Return most common sentiment
            return str(Counter(sentiments).most_common(1)[0][0])

        except Exception as e:
            logger.warning(f"ML sentiment analysis failed - {str(e)}")
            return self._basic_sentiment(content)

    def _generate_summary(self, content: str) -> str:
        """Generate a summary using basic approach."""
        return self._basic_summary(content)

    def _extract_habits(self, content: str) -> List[Dict[str, Any]]:
        """Extract habits and their frequencies."""
        # Use your existing habits list from task_analyzer.py
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

        content_lower: str = content.lower()
        habit_counts: Counter[str] = Counter()

        for habit in habits_list:
            count = content_lower.count(habit)
            if count > 0:
                habit_counts[habit] = count

        return [
            {"name": habit, "count": count}
            for habit, count in habit_counts.most_common()
        ]

    def _extract_projects(self, content: str) -> List[Dict[str, Any]]:
        """Extract project mentions and their frequencies."""
        # Look for project indicators like #project or [Project]
        project_pattern = r"(?:#project/|#p/|\[Project\]:\s*)([^\n\[\]#]+)"
        projects = re.findall(project_pattern, content, re.IGNORECASE)

        project_counts = Counter(p.strip().lower() for p in projects)
        return [
            {"name": project, "count": count}
            for project, count in project_counts.most_common()
        ]

    def _basic_summary(self, content: str) -> str:
        """Generate a basic summary by taking the first few sentences."""
        sentences = content.split(".")
        summary = ". ".join(sentences[:3]) + "."
        return summary.strip()

    def _basic_sentiment(self, content: str) -> str:
        """Generate a basic sentiment analysis using comprehensive keyword matching.

        Returns one of: very_positive, positive, neutral, negative, very_negative
        """
        text = content.lower()

        indicators = {
            "very_positive": [
                "!!!",
                "+++",
                "amazing",
                "excellent",
                "fantastic",
                "perfect",
                "incredible",
                "outstanding",
                "brilliant",
                "superb",
                "wonderful",
                "exceptional",
                "thrilled",
                "overjoyed",
                "ecstatic",
                "delighted",
                "magnificent",
                "spectacular",
                "awesome",
                "phenomenal",
                "extraordinary",
                "inspiring",
                "love it",
                "best ever",
                "blessed",
                "grateful",
                "thankful",
                "breakthrough",
                "triumph",
                "success",
                "victory",
            ],
            "positive": [
                "+",
                "good",
                "happy",
                "great",
                "nice",
                "well",
                "glad",
                "pleased",
                "enjoyed",
                "satisfying",
                "productive",
                "accomplished",
                "achieved",
                "improved",
                "better",
                "progress",
                "promising",
                "hopeful",
                "optimistic",
                "motivated",
                "encouraged",
                "calm",
                "peaceful",
                "relaxed",
                "comfortable",
                "content",
                "satisfied",
                "fun",
                "excited",
                "looking forward",
                "proud",
                "confident",
                "successful",
                "effective",
            ],
            "negative": [
                "-",
                "bad",
                "sad",
                "poor",
                "tough",
                "difficult",
                "unfortunate",
                "challenging",
                "frustrated",
                "disappointing",
                "worried",
                "concerned",
                "anxious",
                "stressed",
                "tired",
                "exhausted",
                "overwhelmed",
                "struggle",
                "problem",
                "issue",
                "setback",
                "failed",
                "upset",
                "unhappy",
                "annoyed",
                "irritated",
                "bothered",
                "confused",
                "doubtful",
                "uncertain",
                "uneasy",
                "mediocre",
                "subpar",
                "could be better",
            ],
            "very_negative": [
                "--",
                "terrible",
                "awful",
                "horrible",
                "worst",
                "devastating",
                "miserable",
                "disaster",
                "catastrophe",
                "dreadful",
                "hopeless",
                "despair",
                "depressed",
                "furious",
                "angry",
                "hate",
                "painful",
                "unbearable",
                "crisis",
                "failure",
                "nightmare",
                "tragic",
                "waste",
                "regret",
                "disgusting",
                "appalling",
                "cruel",
                "horrific",
                "emergency",
                "breakdown",
                "impossible",
            ],
        }

        # Initialize scores with weights
        scores = {
            "very_positive": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "very_negative": 0,
        }

        # Check first few paragraphs (increased from 500 to 1000 chars)
        sample_text = text[:1000]

        # Apply weighted scoring
        for sentiment, words in indicators.items():
            for word in words:
                count = sample_text.count(word)
                if sentiment in ["very_positive", "very_negative"]:
                    scores[sentiment] += (
                        count * 2
                    )  # Double weight for extreme sentiments
                else:
                    scores[sentiment] += count

        # Determine the dominant sentiment with more nuanced logic
        max_score = max(scores.values())
        if max_score == 0:
            return "neutral"
        elif max_score <= 2:  # If the highest score is low, lean towards neutral
            return "neutral"

        # Return the sentiment with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def extract_date(self, content: str) -> Optional[str]:
        """Extract date from content using multiple strategies.

        Returns:
            str: ISO format date (YYYY-MM-DD) if found, None otherwise
        """
        try:
            # Common date patterns
            patterns = [
                # ISO format: 2024-03-20
                r"\b\d{4}-\d{2}-\d{2}\b",
                # Common formats: March 20, 2024; 20 March 2024; 20/03/2024
                r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
                r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
                r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
                # Header format: # 2024-03-20 | Wednesday
                r"#\s*(\d{4}-\d{2}-\d{2})",
                # Natural language: "today", "yesterday", "last Friday"
                r"\b(?:today|yesterday|last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
            ]

            # Check first 500 characters for date patterns
            content_start = content[:500].lower()

            # Try each pattern
            for pattern in patterns:
                matches = re.findall(pattern, content_start, re.IGNORECASE)
                if matches:
                    # Try parsing the first match
                    parsed_date = dateparser.parse(matches[0])
                    if parsed_date:
                        return str(parsed_date.strftime("%Y-%m-%d"))

            # If no matches found, try dateparser's generic parsing
            # on the first line (often contains the date)
            first_line = content.split("\n")[0]
            parsed_date = dateparser.parse(first_line)
            if parsed_date:
                return str(parsed_date.strftime("%Y-%m-%d"))

            return None

        except Exception as e:
            logger.warning(f"Date extraction failed - {str(e)}")
            return None
