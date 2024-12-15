import os
from pathlib import Path
import mimetypes
from collections import Counter
from transformers import pipeline
import torch


def get_file_type(file_path):
    """Determine file type based on extension and mime type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "unknown"


def analyze_file_content(file_path, summarizer):
    """Analyze text content of supported file types."""
    try:
        # Only process text-based files
        if not get_file_type(file_path).startswith(
            ("text/", "application/json", "application/xml")
        ):
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content.strip()) == 0:
                return None

            # Truncate content if too long
            content = content[:1000]  # Analyze first 1000 chars
            summary = summarizer(content, max_length=50, min_length=10, do_sample=False)
            return summary[0]["summary_text"]
    except Exception as e:
        return f"Error analyzing file: {str(e)}"


def analyze_directory(directory_path):
    """Analyze a directory and its contents."""
    # Initialize the summarizer
    print("Loading AI model...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Collect directory statistics
    total_files = 0
    file_types = Counter()
    file_summaries = []
    total_size = 0

    print(f"\nAnalyzing directory: {directory_path}")

    # Walk through directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = Path(root) / file
            total_files += 1

            # Get file stats
            file_type = get_file_type(file_path)
            file_types[file_type] += 1
            total_size += file_path.stat().st_size

            # Get file summary if possible
            summary = analyze_file_content(file_path, summarizer)
            if summary:
                file_summaries.append((file, summary))

    # Generate report
    print("\n=== Directory Analysis Report ===")
    print(f"\nTotal Files: {total_files}")
    print(f"Total Size: {total_size / (1024*1024):.2f} MB")

    print("\nFile Types Distribution:")
    for file_type, count in file_types.most_common():
        print(f"- {file_type}: {count} files")

    print("\nFile Content Summaries:")
    for file_name, summary in file_summaries:
        print(f"\n{file_name}:")
        print(f"Summary: {summary}")


def main():
    # Get directory path from user
    directory = input("Enter the directory path to analyze: ")

    if not os.path.exists(directory):
        print("Error: Directory does not exist!")
        return

    analyze_directory(directory)


if __name__ == "__main__":
    main()
