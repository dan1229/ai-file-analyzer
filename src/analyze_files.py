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
        # Only process text-based or structured text (like JSON/XML) files
        ftype = get_file_type(file_path)
        if not (
            ftype.startswith("text/")
            or ftype in ["application/json", "application/xml"]
        ):
            return None

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if len(content.strip()) == 0:
                return None

            # Truncate content if too long to avoid overhead
            truncated_content = content[:1000]
            summary = summarizer(
                truncated_content, max_length=50, min_length=10, do_sample=False
            )
            return summary[0]["summary_text"]
    except Exception as e:
        return f"Error analyzing file: {str(e)}"


def generate_long_report(
    directory_path, total_files, total_size, file_types, file_summaries
):
    """Generate a long descriptive text of the directory and its files."""
    report_lines = []
    report_lines.append(f"Directory: {directory_path}")
    report_lines.append(f"Total number of files: {total_files}")
    report_lines.append(f"Total size (in MB): {total_size / (1024*1024):.2f}")
    report_lines.append("")
    report_lines.append("File type distribution:")
    for ftype, count in file_types.most_common():
        report_lines.append(f"- {ftype}: {count} files")

    if file_summaries:
        report_lines.append("")
        report_lines.append("Individual file summaries:")
        for file_name, summary in file_summaries:
            report_lines.append(f"\nFile: {file_name}")
            report_lines.append(f"Summary: {summary}")

    return "\n".join(report_lines)


def analyze_directory(directory_path, output_file=None):
    """
    Analyze a directory and its contents.

    Args:
        directory_path (str): Path to the directory to analyze
        output_file (str, optional): Path to save the output report
    """
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

    # Generate output content
    output_content = ["=== Directory Analysis Report ==="]
    output_content.append(f"\nTotal Files: {total_files}")
    output_content.append(f"Total Size: {total_size / (1024*1024):.2f} MB")

    output_content.append("\nFile Types Distribution:")
    for file_type, count in file_types.most_common():
        output_content.append(f"- {file_type}: {count} files")

    if file_summaries:
        output_content.append("\nFile Content Summaries:")
        for file_name, summary in file_summaries:
            output_content.append(f"\n{file_name}:")
            output_content.append(f"Summary: {summary}")

    # Generate a long textual report of everything
    long_report = generate_long_report(
        directory_path, total_files, total_size, file_types, file_summaries
    )

    # Produce a final high-level summary
    output_content.append("\n\n=== Final Directory Summary (High-Level) ===")
    final_summary = summarizer(
        long_report, max_length=200, min_length=50, do_sample=False
    )[0]["summary_text"]
    output_content.append(final_summary)

    # Convert output_content to string
    output_text = "\n".join(output_content)

    # Print to stdout
    print(output_text)

    # Write to file if specified
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"\nOutput has been saved to: {output_file}")
        except Exception as e:
            print(f"\nError writing to output file: {str(e)}")


def main():
    # Get directory path from user
    directory = input("Enter the directory path to analyze: ")

    if not os.path.exists(directory):
        print("Error: Directory does not exist!")
        return

    # Get output file path (optional)
    output_file = input("Enter output file path (press Enter to skip): ").strip()
    output_file = output_file if output_file else None

    analyze_directory(directory, output_file)


if __name__ == "__main__":
    main()
