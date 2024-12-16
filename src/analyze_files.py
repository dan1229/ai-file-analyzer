import os
from pathlib import Path
import mimetypes
from collections import Counter
from transformers import pipeline
import torch
from datetime import datetime


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


def analyze_year_wrapped(directory_path, summarizer, year=None):
    """Generate a 'Year Wrapped' style analysis of files from the specified year."""
    # Use current year if not specified
    year = year or datetime.now().year

    print(f"\n=== {year} Wrapped ===")

    year_stats = {
        "total_files": 0,
        "total_size": 0,
        "busiest_month": Counter(),
        "file_types": Counter(),
        "largest_files": [],  # Will store (size, path) tuples
        "most_edited_files": [],  # Will store (edit_count, path) tuples
        "summaries": [],
    }

    # Walk through directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = Path(root) / file

            try:
                # Get file stats
                stat = file_path.stat()
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                created_time = datetime.fromtimestamp(stat.st_ctime)

                # Only process files from the specified year
                if modified_time.year == year or created_time.year == year:
                    year_stats["total_files"] += 1
                    year_stats["total_size"] += stat.st_size

                    # Track monthly activity
                    year_stats["busiest_month"][modified_time.strftime("%B")] += 1

                    # Track file types
                    file_type = get_file_type(file_path)
                    year_stats["file_types"][file_type] += 1

                    # Track large files
                    year_stats["largest_files"].append((stat.st_size, file_path))

                    # Get content summary for text files
                    summary = analyze_file_content(file_path, summarizer)
                    if summary:
                        year_stats["summaries"].append((file, summary))

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

    # Sort and trim largest files
    year_stats["largest_files"].sort(reverse=True)
    year_stats["largest_files"] = year_stats["largest_files"][:5]

    # Generate wrapped report
    wrapped_report = [f"\n🎉 Your {year} in Files 🎉"]

    wrapped_report.append("\n📊 By the Numbers:")
    wrapped_report.append(
        f"- You created or modified {year_stats['total_files']} files"
    )
    wrapped_report.append(
        f"- Total size: {year_stats['total_size'] / (1024*1024):.2f} MB"
    )

    wrapped_report.append("\n📅 Your Busiest Months:")
    for month, count in year_stats["busiest_month"].most_common(3):
        wrapped_report.append(f"- {month}: {count} files")

    wrapped_report.append("\n📁 Your Top File Types:")
    for ftype, count in year_stats["file_types"].most_common(5):
        wrapped_report.append(f"- {ftype}: {count} files")

    wrapped_report.append("\n🏋️ Your Largest Files:")
    for size, path in year_stats["largest_files"]:
        wrapped_report.append(f"- {path.name}: {size / (1024*1024):.2f} MB")

    if year_stats["summaries"]:
        wrapped_report.append("\n📝 Highlights from Your Text Files:")
        for _, summary in year_stats["summaries"][:5]:
            wrapped_report.append(f"- {summary}")

    return "\n".join(wrapped_report)


def analyze_directory(directory_path, output_file=None, year_wrapped=False):
    """
    Analyze a directory and its contents.

    Args:
        directory_path (str): Path to the directory to analyze
        output_file (str, optional): Path to save the output report
        year_wrapped (bool): Whether to perform year wrapped analysis
    """
    # Initialize the summarizer
    print("Loading AI model...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Generate output content
    output_content = []

    if year_wrapped:
        # Only do Year Wrapped analysis
        wrapped_report = analyze_year_wrapped(directory_path, summarizer)
        output_content.append(wrapped_report)
    else:
        # Do regular directory analysis
        output_content = ["=== Directory Analysis Report ==="]

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

    # Ask for analysis type
    analysis_type = input(
        "Choose analysis type (1 for regular analysis, 2 for Year Wrapped): "
    )
    year_wrapped = analysis_type == "2"

    # Get output file path (optional)
    output_file = input("Enter output file path (press Enter to skip): ").strip()
    output_file = output_file if output_file else None

    analyze_directory(directory, output_file, year_wrapped)


if __name__ == "__main__":
    main()
