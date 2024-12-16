import os
from pathlib import Path
import mimetypes
from collections import Counter
from transformers import pipeline  # type: ignore[import-untyped]
import torch
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence
import argparse


def get_file_type(file_path: Union[str, Path]) -> str:
    """Determine file type based on extension and mime type."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "unknown"


def analyze_file_content(file_path: Union[str, Path], summarizer: Any) -> Optional[str]:
    """Analyze text content of supported file types."""
    try:
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

            truncated_content = content[:1000]
            summary = summarizer(
                truncated_content, max_length=50, min_length=10, do_sample=False
            )
            return str(summary[0]["summary_text"])
    except Exception as e:
        return f"Error analyzing file: {str(e)}"


def generate_long_report(
    directory_path: Union[str, Path],
    total_files: int,
    total_size: int,
    file_types: Counter,
    file_summaries: List[Tuple[str, str]],
) -> str:
    """Generate a long descriptive text of the directory and its files."""
    report_lines: List[str] = []
    report_lines.append(f"Directory: {directory_path}")
    report_lines.append(f"Total number of files: {total_files}")
    report_lines.append(f"Total size (in MB): {total_size / (1024 * 1024):.2f}")
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


def process_file_stats(file_path: Path) -> Optional[Dict[str, Any]]:
    """Get basic statistics for a file."""
    try:
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
            "created_time": datetime.fromtimestamp(stat.st_ctime),
            "type": get_file_type(file_path),
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def process_directory_contents(
    directory_path: Union[str, Path], summarizer: Any, year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process directory contents and return statistics.
    If year is specified, only process files from that year.
    """
    stats: Dict[str, Any] = {
        "total_files": 0,
        "total_size": 0,
        "file_types": Counter(),
        "monthly_activity": Counter(),
        "largest_files": [],
        "file_summaries": [],
    }

    SKIP_DIRS: set[str] = {
        "node_modules",
        ".git",
        "venv",
        "env",
        "__pycache__",
        "Library",
        "Applications",
        ".npm",
        ".cache",
        "AppData",
        "Cache",
        "Caches",
    }

    SUMMARY_EXTENSIONS: set[str] = {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".html",
        ".css",
        ".json",
        ".xml",
    }

    total_files = sum(
        len(files)
        for root, dirs, files in os.walk(directory_path)
        if not any(skip_dir in root.split(os.sep) for skip_dir in SKIP_DIRS)
    )

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for root, dirs, files in os.walk(directory_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for file in files:
                file_path = Path(root) / file

                if file_path.stat().st_size > 10_000_000:
                    pbar.update(1)
                    continue

                file_stats = process_file_stats(file_path)
                pbar.update(1)

                if not file_stats:
                    continue

                if year and not (
                    file_stats["modified_time"].year == year
                    or file_stats["created_time"].year == year
                ):
                    continue

                stats["total_files"] += 1
                stats["total_size"] += file_stats["size"]
                stats["file_types"][file_stats["type"]] += 1
                stats["monthly_activity"][
                    file_stats["modified_time"].strftime("%B")
                ] += 1
                stats["largest_files"].append((file_stats["size"], file_path))

                if (
                    file_path.suffix.lower() in SUMMARY_EXTENSIONS
                    and file_stats["size"] < 1_000_000
                ):
                    summary = analyze_file_content(file_path, summarizer)
                    if summary:
                        stats["file_summaries"].append((file, summary))

    stats["largest_files"].sort(reverse=True)
    stats["largest_files"] = stats["largest_files"][:5]

    return stats


def generate_year_wrapped_report(stats: Dict[str, Any], year: int) -> str:
    """Generate a Year Wrapped style report from statistics."""
    report: List[str] = [f"\n🎉 Your {year} in Files 🎉"]

    report.append("\n📊 By the Numbers:")
    report.append(f"- You created or modified {stats['total_files']} files")
    report.append(f"- Total size: {format_size(stats['total_size'])}")

    report.append("\n📅 Your Busiest Months:")
    for month, count in stats["monthly_activity"].most_common(3):
        report.append(f"- {month}: {count} files")

    report.append("\n📁 Your Top File Types:")
    for ftype, count in stats["file_types"].most_common(5):
        report.append(f"- {ftype}: {count} files")

    report.append("\n🏋️ Your Largest Files:")
    for size, path in stats["largest_files"]:
        report.append(f"- {path.name}: {format_size(size)}")

    if stats["file_summaries"]:
        report.append("\n📝 Highlights from Your Text Files:")
        for _, summary in stats["file_summaries"][:5]:
            report.append(f"- {summary}")

    return "\n".join(report)


def generate_regular_report(
    stats: Dict[str, Any], directory_path: Union[str, Path], summarizer: Any
) -> str:
    """Generate a regular analysis report from statistics."""
    report: List[str] = ["=== Directory Analysis Report ==="]

    report.append(f"\nTotal Files: {stats['total_files']}")
    report.append(f"Total Size: {format_size(stats['total_size'])}")

    report.append("\nFile Types Distribution:")
    for file_type, count in stats["file_types"].most_common():
        report.append(f"- {file_type}: {count} files")

    if stats["file_summaries"]:
        report.append("\nFile Content Summaries:")
        for file_name, summary in stats["file_summaries"]:
            report.append(f"\n{file_name}:")
            report.append(f"Summary: {summary}")

    long_report = generate_long_report(
        directory_path,
        stats["total_files"],
        stats["total_size"],
        stats["file_types"],
        stats["file_summaries"],
    )

    report.append("\n\n=== Final Directory Summary (High-Level) ===")
    final_summary = summarizer(
        long_report, max_length=200, min_length=50, do_sample=False
    )[0]["summary_text"]
    report.append(final_summary)

    return "\n".join(report)


def analyze_directory(
    directory_path: Union[str, Path],
    output_file: Optional[str] = None,
    year_wrapped: bool = False,
) -> None:
    """Analyze a directory and its contents."""
    print("Loading AI model...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1,
    )

    year = datetime.now().year if year_wrapped else None
    stats = process_directory_contents(directory_path, summarizer, year)

    if year_wrapped:
        output_text = generate_year_wrapped_report(stats, year or datetime.now().year)
    else:
        output_text = generate_regular_report(stats, directory_path, summarizer)

    print(output_text)

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"\nOutput has been saved to: {output_file}")
        except Exception as e:
            print(f"\nError writing to output file: {str(e)}")


def autorun(
    directory: str,
    output: str,
    year_wrapped: bool = False,
) -> None:
    """
    Automated run function for CI environments.

    Args:
        directory: Path to the directory to analyze
        output: Path where to save the output
        year_wrapped: Whether to generate a year-wrapped style report
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    analyze_directory(directory, output, year_wrapped)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Directory Analysis Tool")
    parser.add_argument("--autorun", action="store_true", help="Run in autorun mode")
    parser.add_argument("--directory", type=str, help="Directory to analyze")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument(
        "--year-wrapped", action="store_true", help="Generate year-wrapped report"
    )

    args = parser.parse_args(argv)

    if args.autorun:
        directory = args.directory or str(Path.home())
        output = args.output or os.path.join(
            "out", datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        )
        os.makedirs(os.path.dirname(output), exist_ok=True)
        autorun(directory, output, args.year_wrapped)
        return

    # Original interactive code
    print("\n" + "=" * 50)
    print("📁 Directory Analysis Tool 📊")
    print("=" * 50 + "\n")

    while True:
        default_dir = str(Path.home())
        directory = input(
            f"📂 Enter the directory path to analyze (press Enter for {default_dir}): "
        ).strip()
        directory = directory if directory else default_dir

        if os.path.exists(directory):
            break
        print("❌ Error: Directory does not exist! Please try again.")

    print("\n📊 Available Analysis Types:")
    print(
        "  1. Regular Analysis  - Detailed directory statistics and content summaries"
    )
    print("  2. Year Wrapped      - Spotify-style yearly file activity overview")

    while True:
        analysis_type = input("\n🔍 Choose analysis type (1 or 2): ").strip()
        if analysis_type in ["1", "2"]:
            break
        print("❌ Please enter either 1 or 2")

    year_wrapped = analysis_type == "2"

    if year_wrapped and directory != str(Path.home()):
        print(
            f"\n💡 Tip: Year Wrapped works best with your home directory ({str(Path.home())})"
        )
        change = input(
            "Would you like to switch to analyzing your home directory? (y/n): "
        ).lower()
        if change.startswith("y"):
            directory = str(Path.home())

    os.makedirs("out", exist_ok=True)

    default_output = os.path.join(
        "out", datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    )

    print("\n💾 Output File:")
    print(f"Default: {default_output}")
    output_file = input("Enter path (or press Enter for default): ").strip()
    output_file = output_file if output_file else default_output

    print("\n" + "=" * 50)
    print("🚀 Starting Analysis...")
    print("=" * 50 + "\n")

    analyze_directory(directory, output_file, year_wrapped)


if __name__ == "__main__":
    main()
