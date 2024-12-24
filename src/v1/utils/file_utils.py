import os
from typing import List


def scan_directory(directory: str, file_types: List[str]) -> List[str]:
    """
    Scan directory for files with specified extensions.

    Args:
        directory: Directory path to scan
        file_types: List of file extensions to look for (without dots)

    Returns:
        List of matching file paths
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(f".{ext}") for ext in file_types):
                file_path = os.path.join(root, file)
                matching_files.append(file_path)
    return matching_files
