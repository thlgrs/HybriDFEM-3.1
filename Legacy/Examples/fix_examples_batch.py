#!/usr/bin/env python3
"""
Batch fix Legacy examples: Fix imports and output paths
This script updates all example files to use proper relative imports and output directories
"""

import os
import re
from typing import List, Tuple

# Standard header comment to add
HEADER_COMMENT = """# ============================================================================
# FIXED: Removed hard-coded paths - use relative imports from Legacy package
# Original code (kept for reference):
# folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
# sys.path.append(str(folder))
# ============================================================================
"""

# Import mapping: old import -> new import
IMPORT_MAP = {
    r'^import Structure as st$': 'from Legacy.Objects import Structure as st',
    r'^import ConstitutiveLaw as mat$': 'from Legacy.Objects import ConstitutiveLaw as mat',
    r'^import Surface as surf$': 'from Legacy.Objects import Surface as surf',
    r'^import Contact as ct$': 'from Legacy.Objects import Contact as ct',
    r'^import ContactPair as cp$': 'from Legacy.Objects import ContactPair as cp',
}

# Setup code for output directory
OUTPUT_DIR_SETUP = """
# Set up output directory
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
if not os.path.exists(save_path):
    os.makedirs(save_path)
"""


def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in directory tree"""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip __pycache__ and out directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', 'out', '.git']]
        for file in files:
            if file.endswith('.py') and file != 'fix_examples_batch.py':
                python_files.append(os.path.join(root, file))
    return python_files


def has_hardcoded_path(content: str) -> bool:
    """Check if file contains hard-coded path"""
    patterns = [
        r"pathlib\.Path\('C:/Users/",
        r'pathlib\.Path\("C:/Users/',
        r"sys\.path\.append",
    ]
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    return False


def fix_file_content(content: str, filepath: str) -> Tuple[str, bool]:
    """Fix imports and paths in file content. Returns (new_content, was_modified)"""
    original_content = content
    lines = content.split('\n')
    new_lines = []
    i = 0
    modified = False

    # Track if we've already added the header
    header_added = False

    while i < len(lines):
        line = lines[i]

        # Skip if already fixed
        if '# FIXED: Removed hard-coded paths' in line:
            new_lines.append(line)
            i += 1
            continue

        # Replace hard-coded path block
        if "pathlib.Path('C:/Users/" in line or 'pathlib.Path("C:/Users/' in line:
            if not header_added:
                new_lines.append('')
                new_lines.extend(HEADER_COMMENT.split('\n'))
                header_added = True
                modified = True
            # Skip the current line and the sys.path.append line
            i += 1
            if i < len(lines) and 'sys.path.append' in lines[i]:
                i += 1
            continue

        # Replace sys.path.append if not already caught
        if 'sys.path.append' in line and 'HybriDFEM' in line:
            # Already handled by previous block or standalone
            if not header_added:
                new_lines.append('')
                new_lines.extend(HEADER_COMMENT.split('\n'))
                header_added = True
                modified = True
            i += 1
            continue

        # Replace old-style imports
        stripped = line.strip()
        replaced = False
        for old_pattern, new_import in IMPORT_MAP.items():
            if re.match(old_pattern, stripped):
                new_lines.append(new_import)
                modified = True
                replaced = True
                break

        if replaced:
            i += 1
            continue

        # Fix reload_modules() call to be optional
        if 'reload_modules()' == stripped and not line.strip().startswith('#'):
            new_lines.append('# reload_modules()  # Uncomment if needed during development')
            modified = True
            i += 1
            continue

        # Update save_path if it's the old style
        if 'save_path = os.path.dirname(os.path.abspath(__file__))' in line:
            new_lines.extend(OUTPUT_DIR_SETUP.strip().split('\n'))
            modified = True
            i += 1
            continue

        if 'save_path = os.getcwd()' in line:
            new_lines.extend(OUTPUT_DIR_SETUP.strip().split('\n'))
            modified = True
            i += 1
            continue

        # Keep the line as-is
        new_lines.append(line)
        i += 1

    new_content = '\n'.join(new_lines)

    # Additional fixes: update file paths to use save_path/out directory
    # Fix .save_structure(filename='...') to use os.path.join
    new_content = re.sub(
        r"\.save_structure\(filename='([^']+)'\)",
        r".save_structure(filename=os.path.join(save_path, '\1'))",
        new_content
    )
    new_content = re.sub(
        r'\.save_structure\(filename="([^"]+)"\)',
        r'.save_structure(filename=os.path.join(save_path, "\1"))',
        new_content
    )

    # Fix St.save_structure calls without filename parameter
    new_content = re.sub(
        r"St\.save_structure\(f'([^']+)'\)",
        r"St.save_structure(os.path.join(save_path, f'\1'))",
        new_content
    )

    if new_content != original_content:
        modified = True

    return new_content, modified


def process_file(filepath: str, dry_run: bool = False) -> bool:
    """Process a single file. Returns True if modified."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()

    # Check if file needs fixing
    if not has_hardcoded_path(content):
        # Still check if imports need fixing
        needs_import_fix = False
        for old_pattern in IMPORT_MAP.keys():
            if re.search(old_pattern, content, re.MULTILINE):
                needs_import_fix = True
                break

        if not needs_import_fix:
            return False

    # Fix the file
    new_content, modified = fix_file_content(content, filepath)

    if modified:
        if dry_run:
            print(f"Would modify: {filepath}")
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed: {filepath}")
        return True

    return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Fix Legacy examples batch')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    parser.add_argument('--directory', default=None, help='Specific directory to process (default: all)')
    args = parser.parse_args()

    # Get the Examples directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = script_dir

    if args.directory:
        examples_dir = os.path.join(script_dir, args.directory)
        if not os.path.exists(examples_dir):
            print(f"Error: Directory not found: {examples_dir}")
            return 1

    # Find all Python files
    python_files = find_python_files(examples_dir)
    print(f"Found {len(python_files)} Python files to process")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===\n")

    # Process each file
    modified_count = 0
    for filepath in sorted(python_files):
        if process_file(filepath, dry_run=args.dry_run):
            modified_count += 1

    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified_count}/{len(python_files)} files")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")

    return 0


if __name__ == '__main__':
    exit(main())
