#!/usr/bin/env python3
"""
Codebase Summarizer

This script takes a folder path as input and creates a summarized representation
of the codebase by sending batches of files to OpenAI's API for summarization.
It respects .gitignore rules and works with text-based files.
"""

import os
import sys
import argparse
import fnmatch
import mimetypes
import logging
import json
from datetime import datetime, timezone, timedelta
import time
import openai

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"
DEFAULT_OUTPUT_FILE = f"processed-codebase_{datetime.now(timezone(timedelta(hours=-5))).strftime('%Y%m%d_%H%M%S')}.json"
DEFAULT_TEXT_EXTENSIONS = [
    ".py", ".js", ".ts", ".html", ".css", ".json", ".md", ".txt", 
    ".jsx", ".tsx", ".vue", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rb",
    ".php", ".swift", ".rs", ".scala", ".sql", ".xml",
]

# Batch processing settings
DEFAULT_BATCH_SIZE = 5
TRUNCATION_LIMIT = 35000  # Character limit for file content before truncation
DEFAULT_MAX_TOKEN_LIMIT = 50000  # Maximum token budget for entire JSON output
DEFAULT_PAUSE_SECONDS = 0  # Default pause between batches

# Verbosity levels
VERBOSITY_QUIET = 0    # Only errors and critical information
VERBOSITY_NORMAL = 1   # Default logging (INFO level)
VERBOSITY_VERBOSE = 2  # Detailed logging

def clean_empty_values(data):
    """
    Recursively remove empty lists, dictionaries, None values and empty strings from a dictionary.
    
    Args:
        data: Dictionary to clean
        
    Returns:
        Cleaned dictionary with empty values removed
    """
    if isinstance(data, dict):
        # Process dictionary
        return {
            k: v for k, v in 
            ((k, clean_empty_values(v)) for k, v in data.items())
            if v and v != {} and v != [] and v != ""
        }
    elif isinstance(data, list):
        # Process list
        return [v for v in (clean_empty_values(v) for v in data) if v and v != {} and v != [] and v != ""]
    else:
        # Return scalar value
        return data

def calculate_batch_size_bytes(batch_files: list) -> int:
    """
    Calculate the total size in bytes of a batch of files, safely handling any errors.
    
    Args:
        batch_files: List of file paths
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    for file_path in batch_files:
        try:
            # Normalize path separators for Windows
            normalized_path = os.path.normpath(file_path)
            if os.path.exists(normalized_path):
                total_size += os.path.getsize(normalized_path)
        except (OSError, IOError) as e:
            logger.warning(f"Could not get size of {file_path}: {str(e)}")
    return total_size

def calculate_total_codebase_size(files: list) -> int:
    """
    Calculate the total size of the codebase based on the list of files.
    
    Args:
        files: List of file paths
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    for file_path in files:
        try:
            # Normalize path separators for Windows
            normalized_path = os.path.normpath(file_path)
            if os.path.exists(normalized_path):
                total_size += os.path.getsize(normalized_path)
        except (OSError, IOError) as e:
            logger.warning(f"Could not get size of {file_path}: {str(e)}")
    return total_size

def calculate_batch_token_limit(batch_size_bytes: int, total_codebase_size: int, max_token_limit: int) -> int:
    """
    Calculate token limit for a batch based on its proportion of the total codebase size.
    
    Args:
        batch_size_bytes: Size of the current batch in bytes
        total_codebase_size: Total size of the codebase in bytes
        max_token_limit: Maximum token limit for the entire output
        
    Returns:
        Token limit for this batch
    """
    # Ensure we don't divide by zero
    if total_codebase_size <= 0:
        return max_token_limit // 10  # Default to 10% if total size unknown
    
    # Calculate proportion and corresponding token limit
    proportion = batch_size_bytes / total_codebase_size
    token_limit = int(proportion * max_token_limit)
    
    # Ensure a minimum sensible limit (at least 500 tokens)
    minimum_limit = 500
    return max(token_limit, minimum_limit)

def set_verbosity(verbosity_level: int) -> None:
    """Set logging level based on verbosity."""
    if verbosity_level == VERBOSITY_QUIET:
        logger.setLevel(logging.WARNING)
    elif verbosity_level == VERBOSITY_NORMAL:
        logger.setLevel(logging.INFO)
    elif verbosity_level == VERBOSITY_VERBOSE:
        logger.setLevel(logging.DEBUG)
    
    logger.debug(f"Verbosity level set to {verbosity_level}")

def is_text_file(file_path: str) -> bool:
    """Determine if a file is text-based using extension and MIME type."""
    # Normalize path for Windows
    normalized_path = os.path.normpath(file_path)
    
    # Check if file exists
    if not os.path.exists(normalized_path):
        logger.warning(f"File not found when checking if text file: {file_path}")
        return False
    
    # Check by extension first (faster)
    ext = os.path.splitext(normalized_path)[1].lower()
    if ext in DEFAULT_TEXT_EXTENSIONS:
        return True
    
    # Fallback to mimetype checking
    mime_type, _ = mimetypes.guess_type(normalized_path)
    if mime_type and mime_type.startswith('text/'):
        return True
    
    # Last resort: try to read as text
    try:
        with open(normalized_path, 'r', encoding='utf-8') as f:
            f.read(1024)  # Try to read a sample
        return True
    except UnicodeDecodeError:
        return False
    except Exception as e:
        logger.warning(f"Error checking if {file_path} is text: {str(e)}")
        return False

def parse_gitignore(gitignore_path: str, verbosity: int) -> list:
    """Parse a .gitignore file and return patterns."""
    patterns = []
    if not os.path.exists(gitignore_path):
        if verbosity >= VERBOSITY_NORMAL:
            logger.info(f"No .gitignore found at {gitignore_path}")
        return patterns
    
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Add the pattern
            patterns.append(line)
    
    return patterns

def is_ignored(file_path: str, root_path: str, ignore_patterns: list) -> bool:
    """Check if a file should be ignored based on gitignore patterns."""
    if not ignore_patterns:
        return False
    
    # Convert to relative path from the root
    rel_path = os.path.relpath(file_path, root_path)
    # Replace backslashes with forward slashes for consistent pattern matching
    rel_path = rel_path.replace('\\', '/')
    
    # Check each pattern
    for pattern in ignore_patterns:
        # Skip empty patterns
        if not pattern:
            continue
            
        # Handle negation (patterns that start with !)
        negated = pattern.startswith('!')
        if negated:
            pattern = pattern[1:]
        
        # Simple pattern matching using fnmatch
        match = fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path, f"*/{pattern}")
        
        if match:
            return not negated
    
    return False

def get_file_tree(directory: str, ignore_patterns: list, verbosity: int) -> str:
    """Generate a tree-style representation of the directory structure."""
    if verbosity >= VERBOSITY_VERBOSE:
        logger.info(f"Generating file tree manually for {directory}")
        
    output = []
    base_dir = os.path.basename(os.path.normpath(directory))
    
    output.append(f"{base_dir}/")
    
    for root, dirs, files in os.walk(directory):
        # Calculate the level for proper indentation
        level = root.replace(directory, '').count(os.sep)
        indent = '│   ' * level
        
        # Skip ignored directories
        if root != directory and is_ignored(root, directory, ignore_patterns):
            dirs[:] = []  # Clear dirs to prevent further traversal
            continue
        
        # Filter out ignored directories to prevent traversal
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), directory, ignore_patterns)]
        
        # Filter files that shouldn't be shown in the tree
        visible_files = [f for f in files if not is_ignored(os.path.join(root, f), directory, ignore_patterns)]
        
        # Add subdirectories to the tree output
        for i, dir_name in enumerate(sorted(dirs)):
            is_last_dir = (i == len(dirs) - 1)
            if is_last_dir and not visible_files:
                output.append(f"{indent}└── {dir_name}/")
            else:
                output.append(f"{indent}├── {dir_name}/")
        
        # Add files to the tree output
        for i, file_name in enumerate(sorted(visible_files)):
            is_last_file = (i == len(visible_files) - 1)
            if is_last_file:
                output.append(f"{indent}└── {file_name}")
            else:
                output.append(f"{indent}├── {file_name}")
    
    return '\n'.join(output)

def get_git_files(directory: str, verbosity: int) -> list:
    """
    Get a list of all files tracked by Git (respecting .gitignore rules).
    Returns an empty list if Git is not available or the directory is not a Git repository.
    """
    try:
        import subprocess
        
        # Save current directory to return to it later
        original_dir = os.getcwd()
        os.chdir(directory)
        
        # Check if this is a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            if verbosity >= VERBOSITY_NORMAL:
                logger.info("Not a git repository. Using manual file discovery.")
            os.chdir(original_dir)
            return []
        
        # Get all tracked files using git ls-files
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info("Getting tracked files from git")
        
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Convert relative paths to absolute paths and normalize them
        git_files = []
        for line in result.stdout.split('\n'):
            if line.strip():
                abs_path = os.path.join(directory, line)
                # Normalize path separators for Windows
                git_files.append(os.path.normpath(abs_path))
        
        # Return to original directory
        os.chdir(original_dir)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"Found {len(git_files)} tracked files from git")
            
        return git_files
    
    except Exception as e:
        if verbosity >= VERBOSITY_NORMAL:
            logger.info(f"Error getting git files: {str(e)}")
        
        # Return to original directory if changed
        try:
            os.chdir(original_dir)
        except:
            pass
            
        return []

def get_git_file_tree(directory: str, verbosity: int) -> str:
    """
    Generate a tree-style representation of the directory structure using Git commands.
    This ensures gitignore rules are followed exactly as Git would.
    """
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Generating file tree for {directory} using Git")
    
    try:
        import subprocess
        
        # Save current directory to return to it later
        original_dir = os.getcwd()
        os.chdir(directory)
        
        # Check if this is a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            if verbosity >= VERBOSITY_NORMAL:
                logger.info("Not a git repository. Using manual tree generation.")
            os.chdir(original_dir)
            return get_file_tree(directory, [], verbosity)
        
        # Get all tracked files using git ls-files
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True
        )
        
        git_files = [line for line in result.stdout.split('\n') if line]
        
        # Return to original directory
        os.chdir(original_dir)
        
        # Build the tree structure
        base_dir = os.path.basename(os.path.normpath(directory))
        tree = {base_dir: {}}
        
        for file_path in sorted(git_files):
            # Normalize path separators for Windows
            file_path = file_path.replace('/', os.sep)
            
            # Split into path components
            parts = file_path.split(os.sep)
            
            # Build tree structure
            current = tree[base_dir]
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # File (leaf node)
                    current[part] = None
                else:  # Directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        
        # Convert tree structure to string representation
        output = []
        output.append(f"{base_dir}/")
        
        def _build_tree_lines(node, prefix="", is_last=False, is_root=False):
            items = list(node.items())
            
            for i, (name, children) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                
                if is_root:
                    new_prefix = ""
                    tree_prefix = ""
                else:
                    tree_prefix = "└── " if is_last_item else "├── "
                    new_prefix = prefix + ("    " if is_last_item else "│   ")
                
                if children is None:  # File
                    output.append(f"{prefix}{tree_prefix}{name}")
                else:  # Directory
                    output.append(f"{prefix}{tree_prefix}{name}/")
                    _build_tree_lines(children, new_prefix, is_last_item)
        
        _build_tree_lines(tree, is_root=True)
        
        return '\n'.join(output)
        
    except Exception as e:
        logger.error(f"Error generating git file tree: {str(e)}")
        if verbosity >= VERBOSITY_NORMAL:
            logger.info("Falling back to manual tree generation")
        return get_file_tree(directory, [], verbosity)

def read_file_content(file_path: str) -> str:
    """Read and return the content of a file."""
    try:
        # Normalize path separators for Windows
        normalized_path = os.path.normpath(file_path)
        if not os.path.exists(normalized_path):
            return f"ERROR: File not found - {file_path}"
            
        with open(normalized_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return f"ERROR: Unable to read file - {str(e)}"

def fix_json_errors(json_str: str) -> str:
    """
    Attempt to fix common JSON syntax errors before parsing.
    
    Args:
        json_str: The JSON string to fix
        
    Returns:
        Fixed JSON string
    """
    # Fix trailing commas in objects (e.g., {"a": 1, "b": 2,})
    import re
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing quotes around property names
    # Note: This is a simplistic approach and might not catch all cases
    json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
    
    # Remove any single quotes used instead of double quotes for strings
    # This is risky and should only be done if we're fairly certain the JSON has this issue
    # json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
    
    return json_str

def batch_summarize_files(batch: list, api_key: str, model: str, verbosity: int, batch_token_limit: int) -> str:
    """
    Send a batch of files to OpenAI API for detailed analysis.
    
    Args:
        batch: List of tuples (file_path, file_content)
        api_key: OpenAI API key
        model: Model to use
        verbosity: Verbosity level
        batch_token_limit: Maximum tokens for the response for this batch
        
    Returns:
        JSON string with detailed analysis of all files
    """
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Analyzing batch of {len(batch)} files with token limit {batch_token_limit}")
    
    # Create batch content
    batch_content = ""
    file_paths = []
    
    for i, (file_path, content) in enumerate(batch, 1):
        rel_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(file_path)))
        file_paths.append(rel_path)
        batch_content += f"\n--- FILE {i}: {rel_path} ---\n\n"
        # Truncate very large files to avoid rate limits
        if len(content) > TRUNCATION_LIMIT:
            truncated_content = content[:TRUNCATION_LIMIT] + "\n\n... [content truncated for length] ...\n"
            batch_content += truncated_content
        else:
            batch_content += content
        batch_content += "\n\n"
    
    # Setup OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Prepare the prompt for detailed analysis
    prompt = f"""
I need you to analyze a batch of source code files and provide a detailed analysis in JSON format. Your ENTIRE response MUST stay UNDER {batch_token_limit} tokens - this is a HARD REQUIREMENT.

FOLLOW THIS EXACT RESPONSE STRUCTURE - DO NOT DEVIATE:
{{
  "files": {{
    "file_path_1": {{
      "file_type": "...",
      "file_purpose": "...",
      "dependencies": ["dep1", "dep2"],
      "classes": [
        {{
          "name": "ClassName",
          "purpose": "Brief description",
          "methods": ["method1", "method2"]
        }}
      ],
      "functions": [
        {{
          "name": "funcName",
          "purpose": "Brief description"
        }}
      ],
      "api_endpoints": [
        {{
          "path": "/path",
          "method": "GET/POST/etc"
        }}
      ],
      "design_patterns": ["Pattern description"],
      "integration_points": ["Integration description"],
      "relationships": ["Relationship description"]
    }},
    "file_path_2": {{
      ...similar structure...
    }}
  }}
}}

STRICT TOKEN LIMIT ENFORCEMENT:
- Your ENTIRE response MUST be under {batch_token_limit} tokens
- If you cannot fit complete analysis for all files within token limits:
  1. Reduce detail while maintaining coverage of all files
  2. Prioritize the most important information for each file
  3. Use concise descriptions instead of full explanations
  4. Omit less critical information when necessary
- Token limits take precedence over completeness of analysis
- Prioritize breadth (covering all files) over depth

CRITICAL JSON RULES:
1. Use DOUBLE QUOTES for all keys and string values
2. NO SEMICOLONS in JSON - ever!
3. NO TRAILING COMMAS after the last item in objects or arrays
4. Every opening {{ or [ must have a matching closing }} or ]
5. Every property name must be quoted: "name": value
6. Arrays use square brackets with comma-separated values
7. DO NOT include any field with empty values - omit them entirely
8. Commas separate items in arrays and objects, but NOT after the last item
9. NEVER use "files" as an array - it must be an object with file paths as keys

For each file, extract the following information (prioritizing the most important aspects):

* **File Type:** Categorize the file (e.g., "model", "service", "api_endpoint", "utility", etc.)
* **File Purpose:** A brief description of the file's main purpose and functionality
* **Dependencies:** List key imports and dependencies used by this file
* **Classes:** List any classes defined in the file, with details for each:
  * **Name:** Class name
  * **Purpose:** A brief description of the class
  * **Methods:** Key methods with their purpose
  * **Inheritance:** Parent classes or interfaces (if significant)
* **Functions:** List top-level functions with:
  * **Name:** Function name
  * **Purpose:** A short description of what the function does
* **API Endpoints:** If the file defines API endpoints:
  * **Path:** The route path 
  * **Method:** HTTP method (GET, POST, etc.)
* **Design Patterns:** Identify any notable design patterns used
* **Integration Points:** Note where this file interfaces with other components
* **Relationships:** How this file relates to other parts of the system

CONSISTENCY GUIDELINES:
1. Use similar detail levels across similar file types
2. Include the same property types for files of similar purposes
3. Apply consistent naming conventions in your descriptions
4. Use consistent terminology throughout the analysis

ANALYSIS PRIORITIES (focus on these when token limits are tight):
1. File purpose and type
2. Key dependencies 
3. Important classes and methods
4. Key functions
5. Integration points and relationships

REQUIRED FILES TO ANALYZE:
{batch_content}

FINAL VALIDATION STEPS (perform these before submitting):
1. Check all brackets and braces are properly matched
2. Verify no trailing commas exist in your JSON
3. Confirm all property names and string values use double quotes
4. Ensure the response is nested exactly as shown in the template
5. Verify "files" is an object, not an array
6. Confirm your response is under the token limit by removing unnecessary detail if needed

Remember: Return ONLY valid JSON - no explanation text, no markdown formatting, no code blocks.
"""
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for retry in range(max_retries):
        try:
            # Call the API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert code analyzer that always returns pure JSON with no markdown formatting or explanations. You strictly adhere to token limits and JSON syntax rules."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and return the JSON content
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error analyzing batch (attempt {retry+1}/{max_retries}): {error_message}")
            
            # If we've reached max retries, give up and return an error JSON
            if retry == max_retries - 1:
                error_json = {
                    "files": {
                        file_path: {
                            "error": f"Analysis failed after {max_retries} attempts: {error_message}",
                            "file_path": file_path
                        } for file_path in file_paths
                    }
                }
                return json.dumps(error_json)
            
            # Otherwise, wait before retrying
            logger.info(f"Waiting {retry_delay} seconds before retrying...")
            time.sleep(retry_delay)
            # Increase delay for next retry (exponential backoff)
            retry_delay *= 2

def optimize_json_output(api_key: str, model: str, input_file: str, output_file: str, 
                        optimization_model: str = None, verbosity: int = VERBOSITY_NORMAL) -> bool:
    """
    Optimize and compress a JSON file by running it through the AI again.
    
    Args:
        api_key: OpenAI API key
        model: Default model to use
        input_file: Path to the input JSON file
        output_file: Path to write the optimized JSON
        optimization_model: Specific model to use for optimization (optional)
        verbosity: Verbosity level
        
    Returns:
        True if optimization succeeded, False otherwise
    """
    # Use specified optimization model or fall back to the default model
    actual_model = optimization_model if optimization_model else model
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Optimizing JSON from {input_file} to {output_file} using model {actual_model}")
    
    try:
        # Read the original JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # Store original size for comparison
        original_json_str = json.dumps(original_data)
        original_size = len(original_json_str)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"Original JSON size: {original_size:,} bytes")
            logger.info(f"Original data has keys: {list(original_data.keys())}")
            if "file_analyses" in original_data:
                logger.info(f"file_analyses contains {len(original_data['file_analyses'])} entries")
        
        # Prepare specific structure for optimization
        prompt = f"""
I need you to optimize and compress the following JSON without losing any essential information.
You must keep the exact same structure with "metadata", "file_tree", and "file_analyses" keys.

OPTIMIZATION GUIDELINES:
1. Preserve the exact same structure with "metadata", "file_tree", and "file_analyses" keys
2. Remove any redundant or duplicated information
3. Shorten descriptions while preserving key insights
4. Make the descriptions more concise but maintain the same meaning
5. Do not remove any file paths or entries from "file_analyses"
6. Keep all metadata intact

CRITICAL JSON RULES:
1. Use DOUBLE QUOTES for all keys and string values (never single quotes)
2. NO trailing commas after the last item in objects or arrays
3. Every property name must be quoted: "name": value
4. DO NOT remove any of the top-level keys (metadata, file_tree, file_analyses)
5. The result MUST have the same structure as the input
6. Carefully check for balanced brackets and braces
7. Ensure there are no syntax errors in the resulting JSON

Here is the JSON to optimize:
{original_json_str}

Return ONLY the optimized JSON with no additional text or explanation. Ensure it's valid JSON that will parse correctly.
"""
        
        # Call the API
        client = openai.OpenAI(api_key=api_key)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info("Calling AI for JSON optimization")
            
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": "You are an expert at optimizing and compressing JSON data while preserving essential information and structure. Your output must be valid JSON with no syntax errors."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # Reduce randomness for more predictable output
        )
        
        # Extract the optimized JSON
        optimized_json_str = response.choices[0].message.content.strip()
        
        # Try to fix common JSON errors before parsing
        try:
            fixed_json_str = fix_json_errors(optimized_json_str)
            
            # Parse to validate and format
            optimized_data = json.loads(fixed_json_str)
            optimized_size = len(fixed_json_str)
            
            if verbosity >= VERBOSITY_VERBOSE:
                logger.info(f"Optimized JSON size: {optimized_size:,} bytes")
                logger.info(f"Optimized data has keys: {list(optimized_data.keys())}")
                if "file_analyses" in optimized_data:
                    logger.info(f"file_analyses contains {len(optimized_data['file_analyses'])} entries")
            
            # Verify the optimization actually reduced the size and preserved structure
            if (optimized_size >= original_size or 
                "metadata" not in optimized_data or 
                "file_tree" not in optimized_data or 
                "file_analyses" not in optimized_data):
                
                if verbosity >= VERBOSITY_NORMAL:
                    logger.info("Optimization did not reduce file size or preserve structure. Using original file.")
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(original_data, out_f, indent=2)
                return False
            
            # Write the optimized JSON to the output file
            with open(output_file, 'w', encoding='utf-8') as out_f:
                json.dump(optimized_data, out_f, indent=2)
                
            if verbosity >= VERBOSITY_NORMAL:
                reduction = (original_size - optimized_size) / original_size * 100
                logger.info(f"JSON optimization complete. Size reduction: {reduction:.2f}% ({original_size:,} → {optimized_size:,} bytes)")
                
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing optimized JSON: {str(e)}")
            
            # Save the problematic JSON for debugging if verbose
            if verbosity >= VERBOSITY_VERBOSE:
                error_file = f"{output_file}.error.json"
                with open(error_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(optimized_json_str)
                logger.info(f"Saved problematic JSON to {error_file} for debugging")
            
            logger.error("Falling back to original JSON")
            with open(output_file, 'w', encoding='utf-8') as out_f:
                json.dump(original_data, out_f, indent=2)
            return False
            
    except Exception as e:
        logger.error(f"Error during JSON optimization: {str(e)}")
        return False

def process_directory(
    directory: str, 
    output_file: str, 
    api_key: str,
    model: str,
    custom_gitignore: str = None,
    preview_ignored: bool = False,
    verbosity: int = VERBOSITY_NORMAL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_git: bool = True,
    max_token_limit: int = DEFAULT_MAX_TOKEN_LIMIT,
    pause_seconds: int = DEFAULT_PAUSE_SECONDS
) -> str:
    """
    Process a directory, analyze its files, and write the results.
    
    Returns:
        Path to the output file
    """
    # Set verbosity level
    set_verbosity(verbosity)
    
    # Get current timestamp for output file
    timestamp = datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S')
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Starting codebase analysis with verbosity level {verbosity}")
    
    # Validate and normalize input directory
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    # Handle .gitignore
    ignore_patterns = []
    gitignore_path = os.path.join(directory, '.gitignore')
    
    if os.path.exists(gitignore_path):
        ignore_patterns = parse_gitignore(gitignore_path, verbosity)
    elif custom_gitignore:
        if os.path.exists(custom_gitignore):
            ignore_patterns = parse_gitignore(custom_gitignore, verbosity)
        else:
            logger.error(f"Custom .gitignore not found: {custom_gitignore}")
            sys.exit(1)
    
    # Try to use Git if requested
    file_tree = ""
    all_files = []
    
    if use_git:
        # Try to get file tree from Git
        file_tree = get_git_file_tree(directory, verbosity)
        
        # Try to get files from Git
        git_files = get_git_files(directory, verbosity)
        if git_files:
            all_files = git_files
            if verbosity >= VERBOSITY_VERBOSE:
                logger.info(f"Using Git to identify {len(all_files)} files")
        else:
            # Fall back to manual scanning
            if verbosity >= VERBOSITY_NORMAL:
                logger.info("Falling back to manual file scanning")
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
            
            file_tree = get_file_tree(directory, ignore_patterns, verbosity)
    else:
        # Use manual file discovery
        if verbosity >= VERBOSITY_NORMAL:
            logger.info("Using manual file scanning (Git disabled)")
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        file_tree = get_file_tree(directory, ignore_patterns, verbosity)
    
    if verbosity >= VERBOSITY_VERBOSE:
        logger.info(f"Found {len(all_files)} total files")
    
    # Filter files based on ignore patterns and file type (if not using Git)
    included_files = []
    ignored_files = []
    binary_files = []
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info("Filtering files...")
    
    for file_path in all_files:
        # If we got files from Git, they're already filtered, just check file type
        if use_git and all_files and len(all_files) > 0 and all_files[0].startswith(directory):
            if is_text_file(file_path):
                included_files.append(file_path)
            else:
                binary_files.append(file_path)
        else:
            # Normal path - check both gitignore and if it's a text file
            if is_ignored(file_path, directory, ignore_patterns):
                ignored_files.append(file_path)
            elif is_text_file(file_path):
                included_files.append(file_path)
            else:
                binary_files.append(file_path)
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Files summary: {len(included_files)} included, {len(ignored_files)} ignored, {len(binary_files)} binary")
    
    # Preview ignored files if requested
    if preview_ignored and (ignored_files or binary_files):
        logger.info("The following files will be ignored:")
        
        for file in ignored_files:
            rel_path = os.path.relpath(file, directory)
            logger.info(f"  - {rel_path} (matched ignore pattern)")
        
        for file in binary_files:
            rel_path = os.path.relpath(file, directory)
            logger.info(f"  - {rel_path} (binary file)")
        
        confirm = input("Continue with these exclusions? (y/n): ").lower()
        if confirm != 'y':
            logger.info("Operation cancelled by user")
            sys.exit(0)
    
    # Calculate total codebase size for token allocation
    total_codebase_size = calculate_total_codebase_size(included_files)
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Total codebase size: {total_codebase_size:,} bytes")
    
    # Calculate total_files before using it
    total_files = len(included_files)
    
    # Create the initial output structure
    output_data = {
        "metadata": {
            "generated_at": timestamp,
            "total_files": total_files,
            "directory": directory,
            "completion_status": "in_progress",
            "total_codebase_size_bytes": total_codebase_size,
            "max_token_limit": max_token_limit
        },
        "file_tree": file_tree,
        "file_analyses": {}
    }
    
    # Write the initial JSON output with empty analyses
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(output_data, out_f, indent=2)
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Created initial output file: {output_file}")
        logger.info(f"Beginning analysis of {total_files} files in batches of {batch_size}...")
    
    # Process files in batches
    for i in range(0, len(included_files), batch_size):
        batch = included_files[i:i + batch_size]
        
        if verbosity >= VERBOSITY_NORMAL:
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} ({len(batch)} files)")
        
        # Calculate batch size in bytes
        batch_size_bytes = calculate_batch_size_bytes(batch)
        
        # Calculate token limit for this batch based on proportion of codebase
        batch_token_limit = calculate_batch_token_limit(batch_size_bytes, total_codebase_size, max_token_limit)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"Batch size: {batch_size_bytes:,} bytes ({batch_size_bytes/total_codebase_size:.2%} of codebase)")
            logger.info(f"Allocated token limit for batch: {batch_token_limit}")
        
        # Load content for each file in the batch
        batch_with_content = []
        for file_path in batch:
            content = read_file_content(file_path)
            batch_with_content.append((file_path, content))
        
        # Analyze the batch
        batch_analysis = batch_summarize_files(batch_with_content, api_key, model, verbosity, batch_token_limit)
        
        # Read the current state of the file
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error reading existing output file: {str(e)}")
            # Initialize with empty data if file is corrupted or missing
            output_data = {
                "metadata": {
                    "generated_at": timestamp,
                    "total_files": total_files,
                    "directory": directory,
                    "completion_status": "in_progress",
                    "total_codebase_size_bytes": total_codebase_size,
                    "max_token_limit": max_token_limit
                },
                "file_tree": file_tree,
                "file_analyses": {}
            }
        
        # Update the analyses with new batch data
        try:
            # Parse the JSON response
            batch_data = json.loads(batch_analysis)
            
            # Check for the expected structure
            if "files" in batch_data:
                # Process each file analysis to remove empty values
                for file_path, analysis in batch_data["files"].items():
                    # Recursively remove empty lists, dictionaries, None values, or empty strings
                    cleaned_analysis = clean_empty_values(analysis)
                    if cleaned_analysis:  # Only add if there's content
                        output_data["file_analyses"][file_path] = cleaned_analysis
            else:
                logger.error("Unexpected JSON structure: 'files' key not found")
                # Try to salvage what we can from the response
                for file_path in batch:
                    rel_path = os.path.relpath(file_path, directory)
                    output_data["file_analyses"][rel_path] = {
                        "error": "Failed to parse analysis",
                        "raw_response": batch_analysis
                    }
            
            # Update progress information
            output_data["metadata"]["completed_batches"] = i//batch_size + 1
            output_data["metadata"]["total_batches"] = (total_files + batch_size - 1)//batch_size
            output_data["metadata"]["files_analyzed"] = len(output_data["file_analyses"])
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            # If we can't parse JSON, just save the raw response
            for file_path in batch:
                rel_path = os.path.relpath(file_path, directory)
                output_data["file_analyses"][rel_path] = {
                    "error": "Failed to parse analysis",
                    "raw_response": batch_analysis
                }
        
        # Write the updated data back to the file
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(output_data, out_f, indent=2)
        
        if verbosity >= VERBOSITY_NORMAL:
            logger.info(f"Completed batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            logger.info(f"Updated output file with {len(output_data['file_analyses'])} file analyses")
        
        # Add a pause between batches to avoid rate limits
        if i + batch_size < len(included_files):
            if verbosity >= VERBOSITY_NORMAL:
                logger.info(f"Pausing for {pause_seconds} seconds before next batch...")
            time.sleep(pause_seconds)
    
    # Update the final status
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        output_data["metadata"]["completion_status"] = "completed"
        output_data["metadata"]["completion_time"] = datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S')
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(output_data, out_f, indent=2)
    except Exception as e:
        logger.error(f"Error updating final status: {str(e)}")
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Processed {len(included_files)} files")
        logger.info(f"Output written to: {output_file}")
    
    return output_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a codebase - analyze its files and create a structured JSON document")
    parser.add_argument("directory", help="Directory containing the codebase to process")
    parser.add_argument("--output", "-o", 
                       default=DEFAULT_OUTPUT_FILE, 
                       help=f"Output file path (default: timestamped file)")
    parser.add_argument("--model", "-m", 
                       default=DEFAULT_MODEL,
                       help=f"OpenAI model to use for analysis (default: {DEFAULT_MODEL})")
    parser.add_argument("--gitignore", "-g",
                       help="Custom .gitignore file path to use instead of looking in the directory")
    parser.add_argument("--preview", "-p", 
                       action="store_true",
                       help="Preview ignored files before processing")
    parser.add_argument("--api-key", "-k",
                       help="OpenAI API key (will use OPENAI_API_KEY environment variable if not provided)")
    parser.add_argument("--verbosity", "-v", type=int, default=VERBOSITY_NORMAL, choices=range(3),
                       help="Verbosity level: 0=quiet, 1=normal, 2=verbose (default: 1)")
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Number of files to process in each batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--use-git", action="store_true", default=True,
                       help="Use Git to determine which files to include (respects .gitignore perfectly)")
    parser.add_argument("--no-git", action="store_false", dest="use_git",
                       help="Don't use Git commands even if Git is available")
    parser.add_argument("--max-token-limit", "-t", type=int, default=DEFAULT_MAX_TOKEN_LIMIT,
                       help=f"Maximum token limit for entire output JSON (default: {DEFAULT_MAX_TOKEN_LIMIT})")
    parser.add_argument("--pause-seconds", "-ps", type=int, default=DEFAULT_PAUSE_SECONDS,
                       help=f"Seconds to pause between batches (default: {DEFAULT_PAUSE_SECONDS})")
    parser.add_argument("--optimize", "-op", action="store_true",
                       help="Optimize the output JSON for size and clarity")
    parser.add_argument("--optimized-output", "-oo", 
                       help="Output file for the optimized JSON (default: 'optimized_' + original filename)")
    parser.add_argument("--optimization-model", "-om",
                       help="Specific model to use for optimization (default: same as analysis model)")
    
    args = parser.parse_args()
    
    # Get OpenAI API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            logger.error("No OpenAI API key provided")
            sys.exit(1)
    
    # Process the directory
    output_file = process_directory(
        directory=args.directory,
        output_file=args.output,
        api_key=api_key,
        model=args.model,
        custom_gitignore=args.gitignore,
        preview_ignored=args.preview,
        verbosity=args.verbosity,
        batch_size=args.batch_size,
        use_git=args.use_git,
        max_token_limit=args.max_token_limit,
        pause_seconds=args.pause_seconds
    )
    
    # Optimize JSON if requested
    if args.optimize:
        optimized_output = args.optimized_output
        if not optimized_output:
            # Generate default optimized filename
            base_name, ext = os.path.splitext(args.output)
            optimized_output = f"{base_name}_optimized{ext}"
        
        logger.info(f"Starting JSON optimization...")
        optimize_json_output(
            api_key=api_key,
            model=args.model,
            input_file=args.output,
            output_file=optimized_output,
            optimization_model=args.optimization_model,
            verbosity=args.verbosity
        )
        logger.info(f"Optimized output written to: {optimized_output}")

if __name__ == "__main__":
    main()