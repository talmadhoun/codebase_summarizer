# Codebase Summarizer

A Python tool that automatically analyzes and summarizes your codebase by leveraging OpenAI's API to generate detailed insights about your code structure, dependencies, and architecture.

## Overview

Codebase Summarizer scans your project directory, processes source files, and creates a comprehensive JSON document that provides:

- A structured file tree representation of your project
- Detailed analysis of each file, including:
  - File type and purpose
  - Dependencies and imports
  - Classes and functions
  - API endpoints
  - Design patterns
  - Integration points
  - Relationships between components

The tool is designed to be git-aware, respecting `.gitignore` rules and focusing on text-based source files while ignoring binary files.

## Features

- 📂 **Smart file discovery** - Respects `.gitignore` rules, works natively with Git repositories
- 🔄 **Batch processing** - Process large codebases by analyzing files in configurable batches
- 📊 **Efficient token usage** - Allocates token budgets proportionally based on file sizes
- 🌲 **File tree visualization** - Generates a tree-style visualization of your codebase structure
- 📝 **Detailed code analysis** - Extracts key information about classes, functions, and architecture
- 🔍 **Content truncation** - Handles large files by truncating content that exceeds limits
- ⚙️ **Configurable options** - Customize batch sizes, models, token limits, and more
- 🗜️ **Output optimization** - Optionally compress and optimize the JSON output

## Requirements

- Python 3.6+
- OpenAI API key
- Required Python packages: 
  - `openai`

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd codebase-summarizer
   ```

2. Install dependencies:
   ```
   pip install openai
   ```

3. Make the script executable (on Unix-based systems):
   ```
   chmod +x codebase_summarizer3.py
   ```

## Usage

### Basic Usage

Analyze a codebase and generate a summary:

```bash
python codebase_summarizer3.py /path/to/your/codebase
```

This will create a timestamped JSON file with the codebase analysis in the current directory.

### Setting OpenAI API Key

You can provide your OpenAI API key in three ways:

1. As a command-line argument:
   ```bash
   python codebase_summarizer3.py /path/to/your/codebase --api-key YOUR_API_KEY
   ```

2. As an environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key
   python codebase_summarizer3.py /path/to/your/codebase
   ```

3. Interactive prompt:
   If no API key is provided, the tool will prompt you to enter it.

### Custom Output Path

Specify a custom output file path:

```bash
python codebase_summarizer3.py /path/to/your/codebase --output output.json
```

### Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output`, | `-o` | Output file path | Timestamped file |
| `--model` | `-m` | OpenAI model to use for analysis | `gpt-4.1-nano-2025-04-14` |
| `--gitignore` | `-g` | Custom .gitignore file path | `.gitignore` in directory |
| `--preview` | `-p` | Preview ignored files before processing | False |
| `--api-key` | `-k` | OpenAI API key | Environment variable |
| `--verbosity` | `-v` | Verbosity level (0=quiet, 1=normal, 2=verbose) | 1 |
| `--batch-size` | `-b` | Number of files to process in each batch | 5 |
| `--use-git` | N/A | Use Git to determine which files to include | True |
| `--no-git` | N/A | Don't use Git commands even if Git is available | False |
| `--max-token-limit` | `-t` | Maximum token limit for entire output JSON | 50000 |
| `--pause-seconds` | `-ps` | Seconds to pause between batches | 0 |
| `--optimize` | `-op` | Optimize the output JSON for size and clarity | False |
| `--optimized-output` | `-oo` | Output file for the optimized JSON | `original_optimized.json` |
| `--optimization-model` | `-om` | Specific model to use for optimization | Same as analysis model |

## Examples

### Basic Analysis

```bash
python codebase_summarizer3.py /path/to/project
```

### Analysis with Custom Settings

```bash
python codebase_summarizer3.py /path/to/project \
  --output project_analysis.json \
  --model gpt-4.1-nano-2025-04-14 \
  --batch-size 10 \
  --verbosity 2 \
  --max-token-limit 80000 \
  --pause-seconds 2
```

### With Output Optimization

```bash
python codebase_summarizer3.py /path/to/project \
  --optimize \
  --optimized-output project_analysis_optimized.json
```

### With Custom .gitignore

```bash
python codebase_summarizer3.py /path/to/project \
  --gitignore /path/to/custom/.gitignore
```

## Output Format

The tool generates a JSON file with the following structure:

```json
{
  "metadata": {
    "generated_at": "2025-04-24 15:30:45",
    "total_files": 120,
    "directory": "/path/to/codebase",
    "completion_status": "completed",
    "total_codebase_size_bytes": 5430000,
    "max_token_limit": 50000
  },
  "file_tree": "codebase/\n├── src/\n│   ├── main.py\n│   └── utils.py\n└── README.md",
  "file_analyses": {
    "src/main.py": {
      "file_type": "module",
      "file_purpose": "Main entry point for the application",
      "dependencies": ["os", "sys", "utils"],
      "classes": [
        {
          "name": "App",
          "purpose": "Main application class",
          "methods": ["run", "initialize", "cleanup"]
        }
      ],
      "functions": [
        {
          "name": "main",
          "purpose": "Entry point function"
        }
      ]
    },
    // More files...
  }
}
```

## Token Usage

The tool tries to efficiently use tokens by:

1. Allocating token budgets proportionally based on file sizes
2. Truncating large files to respect rate limits
3. Optimizing JSON output when the `--optimize` flag is used

## Troubleshooting

### Error: "Not a git repository"
This warning appears when analyzing a directory that is not a Git repository. You can ignore it as the tool will automatically fall back to manual file discovery.

### API Rate Limits
If you encounter rate limit errors:
- Increase the `--pause-seconds` parameter to add more delay between batches
- Reduce the `--batch-size` parameter to process fewer files at once
- Use a different API key with higher rate limits

### JSON Parsing Errors
If the tool reports JSON parsing errors, try:
- Using the `--optimize` flag which includes JSON error correction
- Reducing the `--max-token-limit` parameter
- Using a more capable model with the `--model` parameter

## License
MIT

Created with ❤ by Thaer Almadhoun using Claude 3.7 
