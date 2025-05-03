# Thread Analyser

A command-line application that analyzes articles from RSS feeds using local LLM instances.

## Features

- Extract articles from RSS feeds
- Process articles using local LLM instances (Ollama or LMStudio)
- Generate summaries and analyses based on user-defined prompts
- Configurable analysis scope (timeline or number of recent articles)
- References to source articles in the output
- Support for both interactive and non-interactive modes

## Requirements

- Python 3.8+
- Required Python packages:
  - feedparser (for RSS parsing)
  - requests (for HTTP requests)
  - beautifulsoup4 (for HTML parsing)
  - langchain (for LLM integration)

## Installation

```bash
# Clone the repository
git clone https://github.com/maciejjedrzejczyk/thread-analyser.git
cd thread-analyser

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Interactive Mode

```bash
python thread_analyser.py
```

Follow the interactive prompts to:
1. Enter an RSS feed URL
2. Specify the analysis scope (number of articles or time period)
3. Enter your analysis prompt
4. Configure your local LLM instance (Ollama or LMStudio)

### Non-Interactive Mode

```bash
python thread_analyser.py --non-interactive \
  --feed-url https://example.com/feed.xml \
  --num-articles 10 \
  --prompt "Analyze mentions of AI frameworks" \
  --llm-provider ollama \
  --model llama2 \
  --output-file analysis.txt
```

#### Command-line Arguments

- `--non-interactive`: Run in non-interactive mode
- `--feed-url`: RSS feed URL to analyze
- `--num-articles`: Number of recent articles to analyze
- `--since-date`: Analyze articles since this date (YYYY-MM-DD or relative like '1 week ago')
- `--prompt`: Analysis prompt (what you want to learn from the articles)
- `--llm-provider`: LLM provider to use (ollama or lmstudio)
- `--model`: Model name to use with the LLM provider
- `--api-base`: API base URL for the LLM provider
- `--temperature`: Temperature for the LLM (0.0-1.0, default: 0.7)
- `--output-file`: Save analysis to this file

## Configuration

The application supports both Ollama and LMStudio as local LLM providers. You can configure:

- LLM provider (Ollama or LMStudio)
- Model name
- API endpoint
- Additional parameters

## License

MIT
