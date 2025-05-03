# Thread Analyser - Implementation Documentation

## Overview

Thread Analyser is a command-line application that analyzes articles from RSS feeds using local LLM instances. The application allows users to extract content from RSS feeds, process it using local LLM instances (Ollama or LMStudio), and generate analyses based on user-defined prompts.

## Implementation Details

### Core Components

1. **Article Class**
   - Represents an article extracted from an RSS feed
   - Stores title, link, publication date, and content
   - Provides serialization methods

2. **FeedExtractor Class**
   - Fetches and parses RSS feeds
   - Extracts article content from web pages
   - Filters articles based on date or count
   - Handles pagination for feeds with article limits

3. **LLMProvider Class**
   - Provides a unified interface for different LLM backends
   - Supports Ollama and LMStudio
   - Handles prompt construction and LLM interaction

4. **ThreadAnalyser Class**
   - Main application class
   - Handles user input and orchestrates the analysis process
   - Manages the workflow from feed extraction to LLM analysis

### Key Features

1. **RSS Feed Processing**
   - Extracts articles from RSS feeds
   - Handles various RSS formats and structures
   - Cleans and normalizes content
   - Supports pagination to overcome feed limits

2. **Content Extraction**
   - Extracts main article content from web pages
   - Removes non-content elements (scripts, styles, etc.)
   - Handles different article structures

3. **Flexible Analysis Scope**
   - Supports analysis by number of recent articles
   - Supports analysis by time period (articles since a specific date)
   - Handles relative date expressions (e.g., "1 week ago")

4. **LLM Integration**
   - Supports multiple local LLM providers (Ollama, LMStudio)
   - Configurable model selection and parameters
   - Streaming output for real-time feedback

5. **User Experience**
   - Interactive command-line interface
   - Non-interactive mode for automation and scripting
   - Clear prompts and error handling
   - Option to save analysis results to file

## Design Decisions

1. **Modular Architecture**
   - Separation of concerns between feed extraction, content processing, and LLM interaction
   - Easy to extend with additional LLM providers or content sources

2. **Content Extraction Strategy**
   - Uses a combination of RSS feed content and web scraping
   - Falls back to web scraping if RSS content is insufficient
   - Handles HTML parsing and cleaning

3. **LLM Provider Abstraction**
   - Common interface for different LLM backends
   - Simplifies adding support for new LLM providers
   - Configurable parameters for fine-tuning

4. **Error Handling**
   - Robust error handling for network issues, parsing errors, and LLM failures
   - User-friendly error messages
   - Graceful degradation when parts of the process fail

5. **Dual Mode Operation**
   - Interactive mode for user-guided analysis
   - Non-interactive mode for automation and scripting
   - Consistent behavior between both modes

## Testing

The application includes comprehensive unit tests covering:
- Article class functionality
- Feed extraction and parsing
- Content extraction from web pages
- LLM provider initialization and interaction
- Date parsing and filtering
- User input handling

## Future Enhancements

1. **Additional LLM Providers**
   - Support for more local LLM providers
   - Cloud LLM integration options

2. **Advanced Content Extraction**
   - Improved algorithms for extracting relevant content
   - Support for paywalled content

3. **Batch Processing**
   - Process multiple feeds in one run
   - Save/load feed configurations

4. **Output Formats**
   - Support for different output formats (Markdown, HTML, etc.)
   - Export options for sharing analyses

5. **Persistent Configuration**
   - Save and reuse LLM configurations
   - Favorite feeds and prompts
