#!/usr/bin/env python3
"""
Thread Analyser - A tool to analyze articles from RSS feeds using local LLM instances.
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import re
import time

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class Article:
    """Represents an article extracted from an RSS feed."""
    
    def __init__(self, title: str, link: str, published: datetime.datetime, content: str):
        self.title = title
        self.link = link
        self.published = published
        self.content = content
        
    def __str__(self) -> str:
        return f"{self.title} ({self.published.strftime('%Y-%m-%d')})"
    
    def to_dict(self) -> Dict:
        """Convert article to dictionary for serialization."""
        return {
            "title": self.title,
            "link": self.link,
            "published": self.published.isoformat(),
            "content": self.content
        }


class FeedExtractor:
    """Extracts and processes articles from RSS feeds."""
    
    def __init__(self):
        self.user_agent = "ThreadAnalyser/1.0"
        
    def fetch_feed(self, url: str, page: int = 1) -> feedparser.FeedParserDict:
        """
        Fetch and parse an RSS feed.
        
        Args:
            url: URL of the RSS feed
            page: Page number for paginated feeds (if supported)
            
        Returns:
            Parsed feed
        """
        try:
            # Try to handle pagination if supported by the feed
            paginated_url = url
            if page > 1:
                # Add pagination parameters for common RSS formats
                # This won't work for all feeds, but it's a best effort
                if "?" in url:
                    paginated_url = f"{url}&page={page}"
                else:
                    paginated_url = f"{url}?page={page}"
            
            feed = feedparser.parse(paginated_url, agent=self.user_agent)
            if feed.bozo and hasattr(feed, 'bozo_exception'):
                raise ValueError(f"Invalid RSS feed: {feed.bozo_exception}")
            return feed
        except Exception as e:
            raise ValueError(f"Error fetching feed: {str(e)}")
    
    def extract_content(self, url: str) -> str:
        """Extract the main content from an article URL."""
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script, style, and other non-content elements
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.decompose()
            
            # Extract article content - this is a simple approach and might need refinement
            # for specific websites
            article_content = ""
            
            # Try to find article content in common containers
            article_tags = soup.select("article, .article, .post, .content, main")
            if article_tags:
                article_content = article_tags[0].get_text(separator="\n", strip=True)
            else:
                # Fallback to paragraphs
                paragraphs = soup.find_all("p")
                article_content = "\n".join([p.get_text(strip=True) for p in paragraphs])
            
            # Clean up the content
            article_content = re.sub(r'\n+', '\n', article_content)
            article_content = re.sub(r'\s+', ' ', article_content)
            
            return article_content.strip()
        except Exception as e:
            print(f"Warning: Could not extract content from {url}: {str(e)}")
            return ""
    
    def get_articles(self, feed_url: str, limit: Optional[int] = None, 
                    since_date: Optional[datetime.datetime] = None) -> List[Article]:
        """
        Extract articles from a feed with optional filtering.
        
        Args:
            feed_url: URL of the RSS feed
            limit: Maximum number of recent articles to return
            since_date: Only return articles published after this date
            
        Returns:
            List of Article objects
        """
        articles = []
        page = 1
        max_pages = 5  # Limit the number of pages to try
        target_reached = False
        
        while page <= max_pages and not target_reached:
            try:
                feed = self.fetch_feed(feed_url, page)
                
                if not feed.entries:
                    # No more entries, stop fetching
                    break
                
                for entry in feed.entries:
                    # Extract publication date
                    published = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = datetime.datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published = datetime.datetime(*entry.updated_parsed[:6])
                    else:
                        # Use current time as fallback
                        published = datetime.datetime.now()
                    
                    # Apply date filter if specified
                    if since_date and published < since_date:
                        # If we've reached articles older than our date filter, we can stop
                        target_reached = True
                        break
                    
                    # Extract title and link
                    title = entry.title if hasattr(entry, 'title') else "No title"
                    link = entry.link if hasattr(entry, 'link') else ""
                    
                    # Extract content
                    content = ""
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].value
                    elif hasattr(entry, 'summary') and entry.summary:
                        content = entry.summary
                    
                    # If content is HTML, parse it
                    if content and ("<" in content and ">" in content):
                        soup = BeautifulSoup(content, "html.parser")
                        content = soup.get_text(separator="\n", strip=True)
                    
                    # If content is still empty or too short, try to extract from the article URL
                    if not content or len(content) < 200:
                        if link:
                            content = self.extract_content(link)
                    
                    articles.append(Article(title, link, published, content))
                    
                    # Check if we've reached the limit
                    if limit and len(articles) >= limit:
                        target_reached = True
                        break
                
                # If this page had fewer entries than expected, assume no more pages
                if len(feed.entries) < 10:  # Most feeds return at least 10 per page
                    break
                    
                page += 1
                
            except Exception as e:
                # If we encounter an error after the first page, we can still return what we have
                if page > 1:
                    print(f"Warning: Error fetching page {page}: {str(e)}")
                    print(f"Proceeding with {len(articles)} articles fetched so far.")
                    break
                else:
                    # If the first page fails, re-raise the exception
                    raise
        
        # Sort by publication date (newest first)
        articles.sort(key=lambda x: x.published, reverse=True)
        
        # Apply limit if specified and not already applied during fetching
        if limit and limit > 0 and len(articles) > limit:
            articles = articles[:limit]
            
        return articles


class LLMProvider:
    """Interface for local LLM providers."""
    
    def __init__(self, provider: str, model: str, api_base: str, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.api_base = api_base
        self.kwargs = kwargs
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on the provider."""
        if self.provider == "ollama":
            return Ollama(
                model=self.model,
                base_url=self.api_base,
                callbacks=[StreamingStdOutCallbackHandler()],
                **self.kwargs
            )
        elif self.provider == "lmstudio":
            # LMStudio uses the OpenAI-compatible API
            from langchain_community.llms import OpenAI
            return OpenAI(
                model=self.model,
                openai_api_base=self.api_base,
                openai_api_key="lm-studio",  # Placeholder key
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def analyze(self, articles: List[Article], prompt: str) -> str:
        """
        Analyze articles using the configured LLM.
        
        Args:
            articles: List of articles to analyze
            prompt: User-provided analysis prompt
            
        Returns:
            Analysis result from the LLM
        """
        # Prepare the context with article information
        context = []
        for i, article in enumerate(articles, 1):
            # Truncate content if too long to avoid context window issues
            content = article.content
            if len(content) > 2000:
                content = content[:2000] + "... [content truncated]"
                
            context.append(
                f"ARTICLE {i}:\n"
                f"Title: {article.title}\n"
                f"Published: {article.published.strftime('%Y-%m-%d')}\n"
                f"URL: {article.link}\n"
                f"Content: {content}\n"
            )
        
        articles_context = "\n\n".join(context)
        
        # Create a prompt template with stronger focus on the user's specific request
        template = """
You are an expert content analyst with a focused approach. Your task is to analyze the provided articles to find EXACT matches or highly relevant information related to the following specific request:

{prompt}

Here are the articles to analyze:

{articles}

CRITICAL INSTRUCTIONS:
1. Your PRIMARY goal is to identify which articles (if any) contain information MOST RELEVANT to the user's request.
2. For each relevant article, provide:
   - The article number
   - A brief explanation of WHY it's relevant (citing specific terms, concepts, or technologies mentioned)
   - Direct quotes from the article that demonstrate relevance
3. ONLY include articles that have a STRONG connection to the request - do not stretch for tenuous connections.
4. If NO articles contain information clearly relevant to the request, state this explicitly: "None of the provided articles contain information clearly relevant to [specific request]."
5. Rank articles by relevance if multiple are found.
6. DO NOT provide general summaries of articles or technologies unless they directly relate to the request.
7. Include a "Most Relevant Articles" section at the beginning that lists article numbers in order of relevance.
8. Include a "References" section at the end listing ONLY the articles you referenced.

Your analysis must be evidence-based and focused on helping the user quickly identify which articles (if any) are worth reading based on their specific interest.

Analysis:
"""
        
        prompt_template = PromptTemplate(
            input_variables=["prompt", "articles"],
            template=template
        )
        
        # Create and run the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(prompt=prompt, articles=articles_context)
            return result
        except Exception as e:
            return f"Error during analysis: {str(e)}"


class ThreadAnalyser:
    """Main application class for Thread Analyser."""
    
    def __init__(self):
        self.feed_extractor = FeedExtractor()
    
    def parse_date_input(self, date_str: str) -> datetime.datetime:
        """Parse a user-provided date string."""
        try:
            # Try standard date parsing first
            return date_parser.parse(date_str)
        except Exception:
            # Handle relative dates
            now = datetime.datetime.now()
            
            # Simple relative date handling
            if "ago" in date_str.lower():
                parts = date_str.lower().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    num = int(parts[0])
                    unit = parts[1]
                    
                    if unit in ["day", "days"]:
                        return now - datetime.timedelta(days=num)
                    elif unit in ["week", "weeks"]:
                        return now - datetime.timedelta(days=num*7)
                    elif unit in ["month", "months"]:
                        return now - relativedelta(months=num)
                    elif unit in ["year", "years"]:
                        return now - relativedelta(years=num)
            
            raise ValueError(f"Invalid date format: {date_str}")
    
    def get_user_input(self) -> Tuple[str, Union[int, datetime.datetime], str, Dict]:
        """Get user input for the analysis."""
        print("\n===== Thread Analyser =====\n")
        
        # Get RSS feed URL
        feed_url = input("Enter RSS feed URL: ").strip()
        if not feed_url:
            raise ValueError("RSS feed URL is required")
        
        # Get analysis scope
        print("\nAnalysis scope:")
        print("1. Number of recent articles")
        print("2. Time period (articles since a specific date)")
        scope_choice = input("Choose an option (1/2): ").strip()
        
        scope = None
        if scope_choice == "1":
            try:
                num_articles = int(input("Enter number of recent articles to analyze: ").strip())
                if num_articles <= 0:
                    raise ValueError("Number of articles must be positive")
                scope = num_articles
            except ValueError as e:
                raise ValueError(f"Invalid number: {str(e)}")
        elif scope_choice == "2":
            date_str = input("Enter start date (YYYY-MM-DD or relative like '1 week ago'): ").strip()
            try:
                scope = self.parse_date_input(date_str)
            except ValueError as e:
                raise ValueError(f"Invalid date: {str(e)}")
        else:
            raise ValueError("Invalid choice. Please select 1 or 2.")
        
        # Get analysis prompt
        print("\nEnter your analysis prompt (what you want to learn from these articles).")
        print("Be specific about technologies, frameworks, or concepts you're interested in.")
        print("Examples:")
        print("- \"Find articles discussing LangChain framework or similar RAG tools\"")
        print("- \"Which articles mention serverless architectures for ML workloads?\"")
        print("- \"Identify articles about generative AI for document processing\"")
        prompt = input("Prompt: ").strip()
        if not prompt:
            raise ValueError("Analysis prompt is required")
        
        # Configure LLM
        print("\nLLM Configuration:")
        print("1. Ollama")
        print("2. LMStudio")
        llm_choice = input("Choose an LLM provider (1/2): ").strip()
        
        llm_config = {}
        if llm_choice == "1":
            llm_config["provider"] = "ollama"
            llm_config["model"] = input("Enter model name (e.g., llama2): ").strip() or "llama2"
            llm_config["api_base"] = input("Enter API base URL (default: http://localhost:11434): ").strip() or "http://localhost:11434"
        elif llm_choice == "2":
            llm_config["provider"] = "lmstudio"
            llm_config["model"] = input("Enter model name: ").strip() or "default"
            llm_config["api_base"] = input("Enter API base URL (default: http://localhost:1234/v1): ").strip() or "http://localhost:1234/v1"
        else:
            raise ValueError("Invalid choice. Please select 1 or 2.")
        
        # Additional parameters
        temp = input("Enter temperature (0.0-1.0, default: 0.7): ").strip()
        if temp:
            try:
                llm_config["temperature"] = float(temp)
            except ValueError:
                print("Invalid temperature, using default 0.7")
                llm_config["temperature"] = 0.7
        else:
            llm_config["temperature"] = 0.7
        
        return feed_url, scope, prompt, llm_config
    
    def run(self):
        """Run the Thread Analyser application."""
        try:
            # Get user input
            feed_url, scope, prompt, llm_config = self.get_user_input()
            
            print("\nFetching and processing articles...")
            
            # Extract articles based on scope
            if isinstance(scope, int):
                articles = self.feed_extractor.get_articles(feed_url, limit=scope)
                
                # Check if we got fewer articles than requested
                if len(articles) < scope:
                    print(f"\nWarning: Requested {scope} articles but only found {len(articles)}.")
                    print("This may be due to RSS feed limitations or pagination restrictions.")
                
                scope_desc = f"{len(articles)} recent articles"
            else:
                articles = self.feed_extractor.get_articles(feed_url, since_date=scope)
                scope_desc = f"articles since {scope.strftime('%Y-%m-%d')}"
            
            if not articles:
                print("No articles found matching the criteria.")
                return
            
            print(f"Found {len(articles)} articles to analyze.")
            
            # Initialize LLM provider
            llm_provider = LLMProvider(**llm_config)
            
            print(f"\nAnalyzing {scope_desc} with prompt: '{prompt}'")
            print("This may take some time depending on the number of articles and LLM speed...\n")
            
            # Run analysis
            result = llm_provider.analyze(articles, prompt)
            
            print("\n===== Analysis Complete =====\n")
            print(result)
            
            # Offer to save results
            save = input("\nSave analysis to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Analysis of {scope_desc} from {feed_url}\n")
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(result)
                print(f"Analysis saved to {filename}")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Thread Analyser - Analyze articles from RSS feeds using local LLM instances")
    parser.add_argument("--version", action="version", version="Thread Analyser 1.0")
    
    # Add non-interactive mode arguments
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode using command line arguments")
    parser.add_argument("--feed-url", help="RSS feed URL to analyze")
    parser.add_argument("--num-articles", type=int, help="Number of recent articles to analyze")
    parser.add_argument("--since-date", help="Analyze articles since this date (YYYY-MM-DD or relative like '1 week ago')")
    parser.add_argument("--prompt", help="Analysis prompt (what you want to learn from the articles)")
    parser.add_argument("--llm-provider", choices=["ollama", "lmstudio"], help="LLM provider to use (ollama or lmstudio)")
    parser.add_argument("--model", help="Model name to use with the LLM provider")
    parser.add_argument("--api-base", help="API base URL for the LLM provider")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the LLM (0.0-1.0, default: 0.7)")
    parser.add_argument("--output-file", help="Save analysis to this file (if not specified, will prompt user)")
    
    args = parser.parse_args()
    
    if args.non_interactive:
        # Validate required arguments for non-interactive mode
        if not args.feed_url:
            print("Error: --feed-url is required in non-interactive mode")
            sys.exit(1)
        if not args.prompt:
            print("Error: --prompt is required in non-interactive mode")
            sys.exit(1)
        if not args.llm_provider:
            print("Error: --llm-provider is required in non-interactive mode")
            sys.exit(1)
        if not (args.num_articles or args.since_date):
            print("Error: Either --num-articles or --since-date is required in non-interactive mode")
            sys.exit(1)
        if args.num_articles and args.since_date:
            print("Error: Cannot specify both --num-articles and --since-date")
            sys.exit(1)
            
        # Run in non-interactive mode
        run_non_interactive(args)
    else:
        # Run in interactive mode
        ThreadAnalyser().run()


def run_non_interactive(args):
    """Run the Thread Analyser in non-interactive mode."""
    try:
        analyser = ThreadAnalyser()
        feed_url = args.feed_url
        
        # Determine scope
        if args.num_articles:
            scope = args.num_articles
            scope_desc = f"{args.num_articles} recent articles"
        else:
            try:
                scope = analyser.parse_date_input(args.since_date)
                scope_desc = f"articles since {scope.strftime('%Y-%m-%d')}"
            except ValueError as e:
                print(f"Error parsing date: {str(e)}")
                sys.exit(1)
        
        prompt = args.prompt
        
        # Configure LLM
        llm_config = {
            "provider": args.llm_provider,
            "model": args.model or ("llama2" if args.llm_provider == "ollama" else "default"),
            "api_base": args.api_base or 
                       ("http://localhost:11434" if args.llm_provider == "ollama" else "http://localhost:1234/v1"),
            "temperature": args.temperature
        }
        
        print(f"\nFetching and processing articles from {feed_url}...")
        
        # Extract articles based on scope
        if isinstance(scope, int):
            articles = analyser.feed_extractor.get_articles(feed_url, limit=scope)
            
            # Check if we got fewer articles than requested
            if len(articles) < scope:
                print(f"\nWarning: Requested {scope} articles but only found {len(articles)}.")
                print("This may be due to RSS feed limitations or pagination restrictions.")
            
            scope_desc = f"{len(articles)} recent articles"
        else:
            articles = analyser.feed_extractor.get_articles(feed_url, since_date=scope)
            scope_desc = f"articles since {scope.strftime('%Y-%m-%d')}"
        
        if not articles:
            print("No articles found matching the criteria.")
            return
        
        print(f"Found {len(articles)} articles to analyze.")
        
        # Initialize LLM provider
        llm_provider = LLMProvider(**llm_config)
        
        print(f"\nAnalyzing {scope_desc} with prompt: '{prompt}'")
        print("This may take some time depending on the number of articles and LLM speed...\n")
        
        # Run analysis
        result = llm_provider.analyze(articles, prompt)
        
        print("\n===== Analysis Complete =====\n")
        print(result)
        
        # Save results if output file is specified
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(f"Analysis of {scope_desc} from {feed_url}\n")
                f.write(f"Prompt: {prompt}\n\n")
                f.write(result)
            print(f"\nAnalysis saved to {args.output_file}")
        else:
            # Offer to save results
            save = input("\nSave analysis to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Analysis of {scope_desc} from {feed_url}\n")
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(result)
                print(f"Analysis saved to {filename}")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
