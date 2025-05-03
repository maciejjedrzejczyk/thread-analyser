#!/usr/bin/env python3
"""
Tests for the Thread Analyser application.
"""

import datetime
import unittest
from unittest.mock import patch, MagicMock

from thread_analyser import Article, FeedExtractor, LLMProvider, ThreadAnalyser


class TestArticle(unittest.TestCase):
    """Tests for the Article class."""
    
    def test_article_initialization(self):
        """Test Article initialization and properties."""
        now = datetime.datetime.now()
        article = Article("Test Title", "http://example.com", now, "Test content")
        
        self.assertEqual(article.title, "Test Title")
        self.assertEqual(article.link, "http://example.com")
        self.assertEqual(article.published, now)
        self.assertEqual(article.content, "Test content")
    
    def test_article_string_representation(self):
        """Test Article string representation."""
        date = datetime.datetime(2023, 1, 1)
        article = Article("Test Title", "http://example.com", date, "Test content")
        
        self.assertEqual(str(article), "Test Title (2023-01-01)")
    
    def test_article_to_dict(self):
        """Test Article to_dict method."""
        date = datetime.datetime(2023, 1, 1)
        article = Article("Test Title", "http://example.com", date, "Test content")
        
        expected = {
            "title": "Test Title",
            "link": "http://example.com",
            "published": "2023-01-01T00:00:00",
            "content": "Test content"
        }
        
        self.assertEqual(article.to_dict(), expected)


class TestFeedExtractor(unittest.TestCase):
    """Tests for the FeedExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeedExtractor()
    
    @patch('feedparser.parse')
    def test_fetch_feed_success(self, mock_parse):
        """Test successful feed fetching."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed
        
        result = self.extractor.fetch_feed("http://example.com/rss")
        
        self.assertEqual(result, mock_feed)
        mock_parse.assert_called_once()
    
    @patch('feedparser.parse')
    def test_fetch_feed_error(self, mock_parse):
        """Test feed fetching with error."""
        mock_feed = MagicMock()
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("Test exception")
        mock_parse.return_value = mock_feed
        
        with self.assertRaises(ValueError):
            self.extractor.fetch_feed("http://example.com/rss")
    
    @patch('requests.get')
    def test_extract_content(self, mock_get):
        """Test content extraction from URL."""
        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <body>
                <article>
                    <p>Test paragraph 1</p>
                    <p>Test paragraph 2</p>
                </article>
                <script>alert('test');</script>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        content = self.extractor.extract_content("http://example.com/article")
        
        self.assertIn("Test paragraph 1", content)
        self.assertIn("Test paragraph 2", content)
        self.assertNotIn("alert", content)
    
    @patch('feedparser.parse')
    def test_get_articles_with_limit(self, mock_parse):
        """Test getting articles with a limit."""
        # Create mock feed with entries
        mock_feed = MagicMock()
        mock_feed.bozo = False
        
        # Create 3 entries with different dates
        entry1 = MagicMock()
        entry1.title = "Article 1"
        entry1.link = "http://example.com/1"
        entry1.published_parsed = (2023, 1, 3, 12, 0, 0, 0, 0, 0)
        entry1.summary = "Content 1"
        
        entry2 = MagicMock()
        entry2.title = "Article 2"
        entry2.link = "http://example.com/2"
        entry2.published_parsed = (2023, 1, 2, 12, 0, 0, 0, 0, 0)
        entry2.summary = "Content 2"
        
        entry3 = MagicMock()
        entry3.title = "Article 3"
        entry3.link = "http://example.com/3"
        entry3.published_parsed = (2023, 1, 1, 12, 0, 0, 0, 0, 0)
        entry3.summary = "Content 3"
        
        mock_feed.entries = [entry1, entry2, entry3]
        mock_parse.return_value = mock_feed
        
        # Mock extract_content to avoid actual HTTP requests
        self.extractor.extract_content = MagicMock(return_value="")
        
        # Get 2 most recent articles
        articles = self.extractor.get_articles("http://example.com/rss", limit=2)
        
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0].title, "Article 1")  # Most recent first
        self.assertEqual(articles[1].title, "Article 2")
    
    @patch('feedparser.parse')
    def test_get_articles_with_date_filter(self, mock_parse):
        """Test getting articles with a date filter."""
        # Create mock feed with entries
        mock_feed = MagicMock()
        mock_feed.bozo = False
        
        # Create 3 entries with different dates
        entry1 = MagicMock()
        entry1.title = "Article 1"
        entry1.link = "http://example.com/1"
        entry1.published_parsed = (2023, 1, 3, 12, 0, 0, 0, 0, 0)
        entry1.summary = "Content 1"
        
        entry2 = MagicMock()
        entry2.title = "Article 2"
        entry2.link = "http://example.com/2"
        entry2.published_parsed = (2023, 1, 2, 12, 0, 0, 0, 0, 0)
        entry2.summary = "Content 2"
        
        entry3 = MagicMock()
        entry3.title = "Article 3"
        entry3.link = "http://example.com/3"
        entry3.published_parsed = (2023, 1, 1, 12, 0, 0, 0, 0, 0)
        entry3.summary = "Content 3"
        
        mock_feed.entries = [entry1, entry2, entry3]
        mock_parse.return_value = mock_feed
        
        # Mock extract_content to avoid actual HTTP requests
        self.extractor.extract_content = MagicMock(return_value="")
        
        # Get articles since Jan 2, 2023
        since_date = datetime.datetime(2023, 1, 2)
        articles = self.extractor.get_articles("http://example.com/rss", since_date=since_date)
        
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0].title, "Article 1")
        self.assertEqual(articles[1].title, "Article 2")


class TestLLMProvider(unittest.TestCase):
    """Tests for the LLMProvider class."""
    
    @patch('thread_analyser.Ollama')
    def test_initialize_ollama(self, mock_ollama):
        """Test initializing Ollama provider."""
        mock_ollama.return_value = MagicMock()
        
        provider = LLMProvider(
            provider="ollama",
            model="llama2",
            api_base="http://localhost:11434",
            temperature=0.7
        )
        
        self.assertEqual(provider.provider, "ollama")
        self.assertEqual(provider.model, "llama2")
        self.assertEqual(provider.api_base, "http://localhost:11434")
        mock_ollama.assert_called_once()
    
    @patch('langchain_community.llms.OpenAI')
    def test_initialize_lmstudio(self, mock_openai):
        """Test initializing LMStudio provider."""
        mock_openai.return_value = MagicMock()
        
        provider = LLMProvider(
            provider="lmstudio",
            model="default",
            api_base="http://localhost:1234/v1",
            temperature=0.7
        )
        
        self.assertEqual(provider.provider, "lmstudio")
        self.assertEqual(provider.model, "default")
        self.assertEqual(provider.api_base, "http://localhost:1234/v1")
        mock_openai.assert_called_once()
    
    def test_unsupported_provider(self):
        """Test initializing with unsupported provider."""
        with self.assertRaises(ValueError):
            LLMProvider(
                provider="unsupported",
                model="model",
                api_base="http://localhost:1234"
            )
    
    @patch('thread_analyser.LLMChain')
    def test_analyze(self, mock_chain):
        """Test article analysis."""
        # Mock LLMChain
        mock_chain_instance = MagicMock()
        mock_chain_instance.run.return_value = "Analysis result"
        mock_chain.return_value = mock_chain_instance
        
        # Mock LLM
        mock_llm = MagicMock()
        
        # Create provider with mocked LLM
        provider = LLMProvider(
            provider="ollama",
            model="llama2",
            api_base="http://localhost:11434"
        )
        provider.llm = mock_llm
        
        # Create test articles
        articles = [
            Article("Title 1", "http://example.com/1", datetime.datetime(2023, 1, 1), "Content 1"),
            Article("Title 2", "http://example.com/2", datetime.datetime(2023, 1, 2), "Content 2")
        ]
        
        # Run analysis
        result = provider.analyze(articles, "Test prompt")
        
        self.assertEqual(result, "Analysis result")
        mock_chain_instance.run.assert_called_once()


class TestThreadAnalyser(unittest.TestCase):
    """Tests for the ThreadAnalyser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyser = ThreadAnalyser()
    
    def test_parse_date_input_absolute(self):
        """Test parsing absolute date input."""
        date = self.analyser.parse_date_input("2023-01-01")
        
        self.assertEqual(date.year, 2023)
        self.assertEqual(date.month, 1)
        self.assertEqual(date.day, 1)
    
    def test_parse_date_input_relative(self):
        """Test parsing relative date input."""
        # This is a bit tricky to test precisely due to the relative nature
        # Just check that it returns a datetime object
        date = self.analyser.parse_date_input("1 week ago")
        
        self.assertIsInstance(date, datetime.datetime)
    
    def test_parse_date_input_invalid(self):
        """Test parsing invalid date input."""
        with self.assertRaises(ValueError):
            self.analyser.parse_date_input("invalid date")


if __name__ == "__main__":
    unittest.main()
