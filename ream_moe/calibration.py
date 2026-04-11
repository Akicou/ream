"""
Calibration data utilities for REAM/REAP MoE compression.

This module provides:
- Dataset registry with commonly used calibration datasets
- Calibration batch creation utilities
- Support for various datasets (C4, code, math, etc.)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class CalibrationBatch:
    """
    Generic container for calibration batches.

    Attributes:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class TextDataset(Dataset):
    """
    Simple in-memory text dataset for calibration.

    For production use with large datasets, consider using
    HuggingFace datasets with streaming.
    """

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class DatasetRegistry:
    """
    Registry of calibration datasets with factory functions.

    Supported datasets:
    - c4: General web text (C4 corpus)
    - code: Code corpus (using instruction tuning datasets)
    - math: Math instruction datasets
    - writing: Creative writing prompts
    - custom: User-provided text or dataset
    """

    _datasets: Dict[str, Callable[..., Iterable[str]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a dataset factory function."""

        def decorator(func: Callable[..., Iterable[str]]) -> Callable:
            cls._datasets[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[..., Iterable[str]] | None:
        """Get a dataset factory by name."""
        return cls._datasets.get(name)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return sorted(cls._datasets.keys())


def _load_dataset_with_timeout(
    dataset_path: str,
    split: str = "train",
    config: str | None = None,
    streaming: bool = True,
    max_samples: int = 1000,
) -> Iterable[str]:
    """
    Load a dataset with timeout and error handling.

    This wrapper adds resilience against network issues and memory problems
    that can occur when downloading large datasets from HuggingFace.
    """
    try:
        import datasets
    except ImportError:
        raise ImportError("datasets library not installed")

    try:
        logger.info(f"Loading {dataset_path} (config={config}, split={split}, streaming={streaming})")

        # Load with streaming to avoid downloading entire dataset
        if config:
            ds = datasets.load_dataset(
                dataset_path,
                name=config,
                split=split,
                streaming=streaming,
            )
        else:
            ds = datasets.load_dataset(
                dataset_path,
                split=split,
                streaming=streaming,
            )

        def text_generator():
            count = 0
            consecutive_errors = 0
            max_consecutive_errors = 5

            for example in ds:
                if count >= max_samples:
                    break

                try:
                    # Extract text from various possible field names
                    text = (
                        example.get("text") or
                        example.get("prompt") or
                        example.get("instruction") or
                        example.get("output") or
                        ""
                    )

                    if isinstance(text, str):
                        text = text.strip()
                        if text:  # Only yield non-empty strings
                            yield text
                            count += 1
                            consecutive_errors = 0
                        else:
                            consecutive_errors += 1
                    else:
                        consecutive_errors += 1

                except Exception as e:
                    consecutive_errors += 1
                    logger.debug(f"Error processing sample: {e}")

                # Fail after too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning(f"Too many consecutive errors ({consecutive_errors}), stopping dataset load")
                    break

            if count == 0:
                raise ValueError("No valid text samples found in dataset")

            logger.info(f"Successfully loaded {count} samples from {dataset_path}")

        return text_generator()

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_path}: {e}")
        raise


# Register built-in datasets


@DatasetRegistry.register("c4")
def _load_c4(
    samples: int = 1000,
    split: str = "train",
    streaming: bool = True,  # Unused for hardcoded, kept for compatibility
) -> Iterable[str]:
    """
    Load C4 (Colossal Clean Crawled Corpus) dataset.

    Uses hardcoded texts to avoid OOM from HuggingFace dataset loading.

    Args:
        samples: Number of samples to load
        split: Dataset split ("train", "validation") - unused for hardcoded
        streaming: Whether to use streaming mode - unused for hardcoded

    Returns:
        Iterable of text samples
    """
    return _hardcoded_c4_texts(samples)


@DatasetRegistry.register("code")
def _load_code(
    samples: int = 500,
    streaming: bool = True,  # Unused for hardcoded, kept for compatibility
) -> Iterable[str]:
    """
    Load code instruction dataset.

    Uses hardcoded code instruction texts to avoid OOM.

    Args:
        samples: Number of samples to load
        streaming: Whether to use streaming mode - unused for hardcoded

    Returns:
        Iterable of code-related text samples
    """
    return _hardcoded_code_texts(samples)


@DatasetRegistry.register("math")
def _load_math(
    samples: int = 500,
    streaming: bool = True,  # Unused for hardcoded, kept for compatibility
) -> Iterable[str]:
    """
    Load math instruction dataset.

    Uses hardcoded math instruction texts to avoid OOM.

    Args:
        samples: Number of samples to load
        streaming: Whether to use streaming mode - unused for hardcoded

    Returns:
        Iterable of math-related text samples
    """
    return _hardcoded_math_texts(samples)


@DatasetRegistry.register("writing")
def _load_writing(
    samples: int = 300,
    streaming: bool = True,  # Unused for hardcoded, kept for compatibility
) -> Iterable[str]:
    """
    Load creative writing prompts dataset.

    Uses hardcoded creative writing prompts to avoid OOM.

    Args:
        samples: Number of samples to load
        streaming: Whether to use streaming mode - unused for hardcoded

    Returns:
        Iterable of writing-related text samples
    """
    return _hardcoded_writing_texts(samples)


@DatasetRegistry.register("hardcoded")
def _load_hardcoded(
    samples: int = 2000,
    samples_per_category: int | None = None,
    streaming: bool = True,
) -> Iterable[str]:
    """
    Load comprehensive hardcoded calibration texts.

    This combines all hardcoded categories (general, code, math, writing)
    into one diverse dataset for robust MoE calibration.

    Args:
        samples: Total number of samples to load (across all categories)
        samples_per_category: Number of samples per category (overrides samples if provided)

    Returns:
        Iterable of text samples from all categories
    """
    categories = [
        ("General Knowledge", _hardcoded_c4_texts),
        ("Programming", _hardcoded_code_texts),
        ("Mathematics", _hardcoded_math_texts),
        ("Creative Writing", _hardcoded_writing_texts),
    ]

    if samples_per_category is None:
        samples_per_category = samples // len(categories)

    def text_generator():
        for category_name, category_func in categories:
            for text in category_func(samples_per_category):
                yield text

    return text_generator()


@DatasetRegistry.register("combined")
def _load_combined(
    samples: int = 1000,
    samples_per_category: int | None = None,
    streaming: bool = True,  # Unused for hardcoded, kept for compatibility
) -> Iterable[str]:
    """
    Load a combined dataset with multiple categories.

    Note: This now uses hardcoded texts to avoid OOM from HuggingFace datasets.

    Args:
        samples: Total number of samples to load (across all categories)
        samples_per_category: Number of samples per category (overrides samples if provided)
        streaming: Whether to use streaming mode - unused for hardcoded

    Returns:
        Iterable of text samples from all categories
    """
    # Just use hardcoded dataset with same parameters
    return _load_hardcoded(samples=samples, samples_per_category=samples_per_category)


def _hardcoded_c4_texts(samples: int) -> List[str]:
    """Comprehensive hardcoded texts for C4-style general calibration."""
    texts = [
        # Machine Learning & AI
        "Explain the difference between supervised and unsupervised learning.",
        "Describe how a neural network learns from training data.",
        "What are the main components of a transformer architecture?",
        "How does backpropagation work in neural networks?",
        "Explain the concept of overfitting in machine learning.",
        "What is the purpose of regularization in deep learning?",
        "Describe how gradient descent optimization works.",
        "Explain the bias-variance tradeoff in model training.",
        "What are the key differences between CNNs and RNNs?",
        "How do attention mechanisms improve model performance?",

        # Natural Language Processing
        "What is tokenization and why is it important?",
        "Explain how word embeddings capture semantic meaning.",
        "Describe the process of training a language model.",
        "What is named entity recognition used for?",
        "How does sentiment analysis determine emotional tone?",
        "Explain the challenges of machine translation.",
        "What are the benefits of pre-trained language models?",
        "How does text summarization preserve important information?",
        "Describe the difference between BERT and GPT architectures.",
        "What is the purpose of positional encoding in transformers?",

        # Programming & Software
        "Explain the concept of object-oriented programming.",
        "What are the main principles of functional programming?",
        "Describe how version control systems manage code changes.",
        "Explain the difference between SQL and NoSQL databases.",
        "What are the key features of RESTful APIs?",
        "How do design patterns improve code maintainability?",
        "Describe the software development lifecycle.",
        "What is the purpose of unit testing in software engineering?",
        "Explain the concept of algorithmic complexity.",
        "How do containerization technologies help deployment?",

        # Science & Technology
        "Explain the fundamental principles of quantum mechanics.",
        "Describe the structure and function of DNA.",
        "What are the main theories of evolution?",
        "How does the greenhouse effect work?",
        "Explain the concept of entropy in thermodynamics.",
        "What are the applications of gene editing technology?",
        "Describe the process of nuclear fusion in stars.",
        "How do renewable energy sources compare to fossil fuels?",
        "Explain the principles of vaccination and herd immunity.",
        "What is the significance of the Theory of Relativity?",

        # Mathematics
        "Explain how to solve a quadratic equation.",
        "Describe the geometric interpretation of derivatives.",
        "What are the practical applications of linear algebra?",
        "How does Bayes' theorem apply to real-world problems?",
        "Explain the concept of limits in calculus.",
        "What is the difference between permutation and combination?",
        "Describe the properties of normal distribution.",
        "How do matrices transform geometric shapes?",
        "Explain the fundamental theorem of calculus.",
        "What is the purpose of eigenvalues and eigenvectors?",

        # History & Culture
        "Describe the major causes of the Industrial Revolution.",
        "How did ancient civilizations contribute to modern society?",
        "Explain the significance of the Renaissance period.",
        "What are the main teachings of major world religions?",
        "Describe the impact of colonialism on world history.",
        "How did the printing press change society?",
        "Explain the causes and consequences of World War II.",
        "What role does archaeology play in understanding history?",
        "Describe the evolution of democracy as a political system.",
        "How has globalization affected cultural exchange?",

        # Business & Economics
        "Explain the law of supply and demand.",
        "What are the main functions of financial markets?",
        "Describe different marketing strategies for new products.",
        "How do entrepreneurs identify business opportunities?",
        "Explain the concept of economies of scale.",
        "What is the role of central banks in the economy?",
        "Describe the impact of inflation on purchasing power.",
        "How do companies use data analytics for decision making?",
        "Explain the principles of sustainable business practices.",
        "What are the challenges of international trade?",

        # General Knowledge & Society
        "How does the internet influence modern communication?",
        "Describe the effects of climate change on ecosystems.",
        "What are the main theories of cognitive psychology?",
        "Explain how social media affects public opinion.",
        "Describe the structure of the United Nations.",
        "How does the criminal justice system work?",
        "What are the ethical implications of artificial intelligence?",
        "Explain the concept of digital privacy and security.",
        "Describe the importance of education for social development.",
        "How does urbanization affect quality of life?",
    ]
    return (texts * ((samples // len(texts)) + 1))[:samples]


def _hardcoded_code_texts(samples: int) -> List[str]:
    """Comprehensive hardcoded code texts for programming calibration."""
    texts = [
        # Python Programming
        "Write a Python function to calculate the factorial of a number.",
        "Create a Python class representing a bank account with deposit and withdraw methods.",
        "Implement binary search algorithm in Python.",
        "Write a Python decorator that times function execution.",
        "Create a Python script to read and parse JSON files.",

        # Web Development
        "Create a REST API using FastAPI for managing user data.",
        "Build a Flask web server with authentication endpoints.",
        "Design a database schema for an e-commerce application.",
        "Implement user authentication using JWT tokens.",
        "Create a web scraper using Python and BeautifulSoup.",

        # Data Science & ML
        "Write a PyTorch neural network for image classification.",
        "Implement a random forest classifier using scikit-learn.",
        "Create a data pipeline for processing CSV files with pandas.",
        "Build a recommendation system using collaborative filtering.",
        "Write code to train a language model using HuggingFace transformers.",

        # Algorithms & Data Structures
        "Implement a linked list with insert and delete operations.",
        "Write a function to detect cycles in a graph.",
        "Create a binary search tree with traversal methods.",
        "Implement Dijkstra's algorithm for shortest path finding.",
        "Design a caching system using LRU eviction policy.",

        # System Programming
        "Write a multithreaded program to process files concurrently.",
        "Create a Python script using asyncio for concurrent HTTP requests.",
        "Implement a producer-consumer pattern using queues.",
        "Write a socket server that handles multiple clients.",
        "Create a memory-efficient stream processor for large files.",

        # Testing & Quality
        "Write unit tests for a calculator class using pytest.",
        "Implement integration tests for a REST API.",
        "Create a test fixture that sets up a database connection.",
        "Write a performance benchmarking script for sorting algorithms.",
        "Implement logging configuration for a production application.",

        # DevOps & Infrastructure
        "Create a Dockerfile for a Python web application.",
        "Write a CI/CD pipeline configuration using GitHub Actions.",
        "Implement infrastructure as code using Terraform.",
        "Create a Kubernetes deployment configuration.",
        "Write a bash script to automate deployment process.",

        # Security & Validation
        "Implement input validation for user registration form.",
        "Create password hashing and verification functions.",
        "Write a function to sanitize user input and prevent SQL injection.",
        "Implement rate limiting for an API endpoint.",
        "Create secure session management for a web application.",

        # File & Data Processing
        "Write a script to merge multiple CSV files into one.",
        "Implement a parallel file processor using multiprocessing.",
        "Create a data validator for JSON schema validation.",
        "Write a log parser that extracts error messages.",
        "Implement a file watcher that triggers actions on file changes.",

        # API Integration
        "Create a Python client for the OpenAI API.",
        "Implement retry logic for failed HTTP requests.",
        "Write a function to paginate through API responses.",
        "Create a webhook handler for processing external events.",
        "Implement GraphQL query builder in Python.",
    ]
    return (texts * ((samples // len(texts)) + 1))[:samples]


def _hardcoded_math_texts(samples: int) -> List[str]:
    """Comprehensive hardcoded math texts for mathematical reasoning calibration."""
    texts = [
        # Algebra
        "Solve for x: 2x + 5 = 15",
        "Factor the expression x^2 - 9",
        "Find the roots of x^2 - 5x + 6 = 0",
        "Solve the system of equations: x + y = 10, x - y = 2",
        "Simplify the expression (a + b)^2 - a^2 - b^2",
        "Find the value of x if 3^(x+1) = 27",
        "Solve for x: |2x - 3| = 7",
        "Find the inverse of the function f(x) = 2x + 5",
        "Determine if the sequence is arithmetic: 3, 7, 11, 15, 19",
        "Solve the inequality: 2x + 5 > 3x - 1",

        # Calculus
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
        "Calculate the integral of x^2 from 0 to 2",
        "Use the chain rule to differentiate f(x) = sin(2x^2)",
        "Find the limit of (sin(x))/x as x approaches 0",
        "Determine the critical points of f(x) = x^3 - 3x^2 + 2",
        "Calculate the derivative using implicit differentiation: x^2 + y^2 = 25",
        "Find the area under the curve y = x^2 from x = 0 to x = 3",
        "Use L'Hopital's rule to evaluate the limit of (e^x - 1)/x as x approaches 0",
        "Find the Taylor series expansion of e^x around x = 0",
        "Solve the differential equation: dy/dx = 2x, given y(0) = 1",

        # Geometry & Trigonometry
        "Find the area of a triangle with base 10 and height 6",
        "Calculate the volume of a sphere with radius 5",
        "Find the distance between points (3, 4) and (7, 1)",
        "Determine the value of sin(30°) + cos(60°)",
        "Find the equation of a line passing through points (1, 2) and (3, 6)",
        "Calculate the circumference of a circle with diameter 10",
        "Find the surface area of a cylinder with radius 3 and height 8",
        "Determine the angle between two lines with slopes 2 and -1/2",
        "Find the coordinates of the midpoint of the line segment from (1, 2) to (5, 8)",
        "Prove that triangles ABC and DEF are congruent given their side lengths",

        # Linear Algebra
        "Find the determinant of the 2x2 matrix [[3, 2], [1, 4]]",
        "Multiply the matrices A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]]",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]]",
        "Calculate the dot product of vectors (1, 2, 3) and (4, 5, 6)",
        "Find the inverse of the matrix [[2, 0], [0, 3]]",
        "Determine if the vectors (1, 2) and (2, 4) are linearly independent",
        "Find the rank of the matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
        "Solve the system of linear equations using matrix methods",
        "Calculate the cross product of vectors (1, 0, 0) and (0, 1, 0)",
        "Find the projection of vector v onto vector u",

        # Statistics & Probability
        "Calculate the mean of the dataset: 5, 8, 12, 15, 20",
        "Find the standard deviation of: 2, 4, 6, 8, 10",
        "Calculate the probability of rolling a sum of 7 with two dice",
        "Find the median of: 3, 7, 2, 9, 1, 5, 8",
        "Calculate the permutation P(5, 3)",
        "Find the combination C(10, 4)",
        "Calculate the conditional probability P(A|B) given P(A and B) and P(B)",
        "Find the expected value of a random variable with given probabilities",
        "Calculate the z-score for a value in a normal distribution",
        "Determine the probability of drawing 2 aces from a deck of cards",

        # Number Theory
        "Find the greatest common divisor of 48 and 36",
        "Determine if 17 is a prime number",
        "Calculate the least common multiple of 12 and 18",
        "Find the remainder when 47 is divided by 6",
        "Use Fermat's Little Theorem to solve a modular arithmetic problem",
        "Find all prime factors of 60",
        "Calculate Euler's totient function φ(15)",
        "Solve the congruence: 3x ≡ 7 (mod 11)",
        "Determine if 2^100 - 1 is divisible by 3",
        "Find the last digit of 7^100",

        # Complex Numbers
        "Add the complex numbers (3 + 2i) and (1 - 4i)",
        "Find the magnitude of the complex number 3 + 4i",
        "Calculate (2 + 3i) × (1 - 2i)",
        "Find the complex conjugate of 5 - 7i",
        "Express 2(cos(π/3) + i sin(π/3)) in rectangular form",
        "Calculate i^100",
        "Find all solutions to z^2 = -4",
        "Express e^(iπ) + 1 in simplest form",
        "Find the argument of the complex number -1 + i",
        "Divide (4 + 6i) by (2 + i)",

        # Discrete Math
        "Determine the time complexity of binary search",
        "Find the number of edges in a complete graph with 5 vertices",
        "Calculate the sum of the series 1 + 2 + 3 + ... + 100",
        "Find the 10th term of the Fibonacci sequence",
        "Determine if a graph with 4 vertices and 3 edges is a tree",
        "Calculate 7 choose 3",
        "Find the number of subsets of a set with 5 elements",
        "Solve the recurrence relation: a_n = 2a_(n-1) + 1, with a_0 = 1",
        "Use the pigeonhole principle to solve a problem",
        "Find the number of ways to arrange 5 distinct objects",
    ]
    return (texts * ((samples // len(texts)) + 1))[:samples]


def _hardcoded_writing_texts(samples: int) -> List[str]:
    """Comprehensive hardcoded writing texts for creative calibration."""
    texts = [
        # Story Writing
        "Write a story about a time traveler who accidentally changes history.",
        "Create a narrative about a character who discovers they can see the future in their dreams.",
        "Write a mystery that takes place in a small coastal town.",
        "Tell a story from the perspective of an abandoned house watching the world change.",
        "Create a tale about someone who receives letters from their future self.",

        # Descriptive Writing
        "Describe the experience of waking up in a foreign city for the first time.",
        "Paint a vivid picture of a sunset over the ocean using sensory details.",
        "Describe the atmosphere of a bustling marketplace at sunset.",
        "Write about the feeling of accomplishment after completing a long-term project.",
        "Describe the silence of a forest after a heavy snowfall.",

        # Poetry & Creative Expression
        "Compose a poem about the changing seasons from the perspective of a tree.",
        "Write a haiku about the relationship between technology and nature.",
        "Create a free verse poem about the passage of time.",
        "Write a sonnet exploring the theme of memory and forgetting.",
        "Compose a piece about the sound of rain on different surfaces.",

        # Dialogue & Character
        "Create a dialogue between two old friends who haven't seen each other in decades.",
        "Write a conversation between a mentor and their protégé at a crossroads.",
        "Create a scene where two characters from different time periods meet.",
        "Write an argument between two characters who secretly care about each other.",
        "Create a dialogue where someone tries to explain something impossible to a skeptic.",

        # Reflective & Philosophical
        "Write about someone learning a new skill and overcoming initial frustration.",
        "Explore the concept of home through the eyes of someone who has lost theirs.",
        "Write about the moment someone realizes they've been chasing the wrong goal.",
        "Describe the experience of returning to a place that has completely changed.",
        "Write about the weight of unspoken words in a relationship.",

        # Imaginative & Speculative
        "Write about a world where emotions can be traded as currency.",
        "Create a story set in a city where gravity works differently in each district.",
        "Write about someone who discovers a door that leads to a different memory each time it opens.",
        "Describe a society where people can see the literal weight of their decisions.",
        "Write about a character who can hear the thoughts of inanimate objects.",

        # Technical & Professional Writing
        "Write a product review for a device that doesn't exist yet.",
        "Create a persuasive speech about the importance of failure.",
        "Write a technical guide for operating an imaginary machine.",
        "Compose a formal letter declining an offer that doesn't exist.",
        "Write an executive summary for a project that could change the world.",

        # Personal & Emotional
        "Write about the hardest goodbye you never had to say.",
        "Describe the feeling of being exactly where you're supposed to be.",
        "Write about someone trying to remember a face they've already begun to forget.",
        "Explore the experience of standing at a crossroads with no clear path forward.",
        "Write about the moment someone realizes they've become their parents.",

        # Experimental & Avant-Garde
        "Write a story backwards, starting with the ending and ending with the beginning.",
        "Create a piece where every sentence begins with the next letter of the alphabet.",
        "Write a story where the setting is the main character.",
        "Create a narrative entirely in dialogue without any dialogue tags.",
        "Write about the space between seconds, what happens in those moments.",

        # Genre-Specific Prompts
        "Write a science fiction story about first contact that goes completely wrong.",
        "Create a fantasy scene where magic has a terrible, terrible cost.",
        "Write a romance that begins with a breakup.",
        "Describe a horror story set in broad daylight.",
        "Create a western that takes place in the future.",
    ]
    return (texts * ((samples // len(texts)) + 1))[:samples]


# Backward compatibility aliases
_fallback_texts = _hardcoded_c4_texts
_fallback_code_texts = _hardcoded_code_texts
_fallback_math_texts = _hardcoded_math_texts
_fallback_writing_texts = _hardcoded_writing_texts


def build_calibration_batches(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str] | str | List[str],
    max_seq_len: int = 512,
    batch_size: int = 4,
    samples: int = 1000,
) -> Iterable[CalibrationBatch]:
    """
    Build calibration batches from texts.

    Args:
        tokenizer: Tokenizer to use for encoding
        texts: Text samples to encode. Can be:
            - Iterable of strings (already loaded texts)
            - String name of registered dataset
            - List of strings
        max_seq_len: Maximum sequence length
        batch_size: Batch size for calibration
        samples: Number of samples to load from dataset (when using dataset name)

    Returns:
        Iterable of CalibrationBatch objects

    Examples:
        >>> # Use texts directly
        >>> batches = build_calibration_batches(tokenizer, ["text1", "text2"])
        >>> # Use registered dataset
        >>> batches = build_calibration_batches(tokenizer, "c4")
        >>> # Use with custom sample count
        >>> batches = build_calibration_batches(tokenizer, "c4", samples=500)
    """
    # Handle dataset name
    if isinstance(texts, str):
        factory = DatasetRegistry.get(texts)
        if factory is None:
            logger.warning(f"Dataset '{texts}' not found, using fallback")
            texts = _fallback_texts(samples)
        else:
            # Try to load from dataset, with fallback on failure
            try:
                texts = factory(samples=samples, streaming=True)
            except Exception as e:
                logger.warning(f"Failed to load dataset '{texts}': {e}, using fallback")
                texts = _fallback_texts(samples)

    # Convert to list if it's a generator
    if hasattr(texts, '__iter__') and not isinstance(texts, (list, str)):
        try:
            texts = list(texts)
        except Exception as e:
            logger.warning(f"Failed to iterate over texts: {e}, using fallback")
            texts = _fallback_texts(samples)

    # Ensure we have texts as a list
    if not isinstance(texts, list):
        texts = list(texts)

    # Filter out empty strings
    texts = [t for t in texts if t and t.strip()]

    if not texts:
        raise ValueError("No valid text samples found")

    dataset = TextDataset(texts)

    def collate(batch_texts: List[str]) -> CalibrationBatch:
        enc = tokenizer(
            batch_texts,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return CalibrationBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    for batch in loader:
        yield batch


def get_dataset_factory(name: str) -> Callable[..., Iterable[str]] | None:
    """
    Get a dataset factory function by name.

    Args:
        name: Dataset name (e.g., "c4", "code", "math", "writing", "combined")

    Returns:
        Dataset factory function or None if not found
    """
    return DatasetRegistry.get(name)


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return DatasetRegistry.list_datasets()
