# Services module initialization
# Import summarization service functions for easy access

from .summarization_service import (
    summarize_text_with_detail_level,
    summarize_text_with_word_count,
    detect_language_lingua,
    process_markdown_string,
    SummarizerWithDetailLevel,
    SummarizationLevelEnum,
)

__all__ = [
    "summarize_text_with_detail_level",
    "summarize_text_with_word_count", 
    "detect_language_lingua",
    "process_markdown_string",
    "SummarizerWithDetailLevel",
    "SummarizationLevelEnum"
]
