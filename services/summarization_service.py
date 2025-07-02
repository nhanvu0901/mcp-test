import asyncio
import json
import os
from enum import Enum
from typing import Literal

import regex as re
import semchunk
import tiktoken
from lingua import Language, LanguageDetectorBuilder
from semchunk import Chunker

# Import LLM client from dependencies
from .utils import get_llm_client

# Language detection setup
detector = (
    LanguageDetectorBuilder.from_all_languages()
    .with_preloaded_language_models()
    .build()
)


# Token encoding setup
encoding = tiktoken.encoding_for_model("gpt-4o-mini")


class SummarizationLevelEnum(Enum):
    CONCISE = "concise"
    MEDIUM = "medium"
    DETAILED = "detailed"


SupportedLanguage = Literal["English", "Czech", "Slovakia"]


# PROMPTS
CHUNK_PROMPT_N = """As a professional summarizer, generate a structured JSON output for the provided text chunk. Your output must follow these guidelines:

1. Title: Create a short, clear, and compelling title (within 5-10 words) that captures the core idea or theme of the text chunk. The title should provide a concise overview of the summary.
2. Summary:
   - Create a coherent list of {bullet_num} bullet points that accurately capture the essential points of the text chunk.
   - Ensure that each bullet point is well-structured and reflects the most important information while removing any unnecessary details.
   - Ensure that each bullet point averages about 15-20 words. This balance maintains brevity while providing sufficient detail for clarity and comprehension.
   - Ensure that each bullet point is a complete, meaningful sentence.
3. Content Integrity: Ensure the summary strictly reflects the content of the chunk without introducing external information.
4. Consistent Style: Ensure that the tone, style, and formatting are consistent across the entire summary. Present a unified voice and approach throughout the document.
5. JSON Structure: Return the result as a JSON object with two fields:
   - `title`: A string containing the title.
   - `summary`: A list of strings, each representing a bullet point.

The text chunk:
{chunk}
"""

REFINE_PROMPT_N = """Divide into paragraphs. Return the output under the JSON key `summary`.

The summary:

{merged_summaries}

Further instruction: {further_instruction}
"""

CHUNK_SYS_PROMPT = "You are ChatGPT, a helpful AI that excels at summarization and text manipulation. You always return a valid JSON object with only two keys `title` and `summary`. Unless otherwise instructed, always return the output in the original language of the document."

SYS_PROMPT = "You are ChatGPT, a helpful AI that excels at summarization and text manipulation. You always return a valid JSON object with only one key `summary`. Unless otherwise instructed, always return the output in the original language of the document."

SYS_PROMPT_ADD = " Always return the output in {lang}."


async def process_markdown_string(md_string: str) -> str:
    """Process Markdown string."""
    # Remove Markdown style
    md_string = re.sub(r"\*\*(.*?)\*\*|\*(.*?)\*", r"\1\2", md_string)
    # Replace \n\n-----\n\n with \n\n
    md_string = re.sub(r"\n\n-----\n\n", "\n", md_string)
    # Using \p{Ll} for lowercase Latin and \p{M} for diacritical marks
    return re.sub(r"(?<!\.)\n\n(?=[\p{Ll}\p{M}])", "\n", md_string)


def detect_language_lingua(
    text: str,
) -> SupportedLanguage:
    try:
        language = detector.detect_language_of(text)
        print(f"DEBUG: Detected language object: {language}")
        print(f"DEBUG: Language type: {type(language)}")
        
        if language is None:
            print("DEBUG: Language detection returned None, defaulting to English")
            return "English"

        if language not in (Language.ENGLISH, Language.CZECH, Language.SLOVAK):
            print(f"DEBUG: Unsupported language detected: {language}, defaulting to English")
            return "English"

        # Map language objects to the expected string values
        if language == Language.ENGLISH:
            return "English"
        elif language == Language.CZECH:
            return "Czech"
        elif language == Language.SLOVAK:
            return "Slovakia"
        else:
            # Fallback to English if somehow we get here
            print(f"DEBUG: Unexpected language case: {language}, defaulting to English")
            return "English"
    except Exception as e:
        print(f"DEBUG: Exception in language detection: {e}")
        return "English"


def get_chunker(max_token_chars: int = 1024) -> Chunker:
    chunker = semchunk.chunkerify("gpt-4o", max_token_chars)
    return chunker


def chunk_text(text: str, max_token_chars: int = 1024) -> list[str] | list[list[str]]:
    chunker = get_chunker(max_token_chars)
    chunks = chunker(text)
    return chunks


def embed_len(text: str) -> int:
    return len(encoding.encode(text))


def count_word(text: str) -> int:
    """
    Counts the number of words in the given text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of words in the text.
    """
    words = text.split()
    return len(words)


class SummarizerWithDetailLevel:
    """
    A class that provides summarization functionality with detail level control.
    """

    def __init__(self):
        self.llm_client = get_llm_client()
        self._PROMPT_MAPPING = {
            "chunk_prompt": CHUNK_PROMPT_N,
            "refine_merged_summ_prompt": REFINE_PROMPT_N,
            "sys_prompt": SYS_PROMPT,
            "sys_prompt_add": SYS_PROMPT_ADD,
            "chunk_sys_prompt": CHUNK_SYS_PROMPT,
        }

    async def _acomplete(self, messages, **kwargs):
        response = await self.llm_client.acomplete(messages, **kwargs)
        return json.loads(response.choices[0].message.content)

    BULLET_NUM_MAPPING = {
        512: {
            SummarizationLevelEnum.CONCISE: 3,
            SummarizationLevelEnum.MEDIUM: 4,
            SummarizationLevelEnum.DETAILED: 5,
        },
        1024: {
            SummarizationLevelEnum.CONCISE: 4,
            SummarizationLevelEnum.MEDIUM: 6,
            SummarizationLevelEnum.DETAILED: 8,
        },
        2048: {
            SummarizationLevelEnum.CONCISE: 4,
            SummarizationLevelEnum.MEDIUM: 8,
            SummarizationLevelEnum.DETAILED: 12,
        },
    }

    async def _summarize_text(
        self,
        text: str,
        summarization_level: SummarizationLevelEnum,
        lang: SupportedLanguage | None,
        further_instruction: str,
    ) -> tuple[str, int]:
        # Merge all chunk summaries into a single string
        merged_summary = await self._summarize_multiple_chunks(
            text, summarization_level, lang
        )

        chunks = chunk_text(merged_summary, 400)

        merged_summary = "\n\n".join(chunks)

        merged_summary = await self._refine_merged_summaries(
            merged_summary, lang, further_instruction
        )

        return merged_summary, count_word(merged_summary)

    async def _summarize_multiple_chunks(
        self,
        text: str,
        summarization_level: SummarizationLevelEnum,
        lang: SupportedLanguage | None,
    ) -> str:
        if embed_len(text) < 2048:
            max_token_per_chunk = 512
        elif embed_len(text) < 4096:
            max_token_per_chunk = 1024
        else:
            max_token_per_chunk = 2048
        bullet_num = self.BULLET_NUM_MAPPING[max_token_per_chunk][summarization_level]

        chunks = chunk_text(text, max_token_per_chunk)
        tasks = [self._summarize_chunk(chunk, bullet_num, lang) for chunk in chunks]

        # Execute all tasks concurrently and gather results in order
        results: list[tuple[str, list[str]]] = await asyncio.gather(*tasks)

        # Collect summaries maintaining order
        chunk_summaries = []
        for _, chunk_summary in results:
            # Assuming chunk_summary is a list of bullet points
            bullet_points = "\n".join(chunk_summary)
            chunk_summaries.append(f"{bullet_points}\n")

        # Merge all chunk summaries into a single string
        merged_summaries = "".join(chunk_summaries)

        return merged_summaries

    async def _summarize_chunk(
        self, chunk: str, bullet_num: int, lang: SupportedLanguage | None
    ) -> tuple[str, list[str]]:
        if lang:
            system_prompt = self._PROMPT_MAPPING[
                "chunk_sys_prompt"
            ] + self._PROMPT_MAPPING["sys_prompt_add"].format(lang=lang)
        else:
            system_prompt = self._PROMPT_MAPPING["chunk_sys_prompt"]
        user_message = self._PROMPT_MAPPING["chunk_prompt"].format(
            chunk=chunk, bullet_num=bullet_num
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        content = await self._acomplete(
            messages,
            response_format={"type": "json_object"},
        )
        title, summary = content["title"], content["summary"]
        return title, summary

    async def _refine_merged_summaries(
        self, merged_summaries: str, lang: SupportedLanguage, further_instruction: str
    ) -> str:
        if lang:
            system_prompt = self._PROMPT_MAPPING["sys_prompt"] + self._PROMPT_MAPPING[
                "sys_prompt_add"
            ].format(lang=lang)
        else:
            system_prompt = self._PROMPT_MAPPING["sys_prompt"]
        user_message = self._PROMPT_MAPPING["refine_merged_summ_prompt"].format(
            merged_summaries=merged_summaries,
            further_instruction=further_instruction,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        content = await self._acomplete(
            messages,
            response_format={"type": "json_object"},
        )
        return content["summary"]

    async def summarize(
        self,
        text: str,
        summarization_level: SummarizationLevelEnum,
        lang: SupportedLanguage | None,
        further_instruction: str | None,
    ) -> tuple[str, int]:
        """
        Summarizes the given text based on the specified summarization level and optional further instructions.

        Args:
            text (str): The input text to be summarized.
            summarization_level (SummarizationLevelEnum): The level of detail for the summary.
            lang (SupportedLanguage | None): The language of the text.
            further_instruction (str | None): Additional instructions to customize the summarization process.

        Returns:
            tuple[str, int]: A tuple containing the summary text and the word count of the summary.
        """
        if further_instruction is None:
            further_instruction = ""

        return await self._summarize_text(
            text, summarization_level, lang, further_instruction
        )


# Initialize the summarizer instance
summarizer = SummarizerWithDetailLevel()


async def summarize_text_with_detail_level(
    text: str,
    summarization_level: str = "medium",
    further_instruction: str | None = None,
) -> tuple[str, int]:
    """
    Summarizes the given text based on the specified summarization level and optional further instructions.

    Args:
        text (str): The input text to be summarized.
        summarization_level (str, optional): The level of detail for the summary. Defaults to "medium".
        further_instruction (str, optional): Additional instructions to customize the summarization process. Defaults to None.

    Returns:
        tuple[str, int]: A tuple containing the summary text and the word count of the summary.
    """
    # Process the text for markdown and specific document formats
    text = await process_markdown_string(text)

    try:
        summ_level_enum = SummarizationLevelEnum[summarization_level.upper()]
    except KeyError:
        summ_level_enum = SummarizationLevelEnum.MEDIUM

    text_sample = text[: min(len(text), 1000)]
    try:
        lang = detect_language_lingua(text_sample)
    except ValueError:
        lang = None

    summary, word_count = await summarizer.summarize(
        text=text,
        summarization_level=summ_level_enum,
        lang=lang,
        further_instruction=further_instruction,
    )

    return summary, word_count


# Simple word count summarization (basic implementation)
async def summarize_text_with_word_count(
    text: str, num_words: int = 100
) -> tuple[str, int]:
    """
    Summarizes the given text based on the specified word count.

    Args:
        text (str): The input text to be summarized.
        num_words (int): The target number of words for the summary.

    Returns:
        tuple[str, int]: A tuple containing the summary text and the word count of the summary.
    """
    # Process the text for markdown
    text = await process_markdown_string(text)
    
    # Use detail level summarization with appropriate level based on word count
    if num_words <= 50:
        level = "concise"
    elif num_words <= 150:
        level = "medium"
    else:
        level = "detailed"
    
    # Create instruction for word count
    instruction = f"Create a summary with approximately {num_words} words."
    
    summary, word_count = await summarize_text_with_detail_level(
        text, level, instruction
    )
    
    return summary, word_count 