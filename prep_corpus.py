"""
Prepare the Gouda 2.0 pretraining corpus.

The corpus is assembled from multiple Hugging Face datasets according to
token-level target proportions.

Output:
    data/gouda-2.0/
        train_00000.npy
        train_00001.npy
        ...
        metadata.json

Each .npy shard:
    - contains np.uint16 token IDs
    - is approximately SHARD_SIZE tokens
    - contains documents from all corpus sources
    - separates documents with the tokenizer's EOS token
    - does not intentionally split documents across shard boundaries

Pretraining document format:

    [document tokens] [EOS]
    [document tokens] [EOS]
    ...
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(ROOT, "data", "gouda-2.0")
DOWNLOAD_DIR = os.path.join(ROOT, "data", "_downloads")

TOKENIZER = "alphaedge-ai/Qwen3-1.7B-eng-32768"

TOTAL_TOKENS = 10e9
# Documents should not normally be split merely to hit this exactly.
SHARD_SIZE = 100e6
# Number of documents passed to the tokenizer at once.
TOKENIZE_BATCH_SIZE = 2_000
# Ignore extremely short documents.
MIN_DOCUMENT_TOKENS = 16
# Reproducible file/source ordering.
SEED = 1234
# Number of concurrent downloads.
DOWNLOAD_WORKERS = 2

SOURCES = {
    "dclm-edu": dict(repo="HuggingFaceTB/dclm-edu", subset=None, kind="text", col="text", target=3.0e9,),
    "fineweb-edu": dict(repo="HuggingFaceTB/smollm-corpus", subset="fineweb-edu-dedup", kind="text", col="text", target=2.0e9,),
    "finepdfs": dict(repo="HuggingFaceFW/finepdfs-edu", subset="eng_Latn", kind="text", col="text", target=1.25e9,),
    "fineweb": dict(repo="HuggingFaceFW/fineweb_100BT-shuffled", subset=None, kind="text", col="text", target=1.0e9,),
    "finemath": dict(repo="HuggingFaceTB/finemath", subset="finemath-4plus", kind="text", col="text", target=1.0e9,),
    "wikipedia": dict(repo="wikimedia/wikipedia", subset="20231101.en", kind="text", col="text", target=1.0e9,),
    "cosmopedia": dict(repo="HuggingFaceTB/smollm-corpus", subset="cosmopedia-v2", kind="text", col="text", target=0.75e9,),
}


# =============================================================================
# Source state
# =============================================================================

@dataclass
class SourceState:
    """
    Runtime state for one corpus source.

    You may add/remove fields later if your implementation makes another
    representation cleaner.
    """

    name: str
    config: dict

    target_tokens: int
    tokens_written: int = 0

    files: list[str] | None = None
    file_index: int = 0

    document_iterator: Iterator[str] | None = None

    finished: bool = False

    @property
    def remaining_tokens(self) -> int:
        """
        Return the number of tokens this source still needs to contribute.

        Requirements:
        - Never return a negative number.
        - target_tokens and tokens_written are measured in tokens.

        Returns:
            int
        """
        pass


# =============================================================================
# Tokenizer
# =============================================================================

def load_tokenizer():
    """
    Load and validate the tokenizer used by Gouda 2.0.

    Requirements:
    - Load TOKENIZER using Hugging Face AutoTokenizer.
    - Determine EOS from the tokenizer itself.
    - Do NOT hard-code the EOS ID.
    - Verify that eos_token_id exists.
    - Verify that every token ID the tokenizer can produce fits inside
      np.uint16.
    - Be careful that added tokens may exist beyond tokenizer.vocab_size.
    - Do not apply the chat template here.
    - Print useful tokenizer information:
        tokenizer name
        effective vocabulary size / maximum token ID
        EOS token
        EOS token ID

    Returns:
        The loaded tokenizer.
    """
    pass


# =============================================================================
# Corpus validation
# =============================================================================

def validate_sources(sources: dict) -> None:
    """
    Validate the corpus configuration before downloading anything.

    Requirements:
    - Every source must have:
        repo
        kind
        col
        target
    - Every source in this pipeline should currently be kind="text".
    - Every target must be positive.
    - Calculate the sum of all source targets.
    - Verify that it equals TOTAL_TOKENS.
    - Raise useful errors rather than silently continuing with a bad config.

    Returns:
        None
    """
    pass


# =============================================================================
# Hugging Face file discovery
# =============================================================================

def list_dataset_files(source: dict) -> list[str]:
    """
    Find the parquet files belonging to one dataset/subset.

    Requirements:
    - Query the Hugging Face dataset repository.
    - Find relevant .parquet files.
    - If the source has a subset, restrict files to that subset.
    - Ignore unrelated files such as README/config metadata.
    - Sort before shuffling so behaviour does not depend on API ordering.
    - Shuffle deterministically using SEED.
    - Include sensible retry handling for transient API failures.
    - Do not download any files here.

    Important:
    Different Hugging Face repositories may organise subsets differently.
    Your implementation needs to determine how the configured subset maps
    onto repository paths.

    Returns:
        list[str]:
            Remote parquet filenames in deterministic randomized order.
    """
    pass


# =============================================================================
# Downloading
# =============================================================================

def download_file(source: dict, filename: str) -> str:
    """
    Download one parquet file.

    Requirements:
    - Download from source["repo"].
    - Treat the repository as a Hugging Face dataset.
    - Store temporary downloads beneath DOWNLOAD_DIR.
    - Retry transient failures.
    - Use increasing waits between retries.
    - Raise an error after the retry limit rather than silently skipping data.

    Returns:
        str:
            Local filesystem path to the downloaded parquet file.
    """
    pass


def delete_download(path: str) -> None:
    """
    Delete a downloaded parquet file after it has been consumed.

    Requirements:
    - Remove the local temporary file.
    - Do not crash if it has already disappeared.
    - Never delete anything outside the download/cache area.

    Returns:
        None
    """
    pass


# =============================================================================
# Reading documents
# =============================================================================

def iter_documents(source: dict, path: str) -> Iterator[str]:
    """
    Stream raw documents from one parquet file.

    Requirements:
    - Open the file with pyarrow.parquet.ParquetFile.
    - Process one row group at a time.
    - Read only the required text column if practical.
    - Obtain the column name from source["col"].
    - Ignore null/empty documents.
    - Avoid loading the complete parquet file into RAM.
    - Yield raw strings.
    - Do not tokenize here.
    - Do not add EOS here.

    Yields:
        str:
            One raw pretraining document at a time.
    """
    pass


# =============================================================================
# Tokenization
# =============================================================================

def tokenize_batch(
    texts: list[str],
    tokenizer,
) -> list[np.ndarray]:
    """
    Tokenize a batch of raw pretraining documents.

    Requirements:
    - Tokenize the whole batch efficiently.
    - Use add_special_tokens=False.
    - Do NOT apply the chat template.
    - Do NOT add User:/Assistant: prefixes.
    - Reject documents containing fewer than MIN_DOCUMENT_TOKENS tokens.
    - Append exactly ONE tokenizer.eos_token_id to every accepted document.
    - Convert each accepted document to np.uint16.
    - Preserve document boundaries: one ndarray per document.

    Example conceptual result:

        [
            [41, 928, 17, ..., EOS],
            [581, 92, 772, ..., EOS],
            ...
        ]

    Returns:
        list[np.ndarray]:
            Tokenized documents including their terminating EOS token.
    """
    pass


def iter_tokenized_documents(
    documents: Iterator[str],
    tokenizer,
) -> Iterator[np.ndarray]:
    """
    Convert a raw document iterator into a tokenized document iterator.

    Requirements:
    - Accumulate raw documents into batches of TOKENIZE_BATCH_SIZE.
    - Pass full batches to tokenize_batch().
    - Yield each resulting tokenized document individually.
    - Handle the final partial batch.
    - Keep memory usage bounded.
    - Do not accidentally drop the final batch.

    Yields:
        np.ndarray:
            One uint16 tokenized document ending in EOS.
    """
    pass


# =============================================================================
# Source streaming
# =============================================================================

def create_source_states(sources: dict) -> dict[str, SourceState]:
    """
    Construct the runtime state for every corpus source.

    Requirements:
    - Create one SourceState per SOURCES entry.
    - Convert configured targets to integers.
    - Discover each source's parquet files using list_dataset_files().
    - Set initial counters correctly.
    - Detect sources for which no usable files were found.

    Returns:
        dict[str, SourceState]:
            Mapping from source name to runtime state.
    """
    pass


def open_next_file(
    state: SourceState,
    tokenizer,
    download_pool: ThreadPoolExecutor,
) -> bool:
    """
    Advance a source to its next available parquet file.

    Requirements:
    - Determine the next unconsumed filename.
    - Download it.
    - Create:
        iter_documents(...)
            ->
        iter_tokenized_documents(...)
    - Store that iterator on the SourceState.
    - Advance file_index appropriately.
    - Eventually detect when the source has no files remaining.
    - Mark the source finished when appropriate.

    Optional performance challenge:
    - Arrange things so the next parquet file can be downloading while the
      current parquet file is being tokenized.

    Returns:
        bool:
            True if a new file was successfully opened.
            False if the source has no files remaining.
    """
    pass


def next_document(
    state: SourceState,
    tokenizer,
    download_pool: ThreadPoolExecutor,
) -> np.ndarray | None:
    """
    Obtain the next tokenized document from a source.

    Requirements:
    - Read from state.document_iterator if one exists.
    - If that iterator is exhausted:
        * clean up its downloaded parquet file
        * open the next file
        * continue
    - Keep doing this until either:
        * a tokenized document is available, or
        * the source has no data remaining.
    - Do not load multiple parquet files into RAM unnecessarily.

    Returns:
        np.ndarray | None:
            Next uint16 document including EOS,
            or None when the source is exhausted.
    """
    pass


# =============================================================================
# Mixture scheduling
# =============================================================================

def active_sources(
    states: dict[str, SourceState],
) -> list[SourceState]:
    """
    Return sources which can still contribute to the corpus.

    Requirements:
    - Exclude sources that are finished.
    - Exclude sources whose token target has already been reached.
    - Preserve enough information for choose_source() to calculate weights.

    Returns:
        list[SourceState]
    """
    pass


def choose_source(
    states: dict[str, SourceState],
    rng: random.Random,
) -> SourceState:
    """
    Choose which source should supply the next document.

    Requirements:
    - Consider only active sources.
    - Weight each source according to its REMAINING token budget.

      Conceptually:

          weight_i = remaining_tokens_i

          P(source_i) =
              remaining_tokens_i / sum(all remaining tokens)

    - Use the supplied seeded RNG for reproducibility.
    - Do not create a gigantic list containing one entry per token.
    - Raise a useful error if no sources remain.

    Returns:
        SourceState:
            The selected source.
    """
    pass


# =============================================================================
# Shard construction
# =============================================================================

def should_accept_document(
    state: SourceState,
    document: np.ndarray,
) -> bool:
    """
    Decide whether a document should still be accepted from this source.

    The source target is a token budget, but documents should normally remain
    intact.

    Requirements:
    - Compare len(document) with state.remaining_tokens.
    - Decide how much target overshoot is acceptable in order to preserve
      complete documents.
    - Prevent pathological final documents from causing enormous overshoots.
    - Keep the policy deterministic and consistent across sources.

    This function deliberately leaves the exact boundary policy for you to
    choose.

    Returns:
        bool
    """
    pass


def write_shard(
    documents: list[np.ndarray],
    shard_index: int,
    output_dir: str,
) -> dict:
    """
    Write one completed training shard.

    Requirements:
    - Concatenate the supplied tokenized documents.
    - Verify the final dtype is np.uint16.
    - Save using np.save(), producing a genuine .npy file.
    - Use a filename such as:

          train_00000.npy
          train_00001.npy

    - Avoid leaving a corrupted final shard if the process dies halfway
      through writing.
    - Record useful information about the resulting shard.

    Returns:
        dict:
            Metadata describing the shard, for example:
                filename
                number of tokens
                number of documents
    """
    pass


def build_corpus(
    states: dict[str, SourceState],
    tokenizer,
    output_dir: str,
) -> dict:
    """
    Build the complete mixed Gouda 2.0 pretraining corpus.

    This is the main orchestration exercise.

    Requirements:

    1. Create a reproducible RNG using SEED.

    2. Maintain a current shard containing tokenized documents.

    3. Repeatedly:
        - choose a source using choose_source()
        - request its next tokenized document
        - handle exhausted sources
        - decide whether the document should be accepted
        - append accepted documents to the current shard
        - update that source's token counter

    4. Source selection should naturally produce approximately:

          DCLM-Edu       30.0%
          FineWeb-Edu    20.0%
          FinePDFs       12.5%
          FineWeb        10.0%
          FineMath       10.0%
          Wikipedia      10.0%
          Cosmopedia      7.5%

       without manually constructing those proportions inside every shard.

    5. Once the current shard reaches approximately SHARD_SIZE:
        - write it
        - clear its buffer
        - continue into the next shard

    6. Do not intentionally split normal documents between shards merely to
       make a shard exactly SHARD_SIZE tokens.

    7. Write the final partial shard if one remains.

    8. Keep memory usage bounded. At no point should the complete 10B-token
       corpus exist in RAM.

    9. Maintain enough statistics to construct metadata.json.

    10. Use DOWNLOAD_WORKERS / ThreadPoolExecutor where useful so downloads
        can overlap processing.

    Returns:
        dict:
            Complete corpus metadata.
    """
    pass


# =============================================================================
# Metadata
# =============================================================================

def build_metadata(
    states: dict[str, SourceState],
    shard_metadata: list[dict],
    tokenizer,
) -> dict:
    """
    Construct metadata describing the finished corpus.

    Requirements:
    - Record:
        tokenizer name
        EOS token
        EOS token ID
        dtype
        configured total token target
        actual total tokens
        shard size target
        number of shards
        random seed

    - For every source record:
        repository
        subset
        configured target
        actual number of accepted tokens
        actual percentage of final corpus

    - Include the metadata returned by write_shard() for every shard.

    - Make everything JSON serializable.

    Returns:
        dict
    """
    pass


def save_metadata(metadata: dict, output_dir: str) -> None:
    """
    Atomically save metadata.json.

    Requirements:
    - Write to a temporary file first.
    - Only replace metadata.json once the complete JSON has been written.
    - Pretty-print it so humans can inspect it easily.

    Returns:
        None
    """
    pass


# =============================================================================
# Reporting
# =============================================================================

def print_progress(
    states: dict[str, SourceState],
    shard_index: int,
    current_shard_tokens: int,
    start_time: float,
) -> None:
    """
    Print useful preparation progress.

    Requirements:
    - Calculate total accepted tokens.
    - Show progress toward TOTAL_TOKENS.
    - Show the current shard number.
    - Show current shard fill.
    - Show overall average tokens/second.
    - Ideally show per-source progress toward its target.
    - Keep output readable rather than printing once per document.

    Returns:
        None
    """
    pass


def print_final_summary(
    states: dict[str, SourceState],
    metadata: dict,
    elapsed: float,
) -> None:
    """
    Print a concise final corpus summary.

    Requirements:
    - Actual total token count.
    - Number of shards.
    - Total preparation time.
    - Average throughput.
    - Target vs actual tokens for every source.
    - Actual mixture percentages.
    - Output directory.

    Returns:
        None
    """
    pass


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Prepare the complete Gouda 2.0 corpus.

    Requirements:
    - Validate configuration.
    - Create required directories.
    - Load and validate the tokenizer.
    - Create source states.
    - Build the mixed corpus.
    - Construct and save metadata.
    - Print the final summary.
    """
    pass


if __name__ == "__main__":
    main()