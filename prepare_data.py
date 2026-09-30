import multiprocessing as mp
import os

import numpy as np
from datasets import load_dataset
import pyarrow
from tokenizers import Tokenizer
from tqdm import tqdm


REMOTE_NAME = "sample-10BT"
LOCAL_DIR = "edu_fineweb10B"

TOKENIZER = "alphaedge-ai/Qwen3-1.7B-eng-32768"
EOT_TOKEN = "<|endoftext|>"

SHARD_SIZE = 100_000_000


tokenizer = Tokenizer.from_pretrained(TOKENIZER)

if tokenizer.get_vocab_size() != 32768:
    raise RuntimeError(f"Expected vocab size 32768, got {tokenizer.get_vocab_size()}")

EOT = tokenizer.token_to_id(EOT_TOKEN)

if EOT is None:
    raise RuntimeError(f"{EOT_TOKEN} not found in tokenizer")


def tokenize(doc):
    ids = tokenizer.encode(doc["text"], add_special_tokens=False).ids

    tokens = np.empty(len(ids) + 1, dtype=np.uint16)
    tokens[0] = EOT
    tokens[1:] = ids

    return tokens


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options = pyarrow.CacheOptions(
            prefetch_limit = 4
            range_size_limit = 128 << 20,
        ),
    )

    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=REMOTE_NAME,
        split="train",
        streaming=True,
    )

    nprocs = max(1, (os.cpu_count() or 2) // 2)

    print(f"Dataset: FineWeb-Edu {REMOTE_NAME}")
    print(f"Tokenizer: {TOKENIZER}")
    print(f"Vocabulary: {tokenizer.get_vocab_size():,}")
    print(f"EOT: {EOT_TOKEN} ({EOT})")
    print(f"Workers: {nprocs}")
    print(f"Shard size: {SHARD_SIZE:,} tokens")

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty(SHARD_SIZE, dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            offset = 0

            while offset < len(tokens):
                if progress_bar is None:
                    split = "val" if shard_index == 0 else "train"
                    progress_bar = tqdm(
                        total=SHARD_SIZE,
                        unit="tokens",
                        desc=f"Shard {shard_index} ({split})",
                    )

                remaining = SHARD_SIZE - token_count
                n = min(remaining, len(tokens) - offset)

                all_tokens_np[token_count:token_count + n] = tokens[offset:offset + n]

                token_count += n
                offset += n
                progress_bar.update(n)

                if token_count == SHARD_SIZE:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(
                        LOCAL_DIR,
                        f"edufineweb_{split}_{shard_index:06d}",
                    )

                    write_datafile(filename, all_tokens_np)

                    progress_bar.close()
                    progress_bar = None

                    shard_index += 1
                    token_count = 0

        if token_count:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                LOCAL_DIR,
                f"edufineweb_{split}_{shard_index:06d}",
            )

            write_datafile(filename, all_tokens_np[:token_count])

            if progress_bar is not None:
                progress_bar.close()

    print(f"Done. Wrote {shard_index + (token_count > 0)} shards to {LOCAL_DIR}")


if __name__ == "__main__":
    main()