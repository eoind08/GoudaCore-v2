import argparse
import json
import math
import os
import random
import re
import shutil
import time
import unicodedata
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_model, save_model
from tokenizers import Tokenizer

from model import Gouda, ModelConfig


class ShardLoader:
    def __init__(self, data_dir, split, B, T, device, seed=42, shuffle=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.B = B
        self.T = T
        self.device = device
        self.seed = seed
        self.shuffle = shuffle
        self.shards = sorted(self.data_dir.glob(f"*_{split}_*.npy"))

        if not self.shards:
            raise FileNotFoundError(f"no {split} shards found in {self.data_dir}")

        self.lengths = [int(np.load(p, mmap_mode="r").shape[0]) for p in self.shards]
        self.total_tokens = sum(self.lengths)
        self.rng = random.Random(seed)
        self.order = []
        self.order_pos = 0
        self.position = 0
        self.pass_index = 0
        self.tokens = None
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.order = list(range(len(self.shards)))
        if self.shuffle:
            self.rng.shuffle(self.order)
        self.order_pos = 0
        self.position = 0
        self.pass_index = 0
        self._load_shard()

    def _load_shard(self):
        self.tokens = np.load(self.shards[self.order[self.order_pos]], mmap_mode="r")

    def _advance_shard(self):
        self.order_pos += 1
        self.position = 0

        if self.order_pos == len(self.order):
            self.pass_index += 1
            self.order_pos = 0
            self.order = list(range(len(self.shards)))
            if self.shuffle:
                self.rng.shuffle(self.order)

        self._load_shard()

    def next_batch(self):
        n = self.B * self.T

        while len(self.tokens) - self.position < n + 1:
            self._advance_shard()

        buf = np.asarray(
            self.tokens[self.position:self.position + n + 1],
            dtype=np.int64,
        )
        self.position += n

        x = torch.from_numpy(buf[:-1].copy()).view(self.B, self.T)
        y = torch.from_numpy(buf[1:].copy()).view(self.B, self.T)

        return (
            x.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )

    @property
    def current_shard(self):
        path = self.shards[self.order[self.order_pos]]
        match = re.search(r"(\d+)$", path.stem)
        return int(match.group(1)) if match else self.order[self.order_pos]

    @property
    def shard_progress(self):
        return min(self.position / len(self.tokens), 1.0)

    def state_dict(self):
        return {
            "order": self.order,
            "order_pos": self.order_pos,
            "position": self.position,
            "pass_index": self.pass_index,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        self.order = state["order"]
        self.order_pos = state["order_pos"]
        self.position = state["position"]
        self.pass_index = state["pass_index"]
        self.rng.setstate(state["rng_state"])
        self._load_shard()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def slugify(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def autocast_context(device, precision):
    if device.type != "cuda":
        return nullcontext()

    if precision == "bfloat16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast("cuda", dtype=torch.float16)

    return nullcontext()


def build_optimizers(model, cfg):
    muon_params = []
    adamw_decay = []
    adamw_no_decay = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 2 and name != "tok_emb.weight":
            muon_params.append(p)
        elif name == "tok_emb.weight":
            adamw_decay.append(p)
        else:
            adamw_no_decay.append(p)

    Muon = getattr(torch.optim, "Muon", None)
    if Muon is None:
        raise RuntimeError(
            "torch.optim.Muon is unavailable. Install a PyTorch version with built-in Muon support."
        )

    muon_cfg = cfg["optimizer"]["muon"]
    adam_cfg = cfg["optimizer"]["adamw"]

    muon = Muon(
        muon_params,
        lr=float(muon_cfg["lr"]),
        momentum=float(muon_cfg.get("momentum", 0.95)),
        weight_decay=float(muon_cfg.get("weight_decay", 0.1)),
        nesterov=bool(muon_cfg.get("nesterov", True)),
        ns_steps=int(muon_cfg.get("ns_steps", 5)),
        adjust_lr_fn=muon_cfg.get("adjust_lr_fn", "match_rms_adamw"),
    )

    adamw = torch.optim.AdamW(
        [
            {
                "params": adamw_decay,
                "weight_decay": float(adam_cfg.get("weight_decay", 0.1)),
            },
            {
                "params": adamw_no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=float(adam_cfg["lr"]),
        betas=tuple(adam_cfg.get("betas", [0.9, 0.95])),
        eps=float(adam_cfg.get("eps", 1e-8)),
        fused=torch.cuda.is_available(),
    )

    return muon, adamw


def wsd_multiplier(step, total_steps, warmup_fraction, decay_fraction, min_lr_ratio):
    warmup_steps = int(total_steps * warmup_fraction)
    decay_steps = int(total_steps * decay_fraction)

    if warmup_steps + decay_steps > total_steps:
        raise ValueError("WSD warmup_fraction + decay_fraction must be <= 1")

    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps

    decay_start = total_steps - decay_steps

    if decay_steps == 0 or step < decay_start:
        return 1.0

    if decay_steps == 1:
        return min_lr_ratio

    progress = (step - decay_start) / (decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def validate(model, loader, steps, device, precision):
    model.eval()
    loader.reset()

    loss = torch.zeros((), device=device)

    for _ in range(steps):
        x, y = loader.next_batch()

        with autocast_context(device, precision):
            _, batch_loss = model(x, y)

        loss += batch_loss.detach()

    model.train()
    return (loss / steps).item()


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, cfg, device):
    model.eval()

    ids = tokenizer.encode(prompt, add_special_tokens=False).ids

    if not ids:
        raise ValueError("generation prompt tokenized to an empty sequence")

    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

    out = model.generate(
        x,
        max_new_tokens=int(cfg.get("max_new_tokens", 128)),
        temperature=float(cfg.get("temperature", 0.8)),
        top_k=cfg.get("top_k", 50),
    )

    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
    model.train()

    return text


def save_tokenizer(tokenizer, output, max_seq_len):
    try:
        from transformers import PreTrainedTokenizerFast

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<|endoftext|>",
            model_max_length=max_seq_len,
        )
        hf_tokenizer.save_pretrained(output)

    except ImportError:
        tokenizer.save(str(output / "tokenizer.json"))


def save_checkpoint(
    save_dir,
    model,
    model_cfg,
    cfg,
    config_path,
    tokenizer,
    train_loader,
    muon,
    adamw,
    completed_steps,
    cumulative_tokens,
):
    save_dir = Path(save_dir)
    tmp_dir = save_dir.parent / f".{save_dir.name}.tmp"

    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    family = slugify(cfg["name"].split("-")[0])
    model_file = Path(__file__).with_name("model.py")

    save_model(
        model,
        str(tmp_dir / "model.safetensors"),
        metadata={
            "format": "pt",
            "model": cfg["name"],
        },
    )

    shutil.copy2(
        model_file,
        tmp_dir / f"modeling_{family}.py",
    )

    shutil.copy2(
        config_path,
        tmp_dir / "training_config.yaml",
    )

    export_config = asdict(model_cfg)
    export_config.update(
        {
            "model_type": family,
            "architectures": ["Gouda"],
            "name": cfg["name"],
            "torch_dtype": cfg["train"]["precision"],
            "tie_word_embeddings": True,
        }
    )

    with open(tmp_dir / "config.json", "w") as f:
        json.dump(export_config, f, indent=2)

    generation_cfg = cfg.get("generation", {})

    with open(tmp_dir / "generation_config.json", "w") as f:
        json.dump(
            {
                "max_new_tokens": generation_cfg.get("max_new_tokens", 128),
                "temperature": generation_cfg.get("temperature", 0.8),
                "top_k": generation_cfg.get("top_k", 50),
                "do_sample": generation_cfg.get("temperature", 0.8) != 0,
                "eos_token_id": tokenizer.token_to_id("<|endoftext|>"),
            },
            f,
            indent=2,
        )

    save_tokenizer(tokenizer, tmp_dir, model_cfg.max_seq_len)

    trainer_state = {
        "completed_steps": completed_steps,
        "cumulative_tokens": cumulative_tokens,
        "train_loader": train_loader.state_dict(),
        "muon": muon.state_dict(),
        "adamw": adamw.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    torch.save(trainer_state, tmp_dir / "trainer_state.pt")

    shutil.rmtree(save_dir, ignore_errors=True)
    tmp_dir.rename(save_dir)

    latest = save_dir.parent.parent / "latest_checkpoint.txt"
    latest.write_text(str(save_dir.resolve()))

    print(f"saved: {save_dir}")


def load_checkpoint(path, model, train_loader, muon, adamw, device):
    path = Path(path)

    load_model(
        model,
        str(path / "model.safetensors"),
        device=str(device),
    )

    state = torch.load(
        path / "trainer_state.pt",
        map_location=device,
        weights_only=False,
    )

    muon.load_state_dict(state["muon"])
    adamw.load_state_dict(state["adamw"])
    train_loader.load_state_dict(state["train_loader"])

    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())

    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    return int(state["completed_steps"]), int(state["cumulative_tokens"])


def main():
    args = parse_args()
    config_path = Path(args.config)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg["name"]
    model_cfg = ModelConfig(**cfg["model"])

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    schedule_cfg = cfg["scheduler"]
    intervals = cfg["intervals"]

    epochs = float(train_cfg["epochs"])
    B = int(train_cfg["micro_batch_size"])
    T = int(train_cfg["sequence_length"])
    batch_tokens = int(train_cfg["batch_tokens"])
    precision = train_cfg.get("precision", "bfloat16")
    seed = int(train_cfg.get("seed", 42))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    validate_every = int(intervals.get("validate_every", 0))
    save_every = int(intervals.get("save_every", 0))
    generate_every = int(intervals.get("generate_every", 0))
    validation_steps = int(intervals.get("validation_steps", 20))

    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    if T > model_cfg.max_seq_len:
        raise ValueError(
            f"sequence_length={T} exceeds max_seq_len={model_cfg.max_seq_len}"
        )

    micro_tokens = B * T

    if batch_tokens % micro_tokens:
        raise ValueError(
            f"batch_tokens ({batch_tokens}) must be divisible by "
            f"micro_batch_size * sequence_length ({micro_tokens})"
        )

    grad_accum_steps = batch_tokens // micro_tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

    data_dir = Path(data_cfg["directory"])

    train_loader = ShardLoader(
        data_dir,
        "train",
        B,
        T,
        device,
        seed=seed,
        shuffle=True,
    )

    val_loader = ShardLoader(
        data_dir,
        "val",
        B,
        T,
        device,
        seed=seed,
        shuffle=False,
    )

    target_tokens = int(train_loader.total_tokens * epochs)
    max_steps = math.ceil(target_tokens / batch_tokens)

    model = Gouda(model_cfg).to(device)

    muon, adamw = build_optimizers(model, cfg)

    tokenizer_name = data_cfg["tokenizer"]
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)

    if tokenizer.get_vocab_size() != model_cfg.vocab_size:
        raise RuntimeError(
            f"tokenizer vocab={tokenizer.get_vocab_size()} "
            f"but model vocab={model_cfg.vocab_size}"
        )

    compile_enabled = bool(train_cfg.get("compile", True))

    train_model = torch.compile(model) if compile_enabled else model

    output_dir = Path(cfg["save"]["output_dir"]) / slugify(name)
    checkpoint_dir = output_dir / "checkpoints"
    log_file = output_dir / "train.log"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    cumulative_tokens = 0

    if args.resume:
        start_step, cumulative_tokens = load_checkpoint(
            args.resume,
            model,
            train_loader,
            muon,
            adamw,
            device,
        )
        print(
            f"resumed from {args.resume} at step {start_step:,}, "
            f"{cumulative_tokens / 1e9:.3f}B tokens"
        )

    muon_base_lr = float(cfg["optimizer"]["muon"]["lr"])
    adamw_base_lr = float(cfg["optimizer"]["adamw"]["lr"])

    warmup_fraction = float(schedule_cfg.get("warmup_fraction", 0.01))
    decay_fraction = float(schedule_cfg.get("decay_fraction", 0.10))
    min_lr_ratio = float(schedule_cfg.get("min_lr_ratio", 0.1))

    param_count = model.num_parameters()

    print()
    print(f"model:              {name}")
    print(f"parameters:         {param_count:,}")
    print(f"device:             {device}")
    print(f"precision:          {precision}")
    print(f"compile:            {compile_enabled}")
    print(f"train tokens:       {train_loader.total_tokens:,}")
    print(f"validation tokens:  {val_loader.total_tokens:,}")
    print(f"epochs:             {epochs:.4f}")
    print(f"target tokens:      {target_tokens:,}")
    print(f"steps:              {max_steps:,}")
    print(f"sequence length:    {T:,}")
    print(f"micro batch:        {B}")
    print(f"micro tokens:       {micro_tokens:,}")
    print(f"gradient accum:     {grad_accum_steps}")
    print(f"batch tokens:       {batch_tokens:,}")
    print(f"Muon lr:            {muon_base_lr:.4e}")
    print(f"AdamW lr:           {adamw_base_lr:.4e}")
    print()

    with open(log_file, "a") as f:
        f.write(
            f"# {name} | params={param_count} | epochs={epochs} | "
            f"steps={max_steps} | batch_tokens={batch_tokens} | "
            f"compile={compile_enabled}\n"
        )

    completed_steps = start_step

    try:
        for step in range(start_step, max_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()

            lr_mult = wsd_multiplier(
                step,
                max_steps,
                warmup_fraction,
                decay_fraction,
                min_lr_ratio,
            )

            muon_lr = muon_base_lr * lr_mult
            adamw_lr = adamw_base_lr * lr_mult

            set_lr(muon, muon_lr)
            set_lr(adamw, adamw_lr)

            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)

            loss_accum = torch.zeros((), device=device)

            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch()

                with autocast_context(device, precision):
                    _, loss = train_model(x, y)
                    scaled_loss = loss / grad_accum_steps

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at step {step + 1}: {loss.item()}"
                    )

                loss_accum += loss.detach() / grad_accum_steps
                scaled_loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

            muon.step()
            adamw.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            dt = time.perf_counter() - t0
            cumulative_tokens += batch_tokens
            completed_steps = step + 1

            epoch = cumulative_tokens / train_loader.total_tokens
            shard_progress = train_loader.shard_progress
            tokens_per_sec = batch_tokens / dt

            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 1024**3
            else:
                vram = 0.0

            log_line = (
                f"step {step + 1:6d}/{max_steps:<6d} | "
                f"epoch {epoch:7.4f}/{epochs:.4f} | "
                f"shard {train_loader.current_shard:4d} | "
                f"shard_progress {shard_progress * 100:6.2f}% | "
                f"tokens {cumulative_tokens / 1e9:8.4f}B | "
                f"loss {loss_accum.item():.6f} | "
                f"lr {adamw_lr:.4e} | "
                f"muon_lr {muon_lr:.4e} | "
                f"norm {norm.item():.4f} | "
                f"tok/sec {tokens_per_sec:8.0f} | "
                f"dt {dt * 1000:7.1f}ms | "
                f"vram {vram:.2f}GB"
            )

            print(log_line)

            with open(log_file, "a") as f:
                f.write(
                    f"{step + 1} train "
                    f"epoch={epoch:.8f} "
                    f"loss={loss_accum.item():.8f} "
                    f"shard={train_loader.current_shard} "
                    f"shard_progress={shard_progress:.8f} "
                    f"tokens={cumulative_tokens} "
                    f"lr={adamw_lr:.10e} "
                    f"muon_lr={muon_lr:.10e} "
                    f"norm={norm.item():.8f} "
                    f"tok_sec={tokens_per_sec:.2f} "
                    f"dt={dt:.6f} "
                    f"vram_gb={vram:.4f}\n"
                )

            if validate_every > 0 and completed_steps % validate_every == 0:
                val_loss = validate(
                    train_model,
                    val_loader,
                    validation_steps,
                    device,
                    precision,
                )

                print(
                    f"step {completed_steps:6d} | "
                    f"epoch {epoch:7.4f} | "
                    f"validation loss {val_loss:.6f} | "
                    f"ppl {math.exp(min(val_loss, 20)):.4f}"
                )

                with open(log_file, "a") as f:
                    f.write(
                        f"{completed_steps} val "
                        f"epoch={epoch:.8f} "
                        f"loss={val_loss:.8f} "
                        f"ppl={math.exp(min(val_loss, 20)):.8f} "
                        f"tokens={cumulative_tokens}\n"
                    )

            if generate_every > 0 and completed_steps % generate_every == 0:
                print("\n--- generations ---")

                for i, prompt in enumerate(cfg["generation"]["prompts"]):
                    generation = generate_sample(
                        model,
                        tokenizer,
                        prompt,
                        cfg["generation"],
                        device,
                    )

                    print(f"[{i + 1}] {generation}")

                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"{completed_steps} generate "
                            f"epoch={epoch:.8f} "
                            f"prompt={json.dumps(prompt)} "
                            f"text={json.dumps(generation)}\n"
                        )

                print("-------------------\n")

            if save_every > 0 and completed_steps % save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{completed_steps:06d}",
                    model,
                    model_cfg,
                    cfg,
                    config_path,
                    tokenizer,
                    train_loader,
                    muon,
                    adamw,
                    completed_steps,
                    cumulative_tokens,
                )

        final_val_loss = validate(
            train_model,
            val_loader,
            validation_steps,
            device,
            precision,
        )

        final_epoch = cumulative_tokens / train_loader.total_tokens

        print()
        print(f"training complete")
        print(f"steps:      {completed_steps:,}")
        print(f"tokens:     {cumulative_tokens:,}")
        print(f"epoch:      {final_epoch:.6f}")
        print(f"val loss:   {final_val_loss:.6f}")
        print(f"val ppl:    {math.exp(min(final_val_loss, 20)):.4f}")

        with open(log_file, "a") as f:
            f.write(
                f"{completed_steps} final "
                f"epoch={final_epoch:.8f} "
                f"val_loss={final_val_loss:.8f} "
                f"ppl={math.exp(min(final_val_loss, 20)):.8f} "
                f"tokens={cumulative_tokens}\n"
            )

        save_checkpoint(
            output_dir / "final",
            model,
            model_cfg,
            cfg,
            config_path,
            tokenizer,
            train_loader,
            muon,
            adamw,
            completed_steps,
            cumulative_tokens,
        )

    except KeyboardInterrupt:
        print("\ntraining interrupted — saving checkpoint")

        save_checkpoint(
            checkpoint_dir / f"interrupt_{completed_steps:06d}",
            model,
            model_cfg,
            cfg,
            config_path,
            tokenizer,
            train_loader,
            muon,
            adamw,
            completed_steps,
            cumulative_tokens,
        )

        raise


if __name__ == "__main__":
    main()