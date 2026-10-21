
import sys
import time
import torch
import torch.nn.functional as F
import yaml

from model import Gouda, ModelConfig

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

model = Gouda(ModelConfig(**cfg["model"])).cuda()
model.train()

B = 8
T = 2048
V = model.config.vocab_size

x = torch.randint(0, V, (B, T), device="cuda")
y = torch.randint(0, V, (B, T), device="cuda")

# First test without compilation
USE_COMPILE = True
if USE_COMPILE:
    model = torch.compile(model)


def standard_loss():
    logits = model(x)
    return F.cross_entropy(
        logits.float().reshape(-1, V),
        y.reshape(-1),
    )


def cce_loss():
    _, loss = model(x, y)
    return loss


def benchmark(name, loss_fn, warmup=5, steps=20):
    for i in range(warmup + steps):
        model.zero_grad(set_to_none=True)

        if i == warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn()

        loss.backward()

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    tok_per_sec = (steps * B * T) / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    print(f"{name}:")
    print(f"  Tokens/sec: {tok_per_sec:,.0f}")
    print(f"  Peak VRAM:  {peak_gb:.2f} GiB")
    print(f"  Loss:       {loss.item():.6f}")

    return tok_per_sec


ce_speed = benchmark("Standard CE", standard_loss)
cce_speed = benchmark("Cut Cross-Entropy", cce_loss)

print(f"\nCCE speed ratio: {cce_speed / ce_speed:.3f}x")
