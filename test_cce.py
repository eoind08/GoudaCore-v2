
import sys
import torch
import torch.nn.functional as F
import yaml

from model import Gouda, ModelConfig

assert torch.cuda.is_available(), "CCE requires CUDA"

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

torch.manual_seed(42)
model = Gouda(ModelConfig(**cfg["model"])).cuda()
model.eval()

vocab = model.config.vocab_size

# Small synthetic batch
tokens = torch.randint(0, vocab, (1, 33), device="cuda")
x = tokens[:, :-1]
y = tokens[:, 1:].clone()

# Test ignored targets too
y[0, -1] = -100

# 1. Standard cross-entropy
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(x)
    ce_loss = F.cross_entropy(
        logits.float().reshape(-1, vocab),
        y.reshape(-1),
        ignore_index=-100,
    )

# 2. Cut Cross-Entropy
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    output, cce_loss = model(x, y)

print(f"Standard CE: {ce_loss.item():.6f}")
print(f"CCE:         {cce_loss.item():.6f}")
print(f"Difference:  {abs(ce_loss.item() - cce_loss.item()):.6f}")

assert output is None
assert torch.isfinite(cce_loss)
assert abs(ce_loss.item() - cce_loss.item()) < 0.05

# 3. Test backward propagation
model.zero_grad(set_to_none=True)

with torch.autocast("cuda", dtype=torch.bfloat16):
    _, loss = model(x, y)

loss.backward()

for name, p in model.named_parameters():
    if p.requires_grad:
        assert p.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(p.grad).all(), f"Bad gradient: {name}"

print("CCE forward and backward tests PASSED")
