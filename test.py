import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "./runs/ricotta-2.0/r1/ricotta_2_0/final"

config = AutoConfig.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

print(type(config))
print(config.model_type)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

print(type(model))
print(f"{model.num_parameters():,} parameters")

inputs = tokenizer(
    "The capital of France is",
    return_tensors="pt",
).to("cuda")

with torch.no_grad():
    output = model(**inputs)

print(output.logits.shape)