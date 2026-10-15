import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./runs/ricotta-2.0/r1-wsd10(minlr0.1)/ricotta_2_0/final"
NUM_EXAMPLES = 10
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16,).to(DEVICE).eval()
dataset = load_dataset("allenai/sciq", split="test",)

@torch.no_grad()
def continuation_loglikelihood(context, continuation):
    """
    Returns:
        total log-likelihood
        mean log-likelihood per continuation token
        number of continuation tokens
    """
    context_ids = tokenizer(context, add_special_tokens=False,).input_ids
    full_ids = tokenizer(context + continuation, add_special_tokens=False,).input_ids
    continuation_ids = full_ids[len(context_ids):]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=DEVICE,)

    logits = model(input_ids=input_ids,).logits.float()

    # logits[i] predicts token i+1
    start = len(context_ids) - 1
    end = len(full_ids) - 1

    continuation_logits = logits[0,start:end,:]
    targets = torch.tensor(continuation_ids, dtype=torch.long, device=DEVICE,)

    log_probs = torch.log_softmax(continuation_logits, dim=-1,)
    token_logprobs = log_probs.gather(1, targets.unsqueeze(1),).squeeze(1)

    total = token_logprobs.sum().item()
    mean = token_logprobs.mean().item()

    return total, mean, len(continuation_ids)


for i, example in enumerate(dataset.select(range(NUM_EXAMPLES))):
    support = example["support"].lstrip()
    question = example["question"]
    choices = [example["distractor1"], example["distractor2"], example["distractor3"], example["correct_answer"],]
    correct_index = 3

    context = (f"{support}\n" f" Question: {question}\n" f" Answer:")
    results = []

    for j, choice in enumerate(choices):
        # lm-eval's default target_delimiter is one space
        continuation = " " + choice

        total_ll, mean_ll, n_tokens = (continuation_loglikelihood( context, continuation,))
        results.append(
            {
                "index": j,
                "choice": choice,
                "total": total_ll,
                "mean": mean_ll,
                "tokens": n_tokens,
            }
        )

    raw_winner = max(results, key=lambda x: x["total"],)["index"]
    norm_winner = max(results, key=lambda x: x["mean"],)["index"]

    print("=" * 100)
    print(f"EXAMPLE {i + 1}")
    print()
    print("SUPPORT:")
    print(support)
    print()
    print("QUESTION:")
    print(question)
    print()

    print(
        f"{'':3} "
        f"{'ANSWER':45} "
        f"{'TOKENS':>6} "
        f"{'TOTAL LL':>12} "
        f"{'MEAN LL':>12}"
    )
    print("-" * 100)

    for r in results:
        correct = "✓" if r["index"] == correct_index else " "
        raw = "R" if r["index"] == raw_winner else " "
        norm = "N" if r["index"] == norm_winner else " "

        answer = r["choice"]
        if len(answer) > 43:
            answer = answer[:40] + "..."

        print(
            f"{correct}{raw}{norm} "
            f"{answer:45} "
            f"{r['tokens']:6d} "
            f"{r['total']:12.4f} "
            f"{r['mean']:12.4f}"
        )

    print()
    print(f"Correct:       {choices[correct_index]}")
    print(f"Raw winner:    {choices[raw_winner]} " f"{'✓' if raw_winner == correct_index else '✗'}")
    print(f"Norm winner:   {choices[norm_winner]} " f"{'✓' if norm_winner == correct_index else '✗'}")

print("=" * 100)