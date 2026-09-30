from datasets import load_dataset

SOURCES = {
    "dclm-edu": dict(repo="HuggingFaceTB/dclm-edu",kind="text",target=3e9,),
    "fineweb-edu": dict(repo="HuggingFaceTB/smollm-corpus",subset="fineweb-edu-dedup",kind="text",target=2e9,),
    "finepdfs": dict(repo="HuggingFaceFW/finepdfs-edu",subset="eng_Latn",kind="text",target=1.25e9,),
    "fineweb": dict(repo="HuggingFaceFW/fineweb_100BT-shuffled",kind="text",target=1e9,),
    "finemath": dict(repo="HuggingFaceTB/finemath",subset="finemath-4plus",kind="text",target=1e9,),
    "wikipedia": dict(repo="wikimedia/wikipedia",subset="20231101.en",kind="text",target=1e9,),
    "cosmopedia": dict(repo="HuggingFaceTB/smollm-corpus",subset="cosmopedia-v2",kind="text",target=0.75e9,),
}