"""Надсилає трафік у локальний сервіс: звичайний (рецензії) або зсунутий (нейтральні твіти)."""
import argparse

import httpx
from datasets import load_dataset

URL = "http://127.0.0.1:8000/predict"


def load_texts(mode: str, n: int) -> list[str]:
    if mode == "normal":
        ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
    else:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
        ds = ds.filter(lambda x: x["label"] == 1)  # 1 = neutral
    ds = ds.shuffle(seed=0).select(range(n))
    return list(ds["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["normal", "drift"])
    parser.add_argument("-n", type=int, default=200)
    args = parser.parse_args()

    texts = load_texts(args.mode, args.n)
    with httpx.Client(timeout=30) as client:
        for i, text in enumerate(texts, 1):
            r = client.post(URL, json={"text": text})
            r.raise_for_status()
            if i % 50 == 0:
                print(f"надіслано {i}/{len(texts)}")
    print("готово")


if __name__ == "__main__":
    main()