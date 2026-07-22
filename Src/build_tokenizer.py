"""
build_tokenizer.py

Builds and saves a Keras Tokenizer (vocabulary) from the caption annotation
file produced by generate_captions.py. Run once, before train.py.

Usage:
    python src/build_tokenizer.py --captions outputs/ALL_CAPTIONS.txt --output_dir outputs/
"""

import argparse
import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer


def load_captions(filename: str) -> dict:
    """Parse a '<file>#<idx> <caption>' annotation file into {image_id: [captions]}."""
    mapping = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id, caption = line.split("#", 1)
            caption = caption.split(" ", 1)[1] if caption[0].isdigit() else caption
            caption = f"startseq {caption.strip()} endseq"
            mapping.setdefault(image_id, []).append(caption)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Build a tokenizer/vocabulary from caption annotations.")
    parser.add_argument("--captions", required=True, help="Path to the caption annotation file.")
    parser.add_argument("--output_dir", required=True, help="Directory to write tokenizer.pkl to.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    captions_dict = load_captions(args.captions)
    all_captions = [c for caps in captions_dict.values() for c in caps]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in all_captions)

    output_path = os.path.join(args.output_dir, "tokenizer.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Vocab size: {vocab_size}")
    print(f"Max caption length: {max_length}")
    print(f"Tokenizer saved -> {output_path}")


if __name__ == "__main__":
    main()
