"""
inference.py

Loads a trained captioning model and tokenizer, and generates a caption for
a single new image using greedy decoding.

Usage:
    python src/inference.py \
        --image path/to/image.jpg \
        --model outputs/caption_model.keras \
        --tokenizer outputs/tokenizer.pkl \
        --max_length 19
"""

import argparse
import os
import pickle

import numpy as np
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

IMAGE_SIZE = (299, 299)


def build_feature_extractor() -> Model:
    base_model = Xception(weights="imagenet")
    return Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)


def extract_feature(image_path: str, feature_model: Model) -> np.ndarray:
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return feature_model.predict(img, verbose=0)


def generate_caption(model: Model, tokenizer, photo_feature: np.ndarray, max_length: int) -> str:
    """Greedy decode: predict one word at a time until <endseq> or max_length."""
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    in_text = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo_feature, seq], verbose=0)
        yhat = np.argmax(yhat[0])
        word = index_word.get(yhat)

        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Generate a caption for a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", required=True, help="Path to the trained .keras model.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.pkl.")
    parser.add_argument("--max_length", type=int, default=19, help="Max caption length used during training.")
    args = parser.parse_args()

    for path, label in [(args.image, "Image"), (args.model, "Model"), (args.tokenizer, "Tokenizer")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    print("Loading model and tokenizer...")
    model = load_model(args.model)
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading Xception feature extractor...")
    feature_model = build_feature_extractor()

    print(f"Extracting features for {args.image}...")
    photo_feature = extract_feature(args.image, feature_model)

    caption = generate_caption(model, tokenizer, photo_feature, args.max_length)
    print(f"\nGenerated caption: {caption}")


if __name__ == "__main__":
    main()
