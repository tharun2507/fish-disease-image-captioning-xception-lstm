import argparse
import os
from pickle import dump

import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

IMAGE_SIZE = (299, 299)
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")


def build_feature_extractor() -> Model:
    """Load Xception with ImageNet weights and strip the classification head."""
    base_model = Xception(weights="imagenet")
    return Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)


def extract_features(data_dir: str, model: Model) -> dict:
    """Walk data_dir, extract a feature vector for every image found.

    Returns a dict mapping image_id (filename without extension) -> feature vector.
    """
    features = {}
    image_paths = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                image_paths.append(os.path.join(root, file))

    print(f"Found {len(image_paths)} images under {data_dir}")

    for path in tqdm(image_paths, desc="Extracting features"):
        try:
            img = load_img(path, target_size=IMAGE_SIZE)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            feature = model.predict(img, verbose=0)
            image_id = os.path.splitext(os.path.basename(path))[0]
            features[image_id] = feature
        except Exception as e:
            print(f"Skipped {path}: {e}")

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract Xception features from a fish image dataset.")
    parser.add_argument("--data_dir", required=True, help="Root directory containing the image dataset.")
    parser.add_argument("--output_dir", required=True, help="Directory to write features_xception.pkl to.")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_feature_extractor()
    print("Xception feature extractor loaded.")

    features = extract_features(args.data_dir, model)

    output_path = os.path.join(args.output_dir, "features_xception.pkl")
    dump(features, open(output_path, "wb"))
    print(f"Saved {len(features)} feature vectors -> {output_path}")


if __name__ == "__main__":
    main()
