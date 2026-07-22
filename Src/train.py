import argparse
import os
import pickle
import random
from pickle import load

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from build_tokenizer import load_captions

BATCH_SIZE = 32
FEATURE_DIM = 2048
EMBEDDING_DIM = 256


def data_generator(captions_dict, features, tokenizer, vocab_size, max_length, batch_size=BATCH_SIZE):
    """Yields (image_features, partial_caption_seq), next_word batches, forever."""
    while True:
        keys = list(captions_dict.keys())
        random.shuffle(keys)
        X1, X2, y = [], [], []

        for key in keys:
            if key not in features:
                continue
            feature = features[key]

            for caption in captions_dict[key]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical(out_seq, num_classes=vocab_size)

                    X1.append(feature[0])
                    X2.append(in_seq)
                    y.append(out_seq)

                    if len(X1) == batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)
                        X1, X2, y = [], [], []


def define_model(vocab_size: int, max_length: int) -> Model:
    """CNN feature input + LSTM caption decoder, fused via addition."""
    inputs1 = Input(shape=(FEATURE_DIM,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(EMBEDDING_DIM, activation="relu")(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(EMBEDDING_DIM)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(EMBEDDING_DIM, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the Xception+LSTM image captioning model.")
    parser.add_argument("--features", required=True, help="Path to features_xception.pkl")
    parser.add_argument("--captions", required=True, help="Path to the caption annotation file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.pkl")
    parser.add_argument("--output_dir", required=True, help="Directory to save model checkpoints to")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading features, captions, and tokenizer...")
    features = load(open(args.features, "rb"))
    captions_dict = load_captions(args.captions)
    tokenizer = pickle.load(open(args.tokenizer, "rb"))

    all_captions = [c for caps in captions_dict.values() for c in caps]
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in all_captions)
    print(f"Vocab size: {vocab_size} | Max caption length: {max_length}")

    model = define_model(vocab_size, max_length)
    model.summary()

    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, "best_model.h5"),
        monitor="loss",
        save_best_only=True,
        verbose=1,
    )

    steps_per_epoch = len(captions_dict) // BATCH_SIZE
    model.fit(
        data_generator(captions_dict, features, tokenizer, vocab_size, max_length),
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        callbacks=[checkpoint],
        verbose=1,
    )

    final_path = os.path.join(args.output_dir, "caption_model.keras")
    model.save(final_path)
    print(f"Training complete. Final model saved -> {final_path}")


if __name__ == "__main__":
    main()
