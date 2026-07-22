# fish-disease-image-captioning-xception-lstm
Image captioning for freshwater fish disease images using Xception transfer learning + LSTM
Fish Disease Image Captioning (Xception + LSTM)

An image captioning system that generates descriptive captions for freshwater fish disease images using transfer learning. A pretrained Xception CNN extracts deep visual features, which are combined with an LSTM-based sequence model to generate natural-language descriptions of the disease condition present in each image.

Overview

Manually describing and cataloguing disease symptoms in aquaculture imagery is slow and inconsistent. This project automates that process by pairing a frozen, ImageNet-pretrained Xception network (as a feature extractor) with a trainable Embedding + LSTM decoder, trained end-to-end via teacher forcing to produce captions describing the visual disease characteristics of each fish image.
