
# RNN Lyrics Generation
![Image](https://github.com/user-attachments/assets/252e361d-1461-472d-8e42-d7f6037ae0b5)


This project explores using Long Short-Term Memory (LSTM) networks to generate song lyrics conditioned on a provided melody. Trained on paired lyric and MIDI data, the model learns to fuse musical and textual features and produce context-appropriate next-word predictions in PyTorch.



## Table of Contents

1. [Overview](#overview)  
2. [Data](#data)  
3. [Model architecture](#model-architecture)  
4. [Training and Experiments](#training-and-experiments)  
5. [Results](#results)  
6. [Conclusion](#conclusion)  
7. [Authors](#authors) 
## Overview

Two variants of an LSTM-based lyric generator were implemented. During training, each time step receives both a Word2Vec embedding of the current word and features extracted from the corresponding segment of the melody. At inference time, the network samples its probability distribution to produce novel lyrics for unseen melodies.

## Data

### Dataset overview

The dataset comprises paired lyric and MIDI files for 600 training songs (`lyrics_train_set.csv`) and 5 test songs (`lyrics_test_set.csv`), with each CSV row containing a song identifier alongside its full lyrics (tokens separated by “&”). One `.mid` file per song encodes note, instrument and timing information.  

To illustrate, here is a snippet of the test set:

![Image](https://github.com/user-attachments/assets/0eaa60a9-0d84-4ee6-8c50-c1caa1b1dbbd)

Most common words: 
![Image](https://github.com/user-attachments/assets/95c24220-7223-402f-a0a8-24ab42944de5)

### Preprocessing

All lyric text was cleaned by removing non-alphanumeric characters, normalizing contractions (for example, “don’t” → “do not”), and filling any missing entries to ensure consistency. The cleaned lyrics were then tokenized into sequences of Word2Vec embeddings. Finally, 10 % of the training songs were held out as a validation set (90 % train / 10 % validation) to tune hyperparameters and guard against overfitting.  

Example of the train DF:

![Image](https://github.com/user-attachments/assets/ee1393a0-d736-41ba-9860-e8e6115e641e)



## Model architecture

Two variants of melody integration were implemented, each feeding into the same LSTM backbone:

### Variant 1: Word-level melody features  
For each word in the lyrics, the corresponding segment of the MIDI piano roll (5 fps) is extracted, normalized by active note count, and summarized via four statistics (mean, std, max, min). Missing or empty segments are filled with small constants to preserve shape. This yields a per-word feature vector that captures local melodic context.

### Variant 2: Song-level melody features  
A single global feature vector is computed for each song, including estimated tempo (BPM), pitch-class histogram, note density (notes/sec), total beat and downbeat counts, and duration. This vector is then concatenated to every time step, providing a high-level summary of the melody throughout the sequence.

### Shared network design  
1. **Inputs**  
   - 300-dim Word2Vec embedding  
   - MIDI feature vector (variant-dependent)  
2. **LSTM layers**  
   - Two stacked LSTMs (hidden size = 256), capturing temporal dependencies  
3. **Fully connected head**  
   - FC1: Linear → ReLU → Dropout (p = 0.2)  
   - FC2: Linear → logits  
4. **Output layer**  
   - Logit clamping & scaling for numerical stability  
   - Softmax over the vocabulary  

### Key hyperparameters  
- **Learning rate:** 0.01  
- **Batch size:** 32  
- **Optimizer:** Adam  
- **Epochs:** up to 50 (early stopping, patience = 3)  
- **Loss:** CrossEntropyLoss  
- **Gradient clipping:** max-norm = 1.0  



  
## Training

Both the word‐level and song‐level variants were trained using the notebook’s `train_model_per_word` and `train_model_per_song` functions. Each run used:

- **Optimizer:** Adam (learning rate = 1 × 10⁻³)  
- **Batch size:** 32  
- **Epochs:** up to 20, with early stopping (patience = 3 epochs) on validation loss  
- **Metrics logged:** training & validation loss, mean squared error, and cosine similarity (to TensorBoard)

Training curves showed a steady decrease in both loss and MSE, accompanied by rising cosine similarity on the validation set. Early stopping triggered after 15 epochs for the word‐level model and 17 epochs for the song‐level model, yielding the checkpoints with the best generalization performance.

Example of training on the word-level method:

![Image](https://github.com/user-attachments/assets/c4dfe9aa-0eb1-4457-9404-e56186b54aee)

Example of training on the song‐level method:

![Image](https://github.com/user-attachments/assets/ce5a6e4d-e9dd-49f1-812b-f950e8098813)## Results

### Word-level Model

![Image](https://github.com/user-attachments/assets/6d252580-8214-4904-a8c8-41a5d0a3bc90) 
![Image](https://github.com/user-attachments/assets/f251b005-8e28-424c-bc39-3269e577b45e)

_Final validation loss: 5.85, MSE: 0.0001, Cosine sim: 0.15_

### Song-level Model

![Image](https://github.com/user-attachments/assets/188b988f-d713-4b79-9890-0d326016de0f)
![Image](https://github.com/user-attachments/assets/a07e5877-bf43-4142-8fa7-19fcc127cc5d)

_Final validation loss: 6.02, MSE: 0.0001, Cosine sim: 0.14_



## Conclusion

Different melody‐feature extraction strategies lead to noticeably different generative behaviors, underlining the importance of how musical context is represented. While fully coherent lyric generation remains an open challenge, the model’s ability to produce varied vocabulary in response to melody is a promising proof of concept. This project deepened our understanding of music–language fusion in RNNs and provides a foundation for future work on richer feature integration, advanced architectures, and improved sampling methods.

## Authors

- Roi Garber

- Nicole Kaplan
