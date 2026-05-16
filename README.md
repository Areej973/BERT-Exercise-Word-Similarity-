# BERT Exercise Word Similarity 

This project demonstrates how to extract and analyze contextual word embeddings from pre-trained language models (such as BERT) using PyTorch and the Hugging Face Transformers library. It highlights the process of dynamic token indexing to capture the exact representation of a target word based on its surrounding context.

## Table of Contents
- Description
- Pipeline Overview
- Key Features
- Installation
- Usage
- Code Explanation

## Description
Traditional word embeddings (like Word2Vec or GloVe) assign a static vector to each word regardless of its context. Modern transformer-based models generate contextualized embeddings, meaning the vector representation of a word changes depending on how it is used in a sentence. This project tokenizes two different sentences containing the same target word, extracts their embeddings from the model's last hidden state, and prepares them for semantic similarity analysis.

## Pipeline Overview
1. **Tokenization:** Text is converted into token IDs, attention masks, and token type IDs using a pre-trained tokenizer.
2. **Validation:** Token IDs are decoded back to human-readable strings to verify the alignment.
3. **Inference:** The tokenized inputs are passed through the model on an evaluation mode to generate hidden states without calculating gradients.
4. **Dynamic Indexing:** A custom function searches for the exact position of the target word inside the tokenized sequence.
5. **Extraction and Conversion:** The specific word embedding is extracted from the last hidden layer and converted into a NumPy array.

## Key Features
- **Dynamic Word Indexing:** Automatically locates target words or subwords within the tokenized inputs instead of using hard-coded index values.
- **GPU Acceleration:** Leverages CUDA for faster model inference.
- **Memory Optimization:** Uses PyTorch evaluation mode and disables gradient calculations to reduce VRAM consumption.

## Installation
To run this script, you need to install PyTorch and Hugging Face Transformers. You can install them using pip:

```bash
pip install torch transformers numpy scipy
