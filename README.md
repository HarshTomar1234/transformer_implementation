# Neural Machine Translation with Transformers

This repository implements the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) for neural machine translation tasks.

![Transformer Architecture](https://miro.medium.com/max/700/1*BHzGVskWGS_3jEcYYi6miQ.png)

---

## Features

- Complete implementation of the Transformer architecture in PyTorch
- Multi-head self-attention mechanism
- Positional encoding using sine and cosine functions
- Label smoothing and dropout for regularization
- Dynamic batching based on sequence lengths
- Training progress tracking with TensorBoard
- Memory-efficient implementation with batch processing
- Support for multiple language pairs

---

## Requirements

To install the required dependencies, ensure you have Python installed and execute the following:

```bash
pip install torch datasets tokenizers tqdm tensorboard psutil
```

---

## Project Structure

```plaintext
├── model.py          # Transformer model implementation
├── dataset.py        # Dataset handling and processing
├── config.py         # Configuration parameters
└── train.py          # Training loop and utilities
```

---

## Quick Start

### Step 1: Install Dependencies

Install the necessary Python libraries:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Training Parameters

Edit the `config.py` file to set your training parameters:

```python
config = {
    'batch_size': 8,
    'num_epochs': 20,
    'd_model': 512,
    'lang_src': 'en',
    'lang_tgt': 'fr',
    'seq_len': 350
}
```

### Step 3: Start Training

Run the training script:

```bash
python train.py
```

---

## Implementation Details

### Key Components

- **Tokenization**: Utilizes a WordLevel tokenizer with special tokens (`[PAD]`, `[UNK]`, `[SOS]`, `[EOS]`)
- **Dataset**: Supports the HuggingFace Datasets API for efficient data handling
- **Memory Management**: Incorporates periodic garbage collection and memory usage monitoring
- **Validation**: Performs greedy decoding for periodic evaluation during training
- **Training**: Employs cross-entropy loss with label smoothing for better generalization
- **Model Saving**: Automatically saves checkpoints after each training epoch

---

## Reference Paper

The implementation is inspired by the original Transformer paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

---

## Key Features from the Paper

- Multi-head attention mechanism for focusing on different positions
- Scaled dot-product attention to stabilize gradients
- Positional encodings to preserve sequence order
- Layer normalization and residual connections for improved training stability
- Label smoothing for enhanced generalization

---

## Performance Optimizations

- Dynamic batching of sequences with similar lengths to optimize memory usage
- Efficient memory management during training
- Periodic validation and automatic model checkpointing
- TensorBoard integration for real-time monitoring of training metrics

---

## Acknowledgments

This implementation is inspired by the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. It also incorporates techniques and best practices from recent Transformer implementations in the PyTorch ecosystem.

