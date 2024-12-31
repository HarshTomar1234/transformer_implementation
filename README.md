# Neural Machine Translation with Transformers

This repository implements the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) for neural machine translation tasks.

![Transformer Architecture](https://miro.medium.com/max/700/1*BHzGVskWGS_3jEcYYi6miQ.png)

## Features

- Complete Transformer architecture implementation in PyTorch
- Multi-head self-attention mechanism
- Positional encoding using sine and cosine functions
- Label smoothing and dropout for regularization  
- Dynamic batching based on sequence lengths
- Training progress tracking with TensorBoard
- Memory-efficient implementation with batch processing
- Support for multiple language pairs

## Requirements
```bash
torch
datasets
tokenizers
tqdm
tensorboard
psutil
```


## Project Structure
```bash
├── model.py          # Transformer model implementation
├── dataset.py        # Dataset handling and processing
├── config.py         # Configuration parameters
└── train.py         # Training loop and utilities
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure training parameters in config.py:
   ```bash
   config = {
    'batch_size': 8,
    'num_epochs': 20,
    'd_model': 512,
    'lang_src': 'en',
    'lang_tgt': 'fr',
    'seq_len': 350
   }
 ```

3. Start Training:
```bash
python train.py
```


   
 


## Implementation Details

Tokenization: WordLevel tokenizer with special tokens ([PAD], [UNK], [SOS], [EOS])
Dataset: Supports HuggingFace datasets API
Memory Management: Periodic garbage collection and memory usage monitoring
Validation: Greedy decoding with periodic evaluation
Training: Cross-entropy loss with label smoothing
Model Saving: Checkpoints saved after each epoch

## Paper Reference
The implementation is based on the original Transformer paper:

```bash
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## Key Features from Paper

Multi-head attention mechanism allowing model to focus on different positions
Scaled dot-product attention for stable gradients
Positional encodings for sequence order information
Layer normalization and residual connections
Label smoothing for better generalization

## Performance Optimizations

Dynamic batching of similar-length sequences
Memory-efficient implementation
Periodic validation and model checkpointing
TensorBoard integration for training monitoring

## Acknowledgments
Implementation inspired by the original paper "Attention Is All You Need" by Vaswani et al. Additional techniques from recent transformer implementations in the PyTorch ecosystem.
   
