import torch 
import torch.nn as nn  
import math  

class InputEmbeddings(nn.Module):  

    def __init__(self, d_model: int, vocab_size: int): 
        super().__init__()  
        self.d_model = d_model  # Store the model's embedding dimension
        self.vocab_size = vocab_size  # Store the vocabulary size
        
        # Create an embedding layer that maps vocabulary indices to dense vectors
        # vocab_size: number of unique tokens, d_model: dimensionality of embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):  # Forward method for processing input
        # Multiply embeddings by sqrt(d_model) for scaling (from original transformer paper)
        # This helps prevent embeddings from growing too large during training
        return self.embedding(x) * math.sqrt(self.d_model)
    


class PositionalEncoding(nn.Module): 

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:  
        super().__init__()  
        
        self.d_model = d_model  # Store model's embedding dimension
        self.seq_len = seq_len  # Store maximum sequence length
        
        # Create a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of zeros to store positional encodings
        # Shape: (seq_len, d_model)
        pos_enc = torch.zeros(seq_len, d_model)
        
        # Create a column vector of positions
        # Shape: (seq_len, 1)
        # arange creates a tensor (0, 1, 2, ..., seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        
        # Create a term for the sinusoidal function
        # Generates a vector of exponentially decreasing values
        # Used to create unique encoding for different positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4, ...)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension
        # Shape changes from (seq_len, d_model) to (1, seq_len, d_model)
        pos_enc = pos_enc.unsqueeze(0)
        
        # Register the positional encoding as a buffer
        # This means it's saved with the model but not treated as a learnable parameter
        self.register_buffer('pos_enc', pos_enc)
    
    def forward(self, x):  # Forward method to add positional encoding to input
        # Adding positional encoding to the input
        # Slicing positional encoding to match input sequence length
        # requires_grad_(False) prevents gradient computation for positional encoding
        x = x + (self.pos_enc[:, :x.shape[1], :]).requires_grad_(False)
        
        # Apply dropout to prevent overfitting
        return self.dropout(x)


# Layer Normalization: Normalizes input across feature dimensions
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        # Ensuring features is an integer
        features = features if isinstance(features, int) else features.item()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter (scaling)
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter (shifting)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

    
# Feed Forward Block: Applies non-linear transformations
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # Two linear transformations with ReLU activation in between
        self.linear_1 = nn.Linear(d_model, d_ff)  # Expand dimension
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # Back to original dimension
    
    def forward(self, x):
        # Transform input: (Batch, seq_len, d_model) → (Batch, seq_len, d_ff) → (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        # d_model: Total dimensionality of the model
        # h: Number of attention heads
        # Ensuring that the model dimension is divisible by the number of heads to get equality in embeddings
        # if d_model is 512 then Each head will get 512 ÷ 8 = 64 dimensions this is essential because :-
               ## Consistent computational complexity
               ## Balanced representation learning
               ## Prevents arbitrary truncation or padding of embeddings
               ## Allows symmetric information processing across heads
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_model = d_model  # Total model dimension
        self.h = h  # Number of attention heads
        self.d_k = d_model // h  # Dimension for each head
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Projection layers for Query, Key, and Value
        # These linear layers transform input to create query, key, and value representations
        # 
        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection
        self.w_v = nn.Linear(d_model, d_model)  # Value projection
        
        # Output projection layer to combine multi-head attention outputs
        self.w_o = nn.Linear(d_model, d_model)  # Output projection

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # Core attention mechanism implementation
        
        # Calculate attention scores
        # 1. Dot product between query and key
        # 2. Scale by square root of key dimension to stabilize gradients
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (typically used in decoder to prevent attending to future tokens)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)  # Large negative value effectively zeros out masked positions
        
        # Softmax to create attention probabilities
        attention_scores = attention_scores.softmax(dim=-1)
        
        # Apply dropout for regularization
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Compute weighted sum of values using attention scores
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Forward pass of multi-head attention
        
        # Project inputs to query, key, and value spaces
        query = self.w_q(q)  # (Batch, seq_len, d_model)
        key = self.w_k(k)    # (Batch, seq_len, d_model)
        value = self.w_v(v)  # (Batch, seq_len, d_model)
        
        # Reshape and transpose to separate multiple heads
        # Rearrange from (Batch, seq_len, d_model) to (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # .view() Method:
        # - Reshapes tensor without copying data
        # - Total elements must remain constant
        # - Like reorganizing a grid without adding/removing cells

        
        # Compute attention for multiple heads
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        
        # Concatenate multi-head outputs
        # Transpose back to (Batch, seq_len, h, d_k) and reshape to (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        #  .contiguous() Method:
            # - Ensures memory is laid out sequentially
            # - Prepares tensor for operations that require contiguous memory
            # - Prevents potential performance bottlenecks

        
        # Final linear projection
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__() 
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)       
        return self.norm(x)    
    


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range (3)])


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x



class Decoder(nn.Module):

    def __init__(self, features: int,  layers: nn.ModuleList) -> None:
        super().__init__() 
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)       
        return self.norm(x)  


class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model)  --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection_layer(x), dim = -1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeddings: InputEmbeddings, tgt_embeddings: InputEmbeddings, src_position: PositionalEncoding, tgt_position: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embeddings(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self , x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048)  -> Transformer:
    # Create the embedding layers
    src_embeddings = InputEmbeddings(d_model, src_vocab_size)
    tgt_embeddings = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_position = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_position = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)    

    # Create the encoder and the decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))


    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embeddings, tgt_embeddings, src_position, tgt_position, projection_layer)


    # Initialize the parameters
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer    
        
        

