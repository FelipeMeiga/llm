## Next Steps / What Remains to Be Implemented

### 1. Tokenization & Vocabulary
- Implement BPE or WordPiece  
  - Extract most frequent pairs from the corpus  
  - Build subword-to-ID table  
- Data loader  
  - Read text files, batch and pad  
  - Create attention mask for padding tokens  

### 2. Embeddings
- Token embedding matrix  
  - Allocate and initialize `[|V| × d_model]`  
  - CUDA kernel for embedding lookup by ID  
- Positional encodings  
  - Sinusoidal or learnable embeddings  
  - Add position embeddings to token embeddings  

### 3. Transformer Decoder-Only Block
- Q/K/V projections (CUDA linear layers)  
  - Forward and backward (compute gradients for weights and biases)  
- Masked self-attention  
  - Causal mask to prevent peeking at future tokens  
  - Stable softmax in CUDA (shared-memory reductions)  
- Multi-head attention  
  - Concatenate head outputs and project back  
- Layer normalization  
  - CUDA kernel for mean/variance calculation + learnable γ/β  
- Feed-forward network (FFN + GELU)  
  - Two linear layers with GELU activation in between  
- Residual connections and normalization  
  - Skip connections before and after the FFN  

### 4. Training Pipeline
- Cross-entropy loss function  
  - Fused softmax + negative log-likelihood + gradient calculation in one CUDA kernel  
- Full backpropagation  
  - Compute gradients for every linear, attention, and LayerNorm operation  
- AdamW optimizer and learning rate scheduler  
  - Moment estimates, weight decay, warm-up and decay schedules  
- Training loop  
  - Forward → compute loss → backward → update weights → log metrics (loss, perplexity)  
- Checkpointing  
  - Save and load model weights to disk  

### 5. Inference & Generation
- Generation strategies  
  - Greedy, top-k sampling, top-p (nucleus) sampling, temperature control  
- Key/value cache  
  - Store past Q/K/V projections to accelerate token-by-token generation  

### 6. Infrastructure & Quality
- Command-line interface and configuration  
  - Flags for training, evaluation, and text generation  
- Unit tests  
  - Numerical validation (e.g., central-difference checks) of forward and backward implementations
