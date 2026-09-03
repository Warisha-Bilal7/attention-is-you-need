**Pass 3: Mathematical Deep-Dive and Tensor Tracing**
Focus on exact dimensional transformations

**Focus Areas: Section 3.4 to Section 5, tracking every matrix multiplication and shape change**
* Trace a token sequence with batch size *B*, sequence length *L*, and embedding dimension $d_{model} = 512$.
* Track how dimensions transform through $W^Q, W^K, W^V$ projection matrices, multi-head concatenation, LayerNorm (Add & Norm), and position-wise Feed-Forward networks ($d_{ff} = 2048$).
* Examine casual masking in the Decoder to understand how auto-regressive generation works during training vs. inference.

