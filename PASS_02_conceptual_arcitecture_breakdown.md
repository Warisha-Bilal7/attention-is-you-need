**Pass 2: Conceptual and Architectural Breakdown**
*  Focus on information flow and tensor paths

**Target Areas: Section 3.2 (Attention) and Section 3.3 (Position-wise Feed-Forward Networks)**
*   *Self-Attention Mechanics:* Map out Q(Query), K (Key) and V(Value) interactions. Understand why dot-products scale with $\sqrt{d_k}$
*   *Multi-Head Dynamics:* Analyze how splitting projection heads allows the network to jointly attend to information from different representation subspaces at different positions.
*   *Positional Encoding:* Track how sequence order is injected using fixed sinusoidal functions ($sin$/$cos$) without adding recurrent steps.

**Simple-Word Explanation of Section 3.2**
Think of attention as a "soft-lookup" in a dictionary. When you want to process a word, you don't look at it in isolation but you look at the surrounding words to fully understand its context.

**Section 3.2.1 Scaled Dot-Product Attention**
This is the engine that decides which words in a sentence should focus on each other using three vectors for every word:
*  *Query(Q)* The word you are trying to process.
*  *Key(K):* The label or tag of every word in a sentence (for example; noun, pronoun, verb)
*  *Value(V):* actual meaning or information of each word

The model multiplies the *Query* of the current word with *Keys* of all other words. This "dot product" produces a score of how the Q is related to other words. These scores are passed through a **Soft-max** function, which squashes them into a percentage distribution that sums up to 0.1(100%). Finally, these percentages are multiplied with the *Value* vector to get a final contextual blend of the words.
* *The Scaling Trick ($\sqrt{d_k}$): As the key dimension ($d_k$) gets larger, the math behind the dot-products produces very large numbers. This pushes the softmax function into flat zones where the gradients become tiny, causing stalls in the training. Dividing by $\sqrt{d_k}$ rescales the scores, keeping mathematical distribution soft and trainable.

**Section 3.2.2 Multi-Head Attention**
If you only run one attention calculation, the model has to condense all relationships into a single weighted average. However, language has multiple layers of complexity at once—a word might have a grammatical relationship with a verb, a coreference relationship with a pronoun, and a positional relationship with its neighbor
*Multi-Head Attention* solves this by running several attention calculation ("heads") in parallel. Instead of looking at the full-sized vectors, the model projects Q, K and V into h (usually 8) separate, smaller dimensions. Each head performs the scaled dot-product calculation independently in its own subspace. One head might learn to focus on nouns, one on pronouns and one on verbs.
Once all heads are finished, the model concatenates (glues) their outputs back together and runs them through a final layers to mix the transformation. Due to reduced dimensions of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

**Section 3.2.3 Application of Attention in our Model**
* *1. Encoder Self-Attention*: The input sentence attends to itself. Q, K and V all come from the previous encoder layer. This allows very word in the input sequence to look at every other word to build its representation.
* *2. Decode Self-Attention (Masked)*: The generated translation attends to itself. However, to prevent the model from cheating during training, apply a casual *mask* which forces position *i* to only attend to itself and preceding words, It blocks out future words by setting their attention score to $-\infty$ so they get a weight of zero.
* *3. Encoder-Decoder Attention (Cross-Attention):* Bridge between the two stacks. Q comes from the previous decoder layer while K and V come from output of the encoder. This lets every word being written in the translation reach back and look at the entire original sentence to find the right contextual words.

[ Input Tensor: X ]  (Batch, Seq_Len, d_model)
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
         [ W^Q ]            [ W^K ]            [ W^V ]      <-- Projection Matrices
             |                  |                  |
             v                  v                  v
        [ Q_proj ]         [ K_proj ]         [ V_proj ]    (Batch, Seq_Len, h * d_k)
             |                  |                  |
             +------------------+------------------+
                                |  (Reshape & Transpose)
                                v
                      [ Split into h Heads ]
               Q, K, V shape: (Batch, h, Seq_Len, d_k)
                                |
                                v
               [ Score Matrix: S = (Q @ K^T) / sqrt(d_k) ]
                         Shape: (Batch, h, Seq_Len, Seq_Len)
                                |
                                v
                     [ Apply Causal Mask (opt.) ]
                                |
                                v
                        [ Softmax(S) ]
                                |
                                v
               [ Weighted Values: Output = Softmax(S) @ V ]
                         Shape: (Batch, h, Seq_Len, d_v)
                                |
                                v
                     [ Concatenate Heads ]
                       Shape: (Batch, Seq_Len, h * d_v)
                                |
                                v
                         [ Linear: W^O ]
                                |
                                v
                    [ Multi-Head Attention Output ] (Batch, Seq_Len, d_model)


**1. Self-Attention Mechanics & Scaled Dot-Products**
Let us map out the mathematical transformations at the tensor level.
*Linear Projection*
We begin with a sequence of token representations grouped into a matrix $X \in \mathbb{R}^{n \times d_{\text{model}}}$, where $n$ is the sequence length and $d_{\text{model}}$ is the model's hidden dimension24. We apply three learnable weight matrices to project $X$:
$$Q = XW^Q \in \mathbb{R}^{n \times d_k}$$ 
$$K = XW^K \in \mathbb{R}^{n \times d_k}$$ 
$$V = XW^V \in \mathbb{R}^{n \times d_v}$$ 
Here, $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, and $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

*Score Calculation*
To compute how each token interacts with all other tokens, we perform a matrix multiplication between the Queries and the transpose of the Keys: 
$$S = QK^T \in \mathbb{R}^{n \times n}$$ 
An element $S_{i, j}$ represents the raw alignment score between the query of token $i$ and the key of token $j$.

*Why Dot-Products Scale with $\sqrt{d_k}$*
The authors show that for large values of $d_k$, dot products grow extremely large in magnitude. To understand why, assume that the components of $q \in \mathbb{R}^{d_k}$ and $k \in \mathbb{R}^{d_k}$ are independent random variables with a mean of 0 and a variance of 1. The dot product is defined as: 
$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$ 
Because the components are independent, their variances add up: 
$$\mathrm{Var}(q \cdot k) = \sum_{i=1}^{d_k} \mathrm{Var}(q_i k_i) = d_k$$ 
Thus, the standard deviation (the typical scale of the raw scores) is $\sqrt{d_k}$. Without scaling, scores are spread out widely. Feeding highly spread-out values into a softmax function causes it to output a probability distribution that is extremely peaked (where the highest value is close to 1.0 and all other values are close to 0.0). In these regions, the gradient of the softmax function is nearly zero, which prevents effective backpropagation and stops the model from learning. Dividing by $\sqrt{d_k}$ scales the variance back to 1, keeping the gradient signal healthy.

*Value Combination*
The normalized attention weights are used to compute a weighted sum of the values: $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \in \mathbb{R}^{n \times d_v}$$

**2. Multi-Head Dynamics & Subspace Representation**
Rather than computing a single attention output with $d_{\text{model}}$-dimensional vectors, the Transformer splits the computation across $h$ parallel heads.
*Tensor Dimension Path*
To implement this efficiently in deep learning frameworks, we use the following tensor dimensions:
* Input Sequence: $X$ of shape (Batch_Size, Seq_Len, d_model).
* Projection: We project $X$ using a single unified linear layer for each of $Q, K,$ and $V$. For queries, we multiply by $W^Q$ to get a shape of (Batch_Size, Seq_Len, h * d_k).
* Reshape: We reshape the tensor to isolate the attention heads: (Batch_Size, Seq_Len, h, d_k).
* Transpose: We transpose dimensions 1 and 2 to get a shape of (Batch_Size, h, Seq_Len, d_k). This puts the head dimension on the outside, allowing us to perform batch matrix multiplications in parallel across all heads and batches simultaneously.
* Attention Output: We calculate scaled dot-product attention on these tensors, resulting in an output of shape (Batch_Size, h, Seq_Len, d_v).
* Re-assembly: We transpose the output back to (Batch_Size, Seq_Len, h, d_v), flatten the last two dimensions to concatenate the heads back to (Batch_Size, Seq_Len, h * d_v), and finally project it using $W^O$ to return to (Batch_Size, Seq_Len, d_model).

*The Power of Different Representation Subspaces*
In a single-head attention setup, all tokens are forced to attend to each other within the same space. If a token needs to capture both its syntactic subject (e.g., three words back) and its semantic pronoun reference (e.g., eight words forward), a single-head system is forced to average these attention distributions. This averaging dilutes both context signals.
Multi-head attention prevents this by using different projection parameter matrices $W_i^Q, W_i^K,$ and $W_i^V$ for each head $i$. This allows the model to project the exact same input tokens into different representation subspaces.
* Head 1 can project the tokens into a subspace where the key-query match excels at identifying positional structures (like attending to the immediate next token).
* Head 2 can project the tokens into a subspace where the key-query match is fine-tuned to capture linguistic attributes (like matching pronouns to nouns).
* Head 3 can focus on verb-argument structures.
By keeping these coordinate spaces separate during the attention step, the model preserves distinct contextual signals, merging them only at the very end when the concatenated head representations are mixed by the final output projection matrix $W^O$
