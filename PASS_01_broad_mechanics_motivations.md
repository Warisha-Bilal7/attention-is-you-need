**_Understanding the "why" before getting started on the math_**

**Key Objective:**
Understanding what problems the authors were solving

**Approach:**
Read only the Abstract, Introduction, Conclusion, and Section 3.1 (Model Architecture)

**Abstract:**
The introduction proposes a new architecture for dominant sequence transduction models: the Transformer. Transformer architecture is based on the attention mechanism which allows neural networks to dynamically weigh the importance of different input elements. 
It disposes the idea of recurrence and convolutions. Attention mechanisms are parallelization and require significantly less time for training.

Now let's discuss what recurrence and convolutions are:

1. Recurrence: 
Recurrence processes data sequentially, one elemental at a time. There is an internal "hidden state" that acts as memory and is updated after every step. It tracks order and history. It processes the first "word" and then passes information from that to the second "word" and so on. It struggles with long-term memory as the information fades over time (vanishing gradient problem), and it cannot be easily parallelized on GPUs since Step B depends on the completion of Step A. It is used primarily in NLPs, time-series forecasting, and speech recognition.

2. Convolutions:
Convolutions process data using spatial filter (kernels) that slide across an input to detect local features to extract structural patterns. While it is highly efficient and parallelizable, it suffers from a limited "receptive field". A standard filter can only see a small patch of data at  time and missed the global context. It is primarily used in computer vision, image generation, medical scan analysis and video processing

**Introduction: (written with NotebookLM)**
Before the introduction of transformers and use of attention mechanism, RNNs, LSTMs and GRUs were established as the SOTA approach of sequence modeling and transduction problems.
These models read text like a human does: one word at a time, from left to right. After processing each word, they updated the "hidden state" memory.
1.  Massive Problems to this approach
   This approach to computation created severe structural limitations:
  *   It was extremely slow: The computer had to process the first word to calculate the memory        state for word 2 and then from word 2 to word 3, it had to wait. This sequential                 dependency forced the multi-tasking GPU chips to work slowly in baby steps 
  *   It suffered from memory loss due to distance issues: If a sentence is long, a word at the        very beginning has to pass its meaning "hand to hand" through every single intermediate          word to reach the end. Along the way, the signal gets diluted or lost, making it                difficult for the AI to connect words that are far apart

2.  The Solution: "Attention is all you need"
    The "Transformer" model is introduced by the authors to solve these limitations. It throws       out recurrence entirely and relies on "Self-Attention" mechanism to capture how words relate     to each other.
    
4.  Why Self-Attention changes everything
    Self-Attention wired every word directly to every other word in a single hop.
    *   Instant speed: The words don't have to wait for each other so the computer can look at           the entire sentence all at once. This allows the training process to be highly                   parallelized and efficient to train on GPUs
    *   Perfect memory: Every word is directly connected so the distance between two positions           is reduced to a constant "one-hop" path making it easier for the AI to learn how far-            apart words relate to one another.

  **Section 3.1 Encoder and Decoder Stacks**
  1.  _Encoder:_ Reading and Understanding the Input: Reads the source sentence -> converts into         list of "context-rich" vectors. Each vector represents a word and its relationship to             all other words in the sentence.
      *   The Stack: Six Identical Layers:     
      *   The Sub-Layers: Two main sub-layers for each of the 6 main layers
                        * Multi-Head Self-Attention:    This is where each token in the                                   sentence looks at every other token to gather context (e.g., resolving                           what the pronoun "it" is referring to)
                          * Position-Wise Feed-Forward Network:    Once a token has gathered                                 clues from the rest of the sentence, this small fully connected                                  network processes that token's vector individually
          
  2 .  _Decoder: Writing the Translation_ Generates the target sentence one word at a time. More complex layout since it generates text step-by-step.
   *   The Stack: Six identical Layers
   *   The Sub-Layers: Three sub-layers
   *   1. Masked Self-Attention: The translation attends to each self but it is masked from looking into the future at words that have not been written yet. This ensures that during training, the model doesn't cheat by looking ahead at the answer
   *   2. Encoder-Decoder (Cross) Attention: This is where the magic of translation happens. The decoder reaches back and looks at the encoder's outputs. It matches what it is currently writing with the original meaning of the source sentence
   *    3. Position-Wise Feed-Forward Network: Just like in the encoder, this processes each vector individually before passing it up.

  3. _The Connective Tissue:_ Residuals and LayerNorm: To train a network that is 12 layers deep (6 encoder layers + 6 decoder layers) without the signal getting lost, the         authors wrap each sub-layer in two things:
      Residual Connections: The input to a sub-layer is added back to its output (x + Sublayer(x)). This means a layer only has to calculate a "correction" or minor               adjustment, giving gradients a direct highway to flow back during training.
      Layer Normalization: This rescales the vectors to ensure that the numbers running through the network remain stable and do not explode or vanish.

      Together, this is written as: LayerNorm(x + Sublayer(x))

  4. _Injecting Order: Positional Encodings_
     Because the self-attention mechanism processes all words at once in parallel, it is naturally permutation invariant (it treats the sentence like an unordered bag of         words). To fix this, the authors add Positional Encodings to the word embeddings at the very bottom of both stacks.
     They use a fixed pattern of sine and cosine waves at different frequencies. This acts like a unique "striped fingerprint" or timestamp stamped onto each word vector,         allowing the model to easily figure out which word came first, second, or third.
