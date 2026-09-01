**_Understanding the "why" before getting started on the math_**

**Key Objective:**
Understanding what problems the authors were solving

**Approach:**
Read only the Abstract, Introduction, Conclusion, and Section 3 (Model Architecture)

**Abstract:**
The introduction proposes a new architecture for dominant sequence transduction models: the Transformer. Transformer architecture is based on the attention mechanism which allows neural networks to dynamically weigh the importance of different input elements. 
It disposes the idea of recurrence and convolutions. Attention mechanisms are parallelization and require significantly less time for training.

Now let's discuss what recurrence and convolutions are:

1. Recurrence: 
Recurrence processes data sequentially, one elemental at a time. There is an internal "hidden state" that acts as memory and is updated after every step. It tracks order and history. It processes the first "word" and then passes information from that to the second "word" and so on. It struggles with long-term memory as the information fades over time (vanishing gradient problem), and it cannot be easily parallelized on GPUs since Step B depends on the completion of Step A. It is used primarily in NLPs, time-series forecasting, and speech recognition.

2. Convolutions:
Convolutions process data using spatial filter (kernels) that slide across an input to detect local features to extract structural patterns. While it is highly efficient and parallelizable, it suffers from a limited "receptive field". A standard filter can only see a small patch of data at  time and missed the global context. It is primarily used in computer vision, image generation, medical scan analysis and video processing

**Introduction:**
