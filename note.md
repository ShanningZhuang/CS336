## Lecture 1

Of course. Here is a preview for your lecture session based on the materials provided.

### Summary
Today's session introduces the rationale for building language models from scratch: to gain a fundamental understanding in an era where models are increasingly large, proprietary, and abstracted away. The lecture argues that "efficiency"—achieving maximum performance for a given resource budget—is the core principle driving modern LM development. It provides a historical overview of LMs, from early neural networks to today's frontier models, and outlines the course's structure. The session then dives into the first key technical component: tokenization. It contrasts different strategies (character, byte, word-based) and details the widely-used Byte-Pair Encoding (BPE) algorithm, which intelligently creates a vocabulary by merging frequent byte sequences from a training corpus.

### Key Concepts
-   **Building from Scratch**: The course philosophy of implementing models from the ground up to gain deep, transferable insights into their mechanics and design.
-   **Efficiency as a Core Principle**: The central idea that maximizing performance for a given data and compute budget drives almost all design decisions in the LM pipeline.
-   **Tokenization**: The crucial first step in any language model pipeline that converts raw text strings into sequences of integer tokens.
-   **Byte-Pair Encoding (BPE)**: A data-driven algorithm that trains a tokenizer by starting with single bytes and iteratively merging the most frequent adjacent pairs.

### Guiding Questions
1.  **Why** is a deep, "from-scratch" understanding of model mechanics still critical for innovation when state-of-the-art progress seems dominated by massive-scale industrial efforts?
2.  **How** does the goal of maximizing "efficiency" create trade-offs and influence design decisions at every level, from tokenization to model architecture and training strategy?
3.  **What-if** we could build effective models that operate directly on raw bytes, bypassing tokenization entirely? What challenges would this solve, and what new ones might it create?
4.  **How** do the different methods of tokenization (character, word, BPE) represent different trade-offs between vocabulary size, sequence length, and the ability to handle rare or new words?

### Warm-up
**Q1**
In the Byte-Pair Encoding (BPE) training algorithm, what is the primary criterion for deciding which pair of tokens to merge into a new token at each step?
   (a) The alphabetical order of the token pair.
   (b) The length of the resulting merged token.
   (c) The frequency of the adjacent token pair in the corpus.
   (d) A pre-defined list of important subwords.

**A1**
(c) The frequency of the adjacent token pair in the corpus. BPE is a greedy algorithm that iteratively finds the most common adjacent pair of tokens and merges them.

**Q2**
Fill in the blanks: A pure byte-based tokenizer has a small, fixed vocabulary size of \_\_\_\_, but it produces very \_\_\_\_ token sequences, which is computationally inefficient for Transformer models.

**A2**
256, long.

**Q3**
You are training a BPE tokenizer on the text: `unusual_users`. The initial sequence of byte tokens is `['u', 'n', 'u', 's', 'u', 'a', 'l', '_', 'u', 's', 'e', 'r', 's']`. The most frequent adjacent pair is `('u', 's')`. What will the token sequence be after performing this single merge?

**A3**
The new sequence will be `['u', 'n', 'z', 'u', 'a', 'l', '_', 'z', 'e', 'r', 's']`, where `z` is the new token representing `us`.

Of course. Here is a review guide to help you consolidate and apply the concepts from the lecture.

### 1. Quick Quiz
**Q1**: The lecture states that today's researchers are becoming "disconnected" from the underlying technology. What is the course's proposed solution to this problem?
**A1**: The course's solution is "understanding via building." By implementing models from scratch, students can gain a fundamental understanding of the mechanics and mindset that transfer even to frontier-scale models.

**Q2**: What is the key difference between the "prefill" and "decode" phases of inference, and which one is typically "memory-bound"?
**A2**: In the **prefill** phase, the model processes the entire prompt at once, which is compute-bound. In the **decode** phase, the model generates one token at a time, which is memory-bound due to the need to constantly access the large KV cache.

**Q3**: According to the lecture, what is the core trade-off that Byte-Pair Encoding (BPE) is designed to solve, compared to pure word-based or byte-based tokenization?
**A3**: BPE solves the trade-off between vocabulary size and sequence length. Word-based tokenizers have huge vocabularies and struggle with new words, while byte-based tokenizers have tiny vocabularies but create excessively long sequences. BPE finds a middle ground.

**Q4**: What are the three levels of "openness" for language models described in the lecture?
**A4**: 
1.  **Closed models** (API access only, e.g., GPT-4o).
2.  **Open-weight models** (weights available, but not data or full training details, e.g., Llama).
3.  **Open-source models** (weights, data, and code are public, e.g., OLMo).

### 2. Concept Network
This network shows how the lecture's core ideas connect. `A → B` means A influences or leads to B. `A ↔ B` means they have a reciprocal or trade-off relationship.

-   **Core Goal: Efficiency** (Best model for a given compute/data budget)
    -   `Efficiency` → Drives all **Design Decisions**.
    -   `Efficiency` ↔ **Scaling Laws** (Scaling laws predict the most efficient allocation of compute).
        -   `Chinchilla's Law` → Recommends optimal `Model Size (N)` vs. `Data Size (D)`.
    -   `Efficiency` → Requires **Hardware-Aware Implementation** (Systems).
        -   `Hardware (GPU)` → Motivates **Custom Kernels** (Triton) & **Parallelism** (Data, Tensor, etc.).

-   **Design Decisions Pipeline**:
    -   **1. Data Curation & Processing** → Filters for high-quality data to avoid wasting compute.
    -   **2. Tokenization**
        -   `Tokenization` ↔ `Sequence Length` vs. `Vocabulary Size` (The core dilemma).
        -   `Byte-Pair Encoding (BPE)` → A practical solution that adapts vocabulary to the data.
    -   **3. Model Architecture**
        -   `Transformer` → Core building block.
        -   `Architectural Variants` (RoPE, SwiGLU, RMSNorm) → Often motivated by improving computational or memory `Efficiency`.
    -   **4. Training** (Optimizer, LR Schedule) → Fine-tunes the process of achieving model convergence efficiently.
    -   **5. Alignment** (SFT, DPO) → Makes the base model more useful, which can be seen as improving its sample `Efficiency` for desired tasks.

### 3. Typical Pitfalls
1.  **Confusing BPE Encoding with Training**: During BPE training, pair frequencies are repeatedly calculated to build the merge rules. However, when *encoding* a new string with a trained tokenizer, you simply apply the fixed, pre-learned merge rules in order—you don't re-calculate any frequencies.
2.  **Misinterpreting "The Bitter Lesson"**: A common mistake is thinking the lesson is "scale is all you need." The lecture clarifies the correct interpretation: *algorithms that scale efficiently* are what truly matter. Algorithmic improvements are even more critical at large scales where waste is prohibitively expensive.
3.  **Underestimating Data Processing**: The lecture shows a sample of raw Common Crawl data to emphasize that high-quality data is not a given. It's easy to think of training data as a clean text file, but in reality, enormous effort goes into cleaning, filtering, and deduplicating raw sources like webpages, a crucial step for efficient training.

### 4. Micro-Application
**Task:**
You are building a BPE tokenizer. Your training corpus is the simple string: `"low lower lowest"`. Perform the first two merge steps of the BPE training algorithm. Start from the initial sequence of UTF-8 byte tokens.

**Hints / Sample Solution:**
**Initial State:**
The string `"low lower lowest"` is first converted to bytes and then to a list of integer tokens. For simplicity, let's represent them as characters:
`['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'l', 'o', 'w', 'e', 's', 't']`

**Step 1: Find and perform the first merge.**
1.  **Count adjacent pairs**:
    -   `(l, o)`: 3 times
    -   `(o, w)`: 3 times
    -   `(w, ' ')`: 1 time
    -   `(' ', l)`: 2 times
    -   `(w, e)`: 2 times
    -   `(e, r)`: 1 time
    -   `(r, ' ')`: 1 time
    -   `(e, s)`: 1 time
    -   `(s, t)`: 1 time
2.  **Find the most frequent pair**: `(l, o)` and `(o, w)` are tied with 3 occurrences. Let's break the tie by choosing the one that appeared first: `(l, o)`.
3.  **Merge**: We create a new token, let's call it `Z`, to represent `lo`.
4.  **New sequence**:
    `['Z', 'w', ' ', 'Z', 'w', 'e', 'r', ' ', 'Z', 'w', 'e', 's', 't']`

**Step 2: Find and perform the second merge.**
1.  **Count new adjacent pairs**:
    -   `(Z, w)`: 3 times
    -   `(w, ' ')`: 1 time
    -   `(' ', Z)`: 2 times
    -   `(w, e)`: 2 times
    -   `(e, r)`: 1 time
    -   `(r, ' ')`: 1 time
    -   `(e, s)`: 1 time
    -   `(s, t)`: 1 time
2.  **Find the most frequent pair**: `(Z, w)` is the most frequent with 3 occurrences.
3.  **Merge**: We create a new token, `Y`, to represent `Zw` (which is `low`).
4.  **New sequence**:
    `['Y', ' ', 'Y', 'e', 'r', ' ', 'Y', 'e', 's', 't']`

After two steps, the tokenizer has learned to represent `"lo"` and `"low"` as single tokens.

Of course. Based on the lecture materials provided in `lecture_01.py`, here is a summary and comparison of the different tokenization algorithms discussed.

The lecture presents four main approaches to tokenization, framing them as a series of trade-offs between vocabulary size, sequence length, and the ability to handle any possible text.

| Algorithm | How It Works | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **1. Character Tokenizer** | Each unique Unicode character is mapped to its integer code point (e.g., `ord('a')`). | - Conceptually simple.<br>- Lossless: can represent any character exactly. | - **Huge Vocabulary**: ~150,000 possible tokens, which is inefficient.<br>- **Poor Compression**: One character becomes one token, leading to long sequences. |
| **2. Byte Tokenizer** | The string is encoded into bytes (using UTF-8), and each byte (an integer from 0-255) becomes a token. | - **Small, Fixed Vocabulary**: Exactly 256 tokens, which is very efficient.<br>- **Complete**: Can represent any text string without "unknown" tokens. | - **Terrible Compression**: The token sequence is as long as the number of bytes, leading to very long sequences that are difficult for models to process. |
| **3. Word Tokenizer** | The text is split into words using rules or regular expressions. Each unique word becomes a token. | - **Intuitive**: Aligns well with how humans process language.<br>- **Good Compression**: Sequences are short (one token per word). | - **Huge & Unbounded Vocabulary**: The number of words is massive.<br>- **Out-of-Vocabulary Problem**: Requires a special `<UNK>` token for new words, which loses information. |
| **4. Byte-Pair Encoding (BPE)** | **A hybrid, data-driven algorithm.**<br>1. **Initialize** with single-byte tokens.<br>2. **Iteratively merge** the most frequent adjacent pair of tokens in a training corpus into a single new token.<br>3. **Repeat** for a set number of merges to build the final vocabulary. | - **Best of Both Worlds**: Balances vocabulary size and sequence length for good compression.<br>- **No Unknown Tokens**: Can fall back to byte-level representation for any new word.<br>- **Adaptive**: Vocabulary is optimized for the specific data it was trained on. | - **It's a "necessary evil"**: Framed as a practical but potentially inelegant workaround for current model limitations.<br>- **Greedy Algorithm**: The merges are locally optimal at each step, not necessarily globally optimal for the entire corpus. |

The lecture also briefly mentions **Tokenizer-Free Approaches** (e.g., `byt5`, `megabyte`) which operate directly on bytes. These are described as promising but not yet scaled to the level of frontier models, positioning **BPE** as the dominant, practical solution used today.

## Lecture 2

Of course! Here is your course preview based on the provided materials.

### Summary
This session provides a bottom-up guide to the primitives of model training, with a strong emphasis on resource accounting for memory and compute. Starting with tensors, it explores the memory implications of data types like float32 and bfloat16. The lecture then dives into compute costs (FLOPs) for key operations like matrix multiplication and backpropagation, introducing concepts like Model FLOPs Utilization (MFU). You'll learn to construct PyTorch models, implement custom optimizers, and build a complete training loop. The session also covers practical best practices, including efficient data loading, checkpointing for fault tolerance, and mixed-precision training to balance speed and stability.

### Key Concepts
-   **Resource Accounting**: Quantifying the memory (bytes) and compute (FLOPs) required for training models.
-   **Floating-Point Precision**: Understanding trade-offs between `float32`, `bfloat16`, and `float16` for memory, speed, and numerical stability.
-   **Tensor Operations & Einops**: Manipulating tensors efficiently and using `einops` for readable, error-proof dimension handling.
-   **FLOPs Calculation**: Estimating the computational cost of forward and backward passes (e.g., ~6 × parameters × tokens).
-   **PyTorch `nn.Module`**: Building custom models by composing layers and managing parameters.
-   **Training Loop Components**: Implementing optimizers, data loaders, checkpointing, and mixed-precision training.

### Guiding Questions
1.  **Why** is `bfloat16` often preferred over `float16` in modern LLM training, despite both using 16 bits of memory per parameter?
2.  **How** does the "6 × parameters × tokens" rule of thumb for training FLOPs arise from the costs of the forward and backward passes?
3.  **What-if** you are training a model and the Model FLOPs Utilization (MFU) is very low? What could be the potential bottlenecks?
4.  **How** do design choices like optimizer (e.g., Adam vs. SGD) and model architecture affect the total memory footprint during training, beyond just parameter count?

### Warm-up
**Q1 (Fill-in-the-blank)**
A 10B parameter model is trained on 1T tokens. The total training compute is approximately \_\_\_\_\_\_ FLOPs.

**A1**
6e22 FLOPs. The lecture establishes that total FLOPs are roughly 6 × (number of parameters) × (number of tokens), so 6 × 10e9 × 1e12 = 6e22.

**Q2 (Multiple-Choice)**
You are using the Adam optimizer to train a model with 1 billion parameters using `float32` precision. Naively, how much GPU memory is required for just the parameters, gradients, and optimizer state (ignoring activations)?
a) ~4 GB
b) ~8 GB
c) ~12 GB
d) ~16 GB

**A2**
d) 16 GB. For each parameter, we store: 4 bytes for the parameter itself, 4 bytes for its gradient, and 8 bytes for Adam's optimizer state (4 for momentum, 4 for variance). Total = (4 + 4 + 8) bytes/param × 1e9 params = 16e9 bytes ≈ 16 GB.

**Q3 (Mini-code)**
You have a tensor `x` with shape `(batch, seq_len, hidden_dim)`. Write a single line of code using `einsum` to calculate the dot product attention scores between all pairs of vectors in the sequence, resulting in a tensor of shape `(batch, seq_len, seq_len)`.

**A3**
`scores = einsum(x, x, "b s1 d, b s2 d -> b s1 s2")`

Of course. Here is your active recall and transfer review.

### 1. Quick Quiz
**Q1:** According to the lecture's rule of thumb, the backward pass requires approximately how many times more FLOPs than the forward pass for a standard dense model?
<br>
**A1:** The backward pass requires roughly **twice** the FLOPs of the forward pass (4 * N * P vs. 2 * N * P, where N is tokens and P is parameters).

**Q2:** If you use `x.transpose(1, 0)` to create a new tensor `y` from `x`, does modifying an element in `x` (e.g., `x[0, 0] = 100`) also change `y`? Why or why not?
<br>
**A2:** **Yes**, `y` will be changed. `transpose()` creates a **view**, not a copy, of the original tensor. Both `x` and `y` point to the same underlying memory storage, so a change via one tensor is visible in the other.

**Q3:** What is the primary benefit of using `bfloat16` over `float16` for deep learning, given they both use 16 bits?
<br>
**A3:** `bfloat16` has the same **dynamic range** as `float32` (8 exponent bits), which prevents numerical underflow/overflow on very small or large numbers, providing more training stability than `float16`. The trade-off is lower precision (fewer mantissa bits).

**Q4:** When switching from an `SGD` optimizer to `AdaGrad`, what additional memory cost per parameter is introduced, and what is stored?
<br>
**A4:** `AdaGrad` adds optimizer state. For each parameter, it stores a running sum of the squares of its gradients. If using `float32`, this adds **4 bytes per parameter** to the memory footprint.

---

### 2. Concept Network
-   **Resource Accounting (Memory & FLOPs)** is the central theme that connects all concepts.
    -   It is driven by **Hardware** (`A100`/`H100` specs) and measured by **Model FLOPs Utilization (MFU)**.
-   **Memory Accounting** ↔ **Tensor `dtype`** (`float32`, `bfloat16`, `fp8`)
    -   Memory for a model is determined by the size of **Parameters**, **Gradients**, **Activations**, and **Optimizer State**.
    -   **Mixed Precision Training** is a technique to optimize this trade-off.
-   **Compute Accounting (FLOPs)** ↔ **Tensor Operations**
    -   Dominated by **Matrix Multiplication** (`@` or `einsum`).
    -   FLOPs for training ≈ **6 × #parameters × #tokens**, comprising the **Forward Pass** (~2NP) and **Backward Pass** (~4NP).
-   **Tensors** are the fundamental data structure.
    -   Their memory layout is defined by **Storage and Strides**.
    -   Manipulated via **Slicing/Views** (cheap) or **Copies** (expensive).
    -   Complex manipulations are made easier by **`einops`** (`rearrange`, `reduce`, `einsum`).
-   **`nn.Module`** is the building block for models.
    -   It contains **`nn.Parameter`** objects, which require careful **Initialization** to ensure training stability.
-   **Training Loop** combines everything:
    -   It iterates over batches from a **Data Loader**.
    -   Performs a forward pass through the **Model** to get a loss.
    -   Performs a backward pass (`loss.backward()`) to compute **Gradients**.
    -   Uses an **Optimizer** (`SGD`, `AdaGrad`) to update parameters.
    -   Relies on **Checkpointing** for fault tolerance.

---

### 3. Typical Pitfalls
1.  **Confusing FLOPs and FLOP/s:** Mistaking the total amount of computation (FLOPs) with the speed of computation (FLOP/s or FLOPS). The first is a quantity, the second is a rate.
2.  **Ignoring Activation Memory:** Forgetting that intermediate activations, especially in models with long sequence lengths or large batch sizes, consume significant GPU memory. Calculations that only account for parameters, gradients, and optimizer state will underestimate the true memory requirement.
3.  **Silent Bugs from Tensor Views:** Accidentally modifying a tensor when you thought you were working with a copy because an operation returned a view (like `transpose`, `view`, or slicing). This can lead to bugs that are very hard to trace. Always use `.clone()` or `.contiguous()` when you explicitly need a new copy.
4.  **Underestimating Backward Pass Cost:** Assuming the backward pass has the same computational cost as the forward pass. As the lecture details, it's approximately twice as expensive in FLOPs, which is critical for accurate training time estimates.

---

### 4. Micro-Application
**Task:**
You are asked to plan a fine-tuning run for a 7-billion-parameter language model on a single H100 GPU (80 GB VRAM). Your goal is to fine-tune it on a 20-billion-token dataset. The training will use mixed precision.

1.  **Memory Check:** You plan to use the AdamW optimizer. For memory efficiency, you'll store the model parameters and gradients in `bfloat16`. However, AdamW maintains a `float32` copy of the parameters for stable updates. Calculate the total memory required for the **model state** (parameters, gradients, optimizer state) and determine if it will fit on the H100.
2.  **Compute & Time Estimate:** Estimate the total FLOPs required for the entire fine-tuning run. Then, using the H100's `bfloat16` performance from the lecture and assuming a realistic Model FLOPs Utilization (MFU) of 50%, calculate the estimated training time in days.

**Hints / Sample Solution:**

1.  **Memory Calculation:**
    -   Parameters (`bfloat16`): 7e9 params × 2 bytes/param = 14 GB
    -   Gradients (`bfloat16`): 7e9 params × 2 bytes/param = 14 GB
    -   AdamW Optimizer State (momentum + variance, both `bfloat16`): 2 × 7e9 params × 2 bytes/param = 28 GB
    -   AdamW `float32` copy of parameters: 7e9 params × 4 bytes/param = 28 GB
    -   **Total State Memory:** 14 + 14 + 28 + 28 = **84 GB**.
    -   **Conclusion:** This will **not fit** on a single 80 GB H100. This calculation doesn't even include memory for activations, highlighting the need for memory-saving techniques like ZeRO.

2.  **Compute & Time Calculation:**
    -   **Total FLOPs:** 6 × #params × #tokens = 6 × 7e9 × 20e9 = 8.4e20 FLOPs.
    -   **H100 Performance (from lecture):** The peak `bfloat16` FLOP/s for a dense H100 is `1979e12 / 2 = 989.5e12` FLOP/s.
    -   **Effective FLOP/s:** Peak FLOP/s × MFU = 989.5e12 × 0.50 = 494.75e12 FLOP/s.
    -   **Total Time (seconds):** Total FLOPs / Effective FLOP/s = 8.4e20 / 494.75e12 ≈ 1.7e6 seconds.
    -   **Total Time (days):** 1.7e6 seconds / (60 sec/min × 60 min/hr × 24 hr/day) ≈ **19.7 days**.