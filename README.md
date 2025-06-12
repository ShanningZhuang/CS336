# CS336: Language Modeling from Scratch

**Course Website:** https://stanford-cs336.github.io/spring2025/  
**YouTube Playlist:** https://www.youtube.com/watch?v=Rvppog1HZJY&list=PLZ959LONyLHz6W_4zcgkGXXNin7nSRnXO

## Course Description

Language models are the foundation of modern NLP applications, enabling a single system to tackle a wide range of tasks. This course provides a deep dive into language models by guiding students through the entire development process from scratch. We will cover every aspect of creating a language model, including:

- Data collection and cleaning for pre-training
- Building the Transformer model architecture
- Model training and optimization
- Evaluation and deployment strategies

This is a 5-unit, implementation-heavy course, so be prepared to write a significant amount of code.

## Prerequisites

- **Proficiency in Python:** Assignments require extensive coding with minimal scaffolding.
- **Deep Learning & Systems Optimization:** Strong familiarity with PyTorch and concepts like the memory hierarchy is expected.
- **Mathematics:** College-level Calculus and Linear Algebra (e.g., MATH 51, CME 100).
- **Probability and Statistics:** Basic knowledge is required (e.g., CS 109).
- **Machine Learning:** Familiarity with ML and deep learning basics (e.g., CS221, CS229, CS230, CS124, CS224N).

## Assignments

The course is structured around five main assignments:

- **Assignment 1: Basics:** Implement the core components (tokenizer, model architecture, optimizer) to train a standard Transformer.
- **Assignment 2: Systems:** Profile, benchmark, and optimize the model using tools like Triton for a custom FlashAttention2 implementation. Build a memory-efficient, distributed training system.
- **Assignment 3: Scaling:** Understand Transformer components and use a training API to fit a scaling law.
- **Assignment 4: Data:** Process raw Common Crawl dumps into high-quality pretraining data through filtering and deduplication.
- **Assignment 5: Alignment and Reasoning RL:** Apply supervised finetuning (SFT) and reinforcement learning (RLHF) to improve model reasoning. An optional part covers safety alignment methods like DPO.

## Schedule & Course Materials

| Lecture | Date         | Topic                                        | Materials                                                                                                                                                             |
|:-------:|:-------------|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1       | Tues April 1 | Overview, tokenization                       | [lecture_01.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F01.json)                                                              |
| 2       | Thurs April 3| PyTorch, resource accounting                 | [lecture_02.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F02.json)                                                              |
| 3       | Tues April 8 | Architectures, hyperparameters               | [lecture 3.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) |
| 4       | Thurs April 10| Mixture of experts                           | [lecture 4.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf)       |
| 5       | Tues April 15| GPUs                                         | [lecture 5.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf)                                         |
| 6       | Thurs April 17| Kernels, Triton                              | [lecture_06.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F06.json)                                                              |
| 7       | Tues April 22| Parallelism                                  | [lecture 7.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) |
| 8       | Thurs April 24| Parallelism                                  | [lecture_08.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F08.json)                                                              |
| 9       | Tues April 29| Scaling laws                                 | [lecture 9.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) |
| 10      | Thurs May 1  | Inference                                    | [lecture_10.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F10.json)                                                              |
| 11      | Tues May 6   | Scaling laws                                 | [lecture 11.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) |
| 12      | Thurs May 8  | Evaluation                                   | [lecture_12.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F12.json)                                                              |
| 13      | Tues May 13  | Data                                         | [lecture_13.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F13.json)                                                              |
| 14      | Thurs May 15 | Data                                         | [lecture_14.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F14.json)                                                              |
| 15      | Tues May 20  | Alignment - SFT/RLHF                         | [lecture 15.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) |
| 16      | Thurs May 22 | Alignment - RL                               | [lecture 16.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf)       |
| 17      | Tues May 27  | Alignment - RL                               | [lecture_17.py](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture%5F17.json)                                                              |
| 18      | Thurs May 29 | Guest Lecture by Junyang Lin                 |                                                                                                                                                                       |
| 19      | Tues June 3  | Guest lecture by Mike Lewis                  |                                                                                                                                                                       |

*For a full list of assignment deadlines and links, please refer to the [official course website](https://stanford-cs336.github.io/spring2025/).*

## Logistics

- **Lectures:** Tuesday/Thursday 3:00-4:20pm in **NVIDIA Auditorium**
- **Contact**: For course-related questions, use public Slack channels. For personal matters, email `cs336-spr2425-staff@lists.stanford.edu`.

### Office Hours
- **Tatsu Hashimoto (Gates 364):** Fridays at 3-4pm
- **Percy Liang (Gates 350):** Fridays at 11am-12pm
- **Marcel Rød (Gates 415):** Mondays 11am-12pm, Wednesdays 11am-12pm
- **Neil Band (Gates 358):** Mondays 4-5pm, Tuesdays 5-6pm
- **Rohith Kuditipudi (Gates 358):** Mondays 10-11am, Wednesdays 10-11am

*This README is based on the Spring 2025 offering of the course and was generated with information from the [course website](https://stanford-cs336.github.io/spring2025/).*
