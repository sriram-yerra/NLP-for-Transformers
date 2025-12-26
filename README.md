**Krish Naik: NLP**

1. Natural language Processing (Day1 - Day5)
2. Advanced NLP (Last 4 Days)
3. Reccurent Neural Network (Day6 - Day8)
4. Long Short Term Memory (Day9 - Day11)
5. Transformers Basics (Day12)

## 🧠 NLP vs RNN vs LSTM vs GRU

Understanding the difference between **NLP**, **RNN**, **LSTM**, and **GRU** is essential when working with sequence and language-based problems in Machine Learning and Deep Learning.

---

### 📘 Natural Language Processing (NLP)
**NLP** is a field of Artificial Intelligence focused on enabling machines to understand, interpret, and generate human language.

**Common tasks:**
- Tokenization
- Part-of-Speech tagging
- Named Entity Recognition
- Sentiment Analysis
- Machine Translation
- Text Summarization

**Techniques & models used:**  
BoW, TF-IDF, RNN, LSTM, GRU, Transformers (BERT, GPT, etc.)

➡️ *NLP defines the problem space.*

---

### 🔁 Recurrent Neural Network (RNN)
**RNN** is a type of neural network designed to handle sequential data by maintaining a hidden state that captures information from previous time steps.

**Key features:**
- Processes sequences step by step
- Shares weights across time steps
- Suitable for short-term dependencies

**Limitations:**
- Suffers from vanishing gradients
- Struggles with long sequences

➡️ *RNN is the basic sequence model.*

---

### 🧠 Long Short-Term Memory (LSTM)
**LSTM** is an advanced form of RNN that introduces a memory cell and gating mechanisms to better capture long-term dependencies.

**Key features:**
- Handles long sequences effectively
- Uses forget, input, and output gates
- Reduces vanishing gradient problem

**Trade-offs:**
- More parameters
- Slower training than vanilla RNN

➡️ *LSTM is RNN with long-term memory.*

---

### 🚪 Gated Recurrent Unit (GRU)
**GRU** is a simplified version of LSTM that combines memory and hidden states using fewer gates.

**Key features:**
- Uses update and reset gates
- Fewer parameters than LSTM
- Faster training with comparable performance

➡️ *GRU is a lightweight alternative to LSTM.*

---

### 📊 Comparison Summary

| Aspect | NLP | RNN | LSTM | GRU |
|--------|-----|-----|------|-----|
Type | Field | Model | Model | Model |
Purpose | Language tasks | Sequence modeling | Long-term memory | Efficient memory |
Handles long dependencies | — | ❌ Poor | ✅ Good | ✅ Good |
Gates | — | ❌ None | ✅ 3 gates | ✅ 2 gates |
Complexity | — | Low | High | Medium |
Speed | — | Fast | Slower | Faster |
Modern usage | — | Rare | Limited | Limited |

---

### 🔗 Relationship
- **NLP (Problem Domain)**
  - **Sequence Models**
    - **RNN**
      - LSTM
      - GRU
    - **Transformers (Attention-based)**

---

### 🚀 Note
While RNN, LSTM, and GRU laid the foundation for sequence modeling, modern NLP systems primarily rely on **Transformer architectures** built entirely on attention mechanisms.

### 🤖 Transformers (Attention-based Models)

**Transformers** are modern deep learning architectures designed for sequence modeling that rely entirely on **self-attention** instead of recurrent connections. They process all tokens in parallel, making them highly efficient and scalable for large datasets.

**Key points:**
- Use **self-attention** to capture long-range dependencies
- Enable **parallel processing** (faster than RNNs)
- Form the backbone of models like **BERT, GPT, T5, and LLaMA**
- Achieve state-of-the-art results in most NLP tasks

➡️ Transformers have become the standard architecture for modern NLP systems.

> **Transformers** are attention-based models that replace recurrence with self-attention, enabling faster and more effective learning of long-range dependencies in NLP.


