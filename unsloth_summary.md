# Unsloth: Lesson Summary (Detailed)

**Subject:** Advanced Programming Language
**Topic:** Fine-tuning large language models (LLMs) with Unsloth

> **How to read this file**
> Sections marked **From the slides** come directly from the lesson.
> Sections marked **Extra explanation** are added background to make the topic easier to understand.

---

## 0. Background: key terms (Extra explanation)

| Term | Simple meaning |
|---|---|
| **LLM** (Large Language Model) | An AI model trained on huge text data, e.g. Llama, Mistral, Gemma |
| **Fine-tuning** | Taking a model that is already trained and training it a little more on your own data, so it gets good at a specific task |
| **GPU** | Graphics card. Much faster than CPU for training AI models |
| **VRAM / memory** | GPU memory. Big models need a lot, which is the main problem in fine-tuning |
| **LoRA** | A trick that trains only a few small extra layers instead of the whole model, saving time and memory |

**Why fine-tune?** A general model knows a little about everything. After fine-tuning on, for example, medical or legal text, it answers that field much better.

---

## 1. Introduction to Unsloth (From the slides)

Unsloth is a platform designed to make fine-tuning of large language models easier and more efficient. It supports models such as **Llama 3, Mistral, Phi-4, and Gemma**.

**What is Unsloth?**
- A Python library that helps fine-tune LLMs efficiently
- Supports Llama 3, Mistral, Phi-4, Gemma
- Reduces memory usage by up to **70%** while keeping accuracy

**Why use Unsloth?**

| Advantage | Meaning |
|---|---|
| Faster training | Optimized for efficiency |
| Less memory usage | Works on devices with limited GPU power |
| High accuracy | Achieves state-of-the-art fine-tuning results |

**In simple words:** Normally you need an expensive GPU to fine-tune an LLM. With Unsloth, you can do it on a smaller GPU (even free Google Colab or Kaggle), faster, with almost the same quality.

---

## 2. Installation Requirements (From the slides)

Before installing Unsloth, check these 4 things:

| # | Requirement | Details |
|---|---|---|
| 1 | **Operating system** | Works on Linux, Windows, macOS, Google Colab, Kaggle |
| 2 | **Python version** | Python **3.9 or later** |
| 3 | **GPU** (optional but recommended) | NVIDIA CUDA GPUs give faster performance. CPU also works but is slower |
| 4 | **pip installed** | Make sure pip (Python package manager) is up to date |

Update pip first:

```bash
pip install --upgrade pip
```

**Install Unsloth (Extra explanation):**

```bash
pip install unsloth
```

> The exact install command can differ by system (CUDA version, Colab, Windows).
> Check the official Unsloth documentation for your setup.

---

## 3. Basic workflow of fine-tuning (Extra explanation)

This is the usual order of steps when using Unsloth:

1. **Install** Unsloth
2. **Load** a pre-trained model (e.g. Llama 3), often in 4-bit to save memory
3. **Add LoRA settings** (rank, alpha, which layers to train). This is the code in the next section
4. **Prepare your dataset** (the data for your specific task)
5. **Train** the model (commonly with `SFTTrainer`)
6. **Test** the model and **save** it

---

## 4. Code snippet: configuring the model (From the slides)

The slide shows how to configure a **vision model** for fine-tuning with Unsloth.

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    use_rslora=False,
)
```

> The snippet is simplified. Other arguments on the slide (e.g. `lora_dropout`, `bias`, `random_state`) were too small to read in the photo.

### 4.1 Which parts of the model to train

All four options are `True` by default.

| Option | What it means |
|---|---|
| `finetune_vision_layers` | Train the layers that understand **images** |
| `finetune_language_layers` | Train the layers that understand **text** |
| `finetune_attention_modules` | Train the **attention** parts (how the model decides which words or image parts matter) |
| `finetune_mlp_modules` | Train the **MLP** parts (feed-forward layers that process the information) |

Set an option to `False` to skip that part. This saves memory and time.

### 4.2 LoRA parameters

| Parameter | Value | Meaning |
|---|---|---|
| `r` | 16 | **Rank**: size of the LoRA adapter. Bigger r = learns more detail but uses more memory. Common values: 8, 16, 32, 64 |
| `lora_alpha` | 16 | **Scaling factor** for how strongly the LoRA update affects the model. The slide recommends 16 (same as r) |
| `use_rslora` | False | **Rank-Stabilized LoRA**. Set `True` to turn on a version that is more stable at high ranks |

**Simple analogy:** The big model is a thick textbook. Instead of rewriting the whole book (full fine-tuning), LoRA adds small sticky notes (adapters). Only the sticky notes are trained, so it is much cheaper. `r` is how big the sticky notes are.

---

## 5. Benefits (From the slides)

- **Task-specific adaptation:** the model becomes more accurate in a specialized area
- **Resource efficiency:** optimized training reduces GPU and memory usage
- **Improved performance:** output is more relevant and precise

---

## Key Takeaways

- Unsloth = faster, lighter LLM fine-tuning (up to 70% less memory)
- Needs Python 3.9+; NVIDIA GPU recommended, CPU possible but slow
- Works with Llama 3, Mistral, Phi-4, Gemma
- LoRA trains only small adapters, not the whole model
- Main settings to remember: `r=16`, `lora_alpha=16`, `use_rslora=False`
- You choose which parts to train: vision, language, attention, MLP

---

## Quick review questions

1. What problem does Unsloth solve?
2. What is the minimum Python version?
3. Is a GPU required? Why is it recommended?
4. What does `r` control in LoRA?
5. What is the difference between attention modules and MLP modules?
