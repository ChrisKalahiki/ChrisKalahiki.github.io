---
layout: default
title: LLM Course – Week 1
permalink: /pages/llm_course/week1/
---

# Week 1: Foundations & Core Concepts

<details>
<summary><strong>Day 1-2: Modern LLM Architecture</strong></summary>
<div markdown="1">

#### Essential Reading
- **"Attention Is All You Need" (Original Transformer Paper)**
  - Link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
  - The foundational paper that introduced transformer architecture

- **Andrej Karpathy's "The Illustrated Transformer"**
  - Link: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
  - Visual explanations that make transformer concepts much clearer

- **"The Annotated Transformer" by Harvard NLP**
  - Link: [https://nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
  - Implementation-focused explanation with code

#### Video Tutorials
- **Stanford CS224N: Natural Language Processing with Deep Learning**
  - Transformer lecture: [https://www.youtube.com/watch?v=ptuGllU5SQQ](https://www.youtube.com/watch?v=ptuGllU5SQQ)
  - Excellent academic explanation of transformers

- **Andrej Karpathy's "Let's build GPT"**
  - Link: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  - Step-by-step implementation of a mini GPT model

#### LLM Evolution & Architecture
- **"Language Models are Few-Shot Learners" (GPT-3 paper)**
  - Link: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
  - Understand how scale changed language models

- **Hugging Face's "How Transformers Work"**
  - Link: [https://huggingface.co/learn/nlp-course/chapter1/4](https://huggingface.co/learn/nlp-course/chapter1/4)
  - Part of their excellent NLP course

#### Interactive Explorations
- **Transformer Visualization**
  - Link: [https://transformer.huggingface.co/](https://transformer.huggingface.co/)
  - Interactive tool to visualize attention in transformers

#### Practical Components
- **Implement a simplified transformer from scratch**
  - Tutorial: [https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb](https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb)

- **Experiment with a pre-trained model using Hugging Face**
  - Link: [https://huggingface.co/docs/transformers/quicktour](https://huggingface.co/docs/transformers/quicktour)

</div>
</details>

<details>
<summary><strong>Day 3: PyTorch Basics</strong></summary>
<div markdown="1">

#### Morning: Core PyTorch Concepts (4 hours)
1. **Tensors & Operations** (2 hours)
   - Resource: [PyTorch Official Tutorial - Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
   - Practice: Create different tensor types, perform indexing, reshaping, and basic math operations
   - Exercises: Convert NumPy arrays to tensors and back, manipulate dimensions

2. **Autograd & Computational Graphs** (2 hours)
   - Resource: [PyTorch Autograd Explained](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
   - Practice: Create simple computations with `requires_grad=True`
   - Exercise: Visualize a simple computational graph and trace gradients backward

#### Afternoon: Building Blocks (4 hours)
1. **Neural Network Components** (2 hours)
   - Resource: [PyTorch nn Module Tutorial](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
   - Practice: Implement linear layers, activation functions, and loss functions
   - Exercise: Build a simple MLP architecture using nn.Module

2. **Optimizers & Training Loop** (2 hours)
   - Resource: [PyTorch Optimization Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
   - Practice: Implement SGD, Adam optimizers
   - Exercise: Write a complete training loop for a toy dataset

</div>
</details>

<details>
<summary><strong>Day 4: Applied PyTorch for NLP/LLMs</strong></summary>
<div markdown="1">

#### Morning: Working with Text Data (4 hours)
1. **Text Processing in PyTorch** (2 hours)
   - Resource: [Text Classification with Torchtext](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
   - Practice: Tokenization, vocabulary building, embedding lookup
   - Exercise: Create a custom text dataset and dataloader

2. **Embeddings & Language Model Basics** (2 hours)
   - Resource: [Word Embeddings in PyTorch](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
   - Practice: Work with pre-trained embeddings
   - Exercise: Implement a simple n-gram language model

#### Afternoon: Building a Simple Transformer (4 hours)
1. **Attention Mechanism Implementation** (2 hours)
   - Resource: [Implementing Transformer Components in PyTorch](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
   - Practice: Code a scaled dot-product attention module
   - Exercise: Add multi-head attention to your implementation

2. **Simple Transformer Model** (2 hours)
   - Resource: [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
   - Practice: Implement encoder and decoder blocks
   - Exercise: Train a tiny transformer on a small dataset

#### Additional Resources
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/) - Start here for fundamentals
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Reference when you need details
- [Learn PyTorch with Python Notebook](https://github.com/yunjey/pytorch-tutorial) - Collection of practical notebooks
- [PyTorch for Deep Learning](https://www.youtube.com/watch?v=GIsg-ZUy0MY) - Freecodecamp's comprehensive tutorial
- [PyTorch Examples](https://github.com/pytorch/examples) - Official example implementations

</div>
</details>

<details>
<summary><strong>Day 5: Transformers Library Fundamentals</strong></summary>
<div markdown="1">

#### Morning: Core Concepts & Models (4 hours)
1. **Introduction to Hugging Face Transformers** (1.5 hours)
   - Resource: [Hugging Face Course - Chapter 1](https://huggingface.co/learn/nlp-course/chapter1/1)
   - Tasks: 
     - Create a Hugging Face account
     - Explore the Model Hub interface
     - Understand the pipeline abstraction

2. **Using Pre-trained Models** (2.5 hours)
   - Resource: [Transformers Quicktour](https://huggingface.co/docs/transformers/quicktour)
   - Tasks:
     - Run inference with 3 different models (BERT, GPT-2, T5)
     - Compare outputs for classification, generation, and summarization
     - Practice using AutoModels and AutoTokenizers

#### Afternoon: Tokenizers Deep Dive (4 hours)
1. **Understanding Tokenization** (2 hours)
   - Resource: [Hugging Face Course - Tokenizers](https://huggingface.co/learn/nlp-course/chapter2/1)
   - Tasks:
     - Explore different tokenization strategies (BPE, WordPiece, SentencePiece)
     - Visualize tokenization results
     - Understand special tokens and their purpose

2. **Custom Tokenizer Usage** (2 hours)
   - Resource: [Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
   - Tasks:
     - Configure tokenizer parameters
     - Handle tokenizer edge cases
     - Practice batch encoding for efficiency

</div>
</details>

<details>
<summary><strong>Day 6: Datasets & Fine-tuning</strong></summary>
<div markdown="1">

#### Morning: Hugging Face Datasets (4 hours)
1. **Working with Datasets Library** (2 hours)
   - Resource: [Datasets Quickstart](https://huggingface.co/docs/datasets/quickstart)
   - Tasks:
     - Load and explore standard NLP datasets (GLUE, SQUAD)
     - Apply dataset transformations and filters
     - Create data processing pipelines

2. **Creating Custom Datasets** (2 hours)
   - Resource: [Custom Datasets Tutorial](https://huggingface.co/docs/datasets/dataset_script)
   - Tasks:
     - Convert your own data into Datasets format
     - Implement dataset streaming for large datasets
     - Practice dataset versioning and sharing

#### Afternoon: Basic Fine-tuning (4 hours)
1. **Fine-tuning for Classification** (2 hours)
   - Resource: [Fine-tuning a Pretrained Model](https://huggingface.co/docs/transformers/training)
   - Tasks:
     - Set up a text classification fine-tuning job
     - Use the Trainer API
     - Evaluate model performance

2. **Model Saving & Sharing** (2 hours)
   - Resource: [Model Sharing Tutorial](https://huggingface.co/docs/transformers/model_sharing)
   - Tasks:
     - Save and load fine-tuned models
     - Push models to the Hugging Face Hub
     - Document model cards effectively

</div>
</details>

<details>
<summary><strong>Day 7: Advanced Transformers Techniques</strong></summary>
<div markdown="1">

#### Morning: Efficient Fine-tuning (4 hours)
1. **Parameter-Efficient Methods** (2 hours)
   - Resource: [PEFT Library Documentation](https://huggingface.co/docs/peft/index)
   - Tasks:
     - Implement LoRA fine-tuning
     - Understand adapter configurations
     - Compare with full fine-tuning on performance and resource usage

2. **Hugging Face Accelerate** (2 hours)
   - Resource: [Accelerate Documentation](https://huggingface.co/docs/accelerate/index)
   - Tasks:
     - Scale training to multiple GPUs
     - Implement mixed precision training
     - Configure gradient accumulation

#### Afternoon: End-to-End Project (4 hours)
1. **Project Planning & Setup** (1 hour)
   - Choose a task: Question-answering, summarization, or classification
   - Select appropriate models and datasets
   - Define evaluation metrics

2. **Implementation & Training** (3 hours)
   - Build complete training pipeline
   - Implement custom evaluation
   - Apply optimization techniques

3. **Model Deployment & Demo** (2 hours)
   - Create a Gradio or Streamlit demo
   - Deploy to Hugging Face Spaces
   - Document your approach comprehensively

#### Additional Resources
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/index)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) - Comprehensive official tutorial
- ["Natural Language Processing with Transformers"](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) by Lewis Tunstall, Leandro von Werra, and Thomas Wolf
- [Hugging Face YouTube Channel](https://www.youtube.com/c/HuggingFace)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Model Hub](https://huggingface.co/models) - Explore state-of-the-art models

</div>
</details>
