---
layout: default
title: LLM Course – Week 2
permalink: /pages/llm_course/week2/
---

# Week 2: Core LLM Skills & Monitoring

<details>
<summary><strong>Day 8: Parameter-Efficient Fine-tuning (PEFT)</strong></summary>

### Morning: PEFT Fundamentals (4 hours)
1. **PEFT Theory & Overview** (2 hours)
   - Resource: [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)
   - Tasks:
     - Understand the motivation behind parameter-efficient methods
     - Learn the mathematical foundations of adapter-based methods
     - Compare traditional fine-tuning vs. PEFT approaches

2. **LoRA (Low-Rank Adaptation)** (2 hours)
   - Resource: [LoRA Paper](https://arxiv.org/abs/2106.09685) and [PEFT LoRA Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)
   - Tasks:
     - Understand low-rank decomposition principles
     - Learn LoRA hyperparameters and their effects
     - Implement basic LoRA with PEFT library

### Afternoon: Advanced PEFT Methods (4 hours)
1. **QLoRA & Other Methods** (2 hours)
   - Resource: [QLoRA Paper](https://arxiv.org/abs/2305.14314) and [Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
   - Tasks:
     - Understand quantization-aware fine-tuning
     - Compare LoRA, QLoRA, and AdaLoRA
     - Learn about prefix tuning and prompt tuning approaches

2. **Hands-on PEFT Project** (2 hours)
   - Resource: [PEFT Examples](https://github.com/huggingface/peft/tree/main/examples)
   - Tasks:
     - Fine-tune a 7B+ parameter model on limited hardware
     - Experiment with different LoRA ranks and configurations
     - Measure performance, memory usage, and training time

</details>

<details>
<summary><strong>Day 9: Instruction Tuning & RLHF</strong></summary>

### Morning: Instruction Tuning (4 hours)
1. **Instruction Tuning Foundations** (2 hours)
   - Resource: [InstructGPT Paper](https://arxiv.org/abs/2203.02155) and [Finetuning LLMs Guide](https://huggingface.co/learn/nlp-course/chapter7/6)
   - Tasks:
     - Understand the difference between pre-training and instruction tuning
     - Learn about instruction dataset creation and curation
     - Study prompt engineering for instruction datasets

2. **Instruction Dataset Preparation** (2 hours)
   - Resource: [TRL Documentation](https://huggingface.co/docs/trl/index) and [Stanford Alpaca Approach](https://github.com/tatsu-lab/stanford_alpaca)
   - Tasks:
     - Explore existing instruction datasets (Alpaca, Dolly, etc.)
     - Understand data formatting for instruction tuning
     - Create a small custom instruction dataset

### Afternoon: RLHF & Advanced Alignment (4 hours)
1. **RLHF Theory & Components** (2 hours)
   - Resource: [TRL RLHF Documentation](https://huggingface.co/docs/trl/main/en/rlhf_tutorial) and [OpenAI's RLHF Approach](https://openai.com/research/learning-from-human-preferences)
   - Tasks:
     - Understand the components of RLHF (reward model, PPO)
     - Learn about preference data collection
     - Study the limitations and challenges of RLHF

2. **Simplified RLHF Implementation** (2 hours)
   - Resource: [TRL Examples](https://github.com/huggingface/trl/tree/main/examples)
   - Tasks:
     - Set up a simplified RLHF pipeline using TRL
     - Train a basic reward model
     - Understand how to use PPO for language model fine-tuning

### Additional Resources
- [PEFT Library](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning methods
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) - For RLHF implementation
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - For 8-bit and 4-bit quantization
- [Fine-tuning LLMs Efficiently with PEFT and LoRA](https://www.philschmid.de/fine-tune-flan-t5-peft)
- [LoRA for Efficient LLM Fine-tuning](https://lightning.ai/pages/community/tutorial/lora-llm/)

</details>

<details>
<summary><strong>Day 10: Experiment Tracking with Weights & Biases</strong></summary>

### Morning: W&B Fundamentals (4 hours)
1. **W&B Setup & Basic Tracking** (2 hours)
   - Resource: [W&B Quickstart](https://docs.wandb.ai/quickstart)
   - Tasks:
     - Create W&B account and install library
     - Log metrics, hyperparameters, and artifacts
     - Create your first experiment dashboard

2. **Advanced Experiment Management** (2 hours)
   - Resource: [W&B for ML](https://docs.wandb.ai/guides)
   - Tasks:
     - Set up experiment sweeps for hyperparameter optimization
     - Use W&B with Hugging Face Trainer
     - Track model artifacts and datasets

### Afternoon: Integration with LLM Training (4 hours)
1. **LLM Fine-tuning with W&B** (2 hours)
   - Resource: [W&B + Hugging Face Integration](https://docs.wandb.ai/guides/integrations/huggingface)
   - Tasks:
     - Monitor fine-tuning experiments
     - Track GPU utilization and training metrics
     - Compare different model configurations

2. **Model Evaluation & Reporting** (2 hours)
   - Resource: [W&B Reports](https://docs.wandb.ai/guides/reports)
   - Tasks:
     - Create shareable experiment reports
     - Visualize model performance across different tasks
     - Set up automated model comparison workflows

</details>

<details>
<summary><strong>Day 11-12: Prompt Engineering</strong></summary>

### Day 11 Morning: Prompt Design Principles (4 hours)
1. **Fundamentals of Prompt Engineering** (2 hours)
   - Resource: [Prompt Engineering Guide](https://www.promptingguide.ai/)
   - Tasks:
     - Learn basic prompt structure and components
     - Understand zero-shot, one-shot, and few-shot prompting
     - Practice with different prompt formats

2. **Advanced Prompting Techniques** (2 hours)
   - Resource: [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
   - Tasks:
     - Implement chain-of-thought reasoning
     - Explore tree-of-thought prompting
     - Practice with step-by-step problem decomposition

### Day 11 Afternoon: System Prompts & Context Management (4 hours)
1. **System Prompt Design** (2 hours)
   - Resource: [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
   - Tasks:
     - Design effective system prompts for different use cases
     - Learn about role-based prompting
     - Practice context window management

2. **Prompt Optimization** (2 hours)
   - Resource: [Anthropic's Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
   - Tasks:
     - A/B test different prompt variations
     - Measure prompt effectiveness
     - Optimize for specific tasks and models

### Day 12: Advanced Prompting & Evaluation
1. **Multi-step Reasoning** (2 hours)
   - Resource: [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)
   - Tasks:
     - Implement reasoning and acting patterns
     - Create prompts for complex problem solving
     - Practice with tool-using prompts

2. **Prompt Evaluation & Testing** (2 hours)
   - Resource: [PromptBench](https://github.com/microsoft/promptbench)
   - Tasks:
     - Set up systematic prompt evaluation
     - Test prompt robustness and reliability
     - Create prompt testing frameworks

3. **Domain-Specific Prompting** (2 hours)
   - Tasks:
     - Design prompts for code generation
     - Create prompts for data analysis
     - Practice with creative writing prompts

4. **Prompt Security & Safety** (2 hours)
   - Resource: [Prompt Injection Guide](https://learnprompting.org/docs/prompt_hacking/injection)
   - Tasks:
     - Understand prompt injection vulnerabilities
     - Learn defensive prompting techniques
     - Practice secure prompt design

</details>

<details>
<summary><strong>Day 13-14: LLM Evaluation</strong></summary>

### Day 13: Evaluation Metrics & Methods

### Morning: Traditional NLP Metrics (4 hours)
1. **Core Evaluation Metrics** (2 hours)
   - Resource: [Hugging Face Evaluate Library](https://huggingface.co/docs/evaluate/index)
   - Tasks:
     - Understand perplexity and its applications
     - Learn BLEU, ROUGE, and METEOR for text generation
     - Practice with classification metrics (F1, accuracy, precision, recall)

2. **LLM-Specific Evaluation** (2 hours)
   - Resource: [LLM Evaluation Papers](https://arxiv.org/abs/2307.03109)
   - Tasks:
     - Learn about instruction-following evaluation
     - Understand helpfulness and harmlessness metrics
     - Explore factuality assessment methods

### Afternoon: Human Evaluation Approaches (4 hours)
1. **Human Evaluation Design** (2 hours)
   - Resource: [Human Evaluation for NLG](https://aclanthology.org/2020.inlg-1.23.pdf)
   - Tasks:
     - Design human evaluation protocols
     - Learn about inter-annotator agreement
     - Practice with crowd-sourcing evaluation

2. **Automated Evaluation with LLMs** (2 hours)
   - Resource: [GPT-4 as a Judge](https://arxiv.org/abs/2306.05685)
   - Tasks:
     - Use LLMs to evaluate other LLMs
     - Create automated evaluation pipelines
     - Compare automated vs. human evaluation

### Day 14: Advanced Evaluation & Benchmarking

### Morning: Comprehensive Benchmarking (4 hours)
1. **Standard Benchmarks** (2 hours)
   - Resource: [HELM Benchmark](https://crfm.stanford.edu/helm/latest/)
   - Tasks:
     - Run models on standard benchmarks (GLUE, SuperGLUE, etc.)
     - Understand benchmark limitations and biases
     - Compare model performance across different tasks

2. **Custom Evaluation Frameworks** (2 hours)
   - Resource: [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
   - Tasks:
     - Set up custom evaluation pipelines
     - Create task-specific evaluation metrics
     - Build reproducible evaluation workflows

### Afternoon: Evaluation Tools & Practices (4 hours)
1. **Evaluation Libraries & Tools** (2 hours)
   - Resource: [Ragas for RAG Evaluation](https://github.com/explodinggradients/ragas)
   - Tasks:
     - Use different evaluation libraries
     - Set up automated evaluation workflows
     - Practice with RAG-specific evaluation metrics

2. **Evaluation Best Practices** (2 hours)
   - Tasks:
     - Design comprehensive evaluation strategies
     - Learn about evaluation data contamination
     - Practice statistical significance testing for model comparison

</details>