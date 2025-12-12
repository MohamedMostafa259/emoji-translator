# 🎭 BART Emoji Translator

> **Turn your text into emojis!**  
> A fine-tuned BART-Large model that translates English text into expressive emoji sequences using Curriculum Learning and LoRA.

![Emoji Translator Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-orange) 
![Model](https://img.shields.io/badge/Model-BART%20Large-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

![App Screenshot](https://raw.githubusercontent.com/MohamedMostafa259/emoji-translator/refs/heads/main/app-screenshot.png)

## Overview

The **Emoji Translator** is a sequence-to-sequence model designed to "translate" English sentences into appropriate emoji representations. Unlike simple keyword matching, this model understands context and sentiment to generate meaningful emoji sequences.

It was fine-tuned using **Curriculum Learning**, starting from simple concept-to-emoji mappings and progressively moving to complex, multi-sentence emotional contexts.

### Key Features
- **Base Model**: `facebook/bart-large`
- **Technique**: Low-Rank Adaptation (LoRA) for efficient fine-tuning.
- **Training Strategy**: 6-Stage Curriculum Learning with memory retention (preventing catastrophic forgetting).
- **Dataset**: Custom dataset generated with **Gemini 3 Pro**, ensuring high-quality and diverse examples.

## Links

- **Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/mohamedmostafa259/emoji-translator-demo)
- **Model Weights**: [HuggingFace Model Hub](https://huggingface.co/mohamedmostafa259/bart-emoji-translator)
- **Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/mohamedmostafa259/english-to-emoji)
- **Training Kernel**: [Kaggle Notebook](https://www.kaggle.com/code/mohamedmostafa259/emoji-translator-curriculum-learning)

## Infrastructure
- Trained on **Kaggle** using 2x T4 GPUs.
- **WandB** used for experiment tracking.
