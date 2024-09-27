---
layout: home
---


# ML@B Reading List
**Last Updated: June 2024**

# Table of Contents
1. [General Introduction to ML](#general-introduction-to-ml)
2. [ML Theory](#ml-theory)
3. [Computer Vision (CV)](#computer-vision-cv)
4. [Natural Language Processing (NLP)](#natural-language-processing-nlp)
5. [Reinforcement Learning (RL)](#reinforcement-learning-rl)
6. [Multimodality](#multimodality)
7. [Miscellaneous](#miscellaneous)
8. [Other Resources](#other-resources)
9. [ML Fundamentals](#ml-fundamentals)
10. [Robotics](#robotics)
11. [BioML](#bioml)
12. [Miscellaneous](#miscellaneous)
13. [Other Resouces](#other-resources)


# Introduction
Welcome to the Machine Learning at Berkeley reading list! This was assembled by students at UC Berkeley, and was designed to be a good reference for those in the intermediate stages of learning ML.

# Beginning Guide
The following papers give you a flavor of each of the sections, and don’t require much extra knowledge beyond basic deep learning concepts (you should know about MLPs/CNNs and how to train them).

### Transformer
[Attention Is All You Need](https://arxiv.org/abs/1706.03762): Original Paper

- This paper introduces the Transformer architecture, a fundamental architecture for modern NLP models like GPT, Llama, and BERT. It explains the mechanics of self-attention, a key component that allows these models to scale effectively and perform well on a variety of tasks.

### Convolutional Neural Networks (CNNs)
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf): Original Paper

- INSERT DESCRIPTION HERE

### Generative Adversarial Networks (GANs)
[Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- This paper introduces GANs, a novel framework for training models in which two neural networks contest with each other in a game, paving the way for significant developments in generative models that can produce highly realistic images and other data samples.

# ML Fundamentals




## Supervised Learning

## Generalization
#### MAML
[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

- Humans are much more efficient than neural networks: We only need a few examples (perhaps even one) of a given class to be able to reliably recognize it. One explanation for this is that we have "learned how to learn"; that is, we have seen many other objects in the real world, so we have an understanding of the general properties of objects. The study of "learning to learn" is termed meta-learning, and the MAML paper introduces a very simple approach for meta-learning: Just train the model to be easy to fine-tune.

# Computer Vision (CV)
## Object Detection / Segmentation
### 2-D Object Detection / Segmentation

#### YOLO
[You Only Look Once](https://arxiv.org/abs/1506.02640)
- This paper introduces YOLO, a real-time object detection system that can detect objects in images with high accuracy and speed. It frames object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.

#### U-Net
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- This paper introduces U-Net, a convolutional network architecture specifically designed for medical image segmentation. It is particularly effective due to its ability to work with very few training images and still produce precise segmentations.

### 3-D Object Detection / Segmentation
#### PointNet
[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- PointNet provides a novel deep neural network architecture that directly processes point clouds, which are the primary form of geometric data. Unlike previous methods, PointNet learns global features invariant to permutations, making it highly effective for 3D object detection and segmentation tasks.

#### Mesh R-CNN
[Mesh R-CNN](https://arxiv.org/abs/1906.02739)
- Mesh R-CNN extends the popular Region-based Convolutional Neural Network (R-CNN) to better handle 3D data by adding a mesh prediction branch. This approach allows for the generation of high-fidelity 3D object reconstructions from standard 2D images, significantly advancing capabilities in 3D object detection and segmentation.
 
# Natural Language Processing (NLP)
## Word Vectors / Embeddings
### Key Papers and Concepts

#### Word2vec
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Words that share common contexts in the corpus are located close to one another in the space.

#### GloVe
[Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- GloVe (Global Vectors) is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

#### FastText
[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
- FastText extends Word2vec model by treating each word as a bag of character n-grams. This enables the model to capture the morphology of words, making it particularly useful for languages with rich inflectional forms. It can also predict vectors for out-of-vocabulary words.

#### ELMo
[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
- ELMo (Embeddings from Language Models) introduces the concept of deep contextualized word representations. It differs from traditional word embeddings like Word2vec by considering the entire context in which a word appears. ELMo representations are functions of the entire input sentence, as opposed to static word embeddings, and are computed on top of two-layer bidirectional language models (biLMs) with character convolutions, which allows the model to incorporate both complex characteristics of word use and how these uses vary across linguistic contexts.

#### BERT Embeddings
[Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- BERT (Bidirectional Encoder Representations from Transformers) represents a new method of pre-training language representations which obtain state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. BERT models can consider the full context of a word by looking at the words that come before and after it—particularly useful for understanding the intent behind search queries.

## Sentence-embeddings
### Key Papers and Concepts

#### SBERT
[SBERT: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- SBERT modifies the BERT architecture to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This is achieved by using a siamese network structure to fine-tune BERT on sentence-pair tasks.

#### Universal Sentence Encoder
[Universal Sentence Encoder for English](https://arxiv.org/abs/1803.11175)
- The Universal Sentence Encoder provides sentence level embeddings by training on a variety of data sources and tasks. It is available in two versions: one trained with a Transformer encoder and the other with a deep averaging network (DAN).

#### LASER
[Language-Agnostic SEntence Representations](https://arxiv.org/abs/1812.10464)
- LASER is a toolkit for calculating and using multilingual sentence embeddings. It uses a single BiLSTM network that is trained on 93 languages, demonstrating strong transfer capabilities across languages.

#### Sentence-T5
[Sentence-T5: Exploring the Limits of Sentence Embeddings with Pre-trained Text-to-Text Models](https://arxiv.org/abs/2108.08877)
- Sentence-T5 adapts the T5 (Text-to-Text Transfer Transformer) model to generate sentence embeddings. It leverages the pre-trained capabilities of T5 and fine-tunes it on sentence similarity tasks to produce embeddings.

## Transformers

### Positional Encodings

#### RoPE
[Rotary Positional Embedding](https://arxiv.org/abs/2104.09864)
- Rotary Positional Embedding (RoPE) integrates absolute positional information with a rotation matrix and is particularly effective in Transformer models. It encodes the absolute position and relative distance of tokens in the sequence, enhancing the model's ability to capture positional relationships in data.

#### Sinusoidal Embeddings
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Introduced in the original Transformer paper, sinusoidal positional encodings use sine and cosine functions of different frequencies to inject relative positional information into the input embeddings. This method allows the model to easily learn to attend by relative positions, since for any fixed offset k, the positional encoding for position p+k can be represented as a linear function of the encoding at position p.

#### Learnable Positional Encodings
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Unlike fixed sinusoidal embeddings, learnable positional encodings are parameters that the model learns during training. This approach allows the model to adapt positional encodings based on the specific requirements of the task, potentially capturing more complex dependencies compared to fixed positional encodings.

#### ALiBi
[Attention with Linear Biases](https://arxiv.org/abs/2108.12409)
- ALiBi (Attention with Linear Biases) proposes a method to handle positional encodings by introducing biases directly into the attention logits. This approach scales linearly with sequence length and allows for efficient handling of longer sequences without the need for complex positional encoding schemes.

### Systems Approaches

#### Flash Attention
[Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Flash Attention introduces a novel approach to computing exact attention in Transformer models that is both fast and memory-efficient. By optimizing the input/output operations and memory access patterns, Flash Attention significantly reduces the computational overhead associated with traditional attention mechanisms, making it feasible to handle larger models and datasets without compromising on performance.

#### Reversible Layers
[Reversible Residual Network](https://arxiv.org/abs/1707.04585)
- Reversible layers offer a method to significantly reduce memory usage during training by allowing the exact reconstruction of input activations from outputs without needing to store intermediate activations. This technique, initially proposed for image processing tasks, has been adapted for use in deep learning models, particularly in systems where memory efficiency is crucial.

#### Sparse Transformers
[Sparse Transformer: Practical Sparse Attention with Sensible Priors](https://arxiv.org/abs/1904.10509)
- Sparse Transformers introduce a novel sparse attention mechanism that reduces the complexity of attention from quadratic to log-linear with respect to sequence length. By strategically limiting attention to a subset of key positions, Sparse Transformers maintain performance while greatly enhancing efficiency, making them suitable for tasks involving very long sequences.

#### Performer
[Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
- The Performer framework redefines attention mechanisms by using random feature maps to approximate the softmax kernel, enabling the scaling of Transformers to extremely long sequences without the quadratic dependency on memory and compute. This approach preserves the benefits of the Transformer architecture while addressing its scalability issues.

### Sub-quadratic attention
#### Linear Attention
[Efficient Attention: Attention with Linear Complexities](https://arxiv.org/abs/1812.01243)
<!-- TODO: Is this the right paper? -->
- Linear Attention reduces the computational complexity of traditional attention mechanisms from quadratic to linear by approximating the softmax function in the attention calculation. This approach allows for handling longer sequences with significantly reduced computational resources, making it practical for large-scale applications.

### Quantization

#### Parameter-efficient tuning
[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- Parameter-efficient tuning methods allow for adapting pre-trained models to new tasks without extensive retraining or modifying the underlying network architecture. These techniques, such as adapters and prompt tuning, provide a way to fine-tune models on specific tasks with minimal additional parameters, preserving the original model's integrity and reducing computational costs.

#### Adapters
[Parameter-Efficient Transfer Learning with Adapters](https://arxiv.org/abs/1902.00751)
- Adapters are small trainable modules inserted between the layers of a pre-trained model. They enable fine-tuning on specific tasks while keeping the majority of the model's parameters frozen. This approach is highly efficient in terms of parameter count and computational resources, making it ideal for deploying large models in resource-constrained environments.

#### Prefix-tuning
[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- Prefix-tuning involves prepending a sequence of continuous vectors (the prefix) to the input of each layer in a Transformer model. This method allows for task-specific adaptation while maintaining the pre-trained parameters unchanged, offering a balance between performance and efficiency in transfer learning scenarios.

#### Low Rank Adaptation (LoRA)
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Low Rank Adaptation (LoRA) projects the model's weights into a lower-dimensional space, where adaptations are made before projecting back to the original high-dimensional space. This technique reduces the number of trainable parameters significantly, facilitating efficient fine-tuning of large models on downstream tasks with limited data.

## Large Language Models
### Pretraining Objectives

#### Masked Language Modeling (MLM)
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Masked Language Modeling (MLM) is a pretraining objective where some percentage of the input tokens are masked at random, and the goal is to predict these masked tokens. This approach allows the model to learn a deep bidirectional representation and was popularized by BERT. It helps the model understand context and relationships between words in a sentence.

#### Causal Language Modeling (CLM)
[GPT: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- Causal Language Modeling (CLM) involves training a model to predict the next word in a sentence, given the previous words. This training objective encourages the model to learn the probability distribution of a word sequence in a language, fostering strong generative capabilities as demonstrated by the GPT series of models.

#### State Space Models (SSM)
[State Space Models for Sequence Data](https://arxiv.org/abs/2312.00752)
<!-- TODO: fill here -->

### Pretraining Datasets
### Instruction tuning / RLHF
#### Zero-shot generalization by fine tuning on many tasks
#### RLHF
#### PPO
#### DPO

### Prompt Engineering
#### Scratchpad
#### Chain of Thought (CoT)
#### Let's Think Step by Step

## Benchmarking
#### GLUE
#### SuperGLUE

#### TruthfulQA

#### MMLU

#### BigBench

## Scaling
#### Scaling Laws
#### Chinchilla
#### Llama

## Long Contexts

## Interpretability

## Risks
### Bias
### Jailbreaking
### Data poisoning
### Leaking memorized info

## NLP Tasks
### Question Answering
#### SQuAD
#### DrQA

### Sentiment Analysis
#### SST
#### VADER

### Parsing

### Summarization

### Translation

### Tagging



# Reinforcement Learning (RL)

# Multimodality

# Robotics

# BioML

# Miscellaneous

# Other Resources
