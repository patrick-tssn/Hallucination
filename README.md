# Hallucination

Learning Materials for Hallucination in Generative Models

<!-- ![image](./assets/cls.png) -->

<p align="center">
  <img src="./assets/cls.png" width="400" href="https://arxiv.org/pdf/2109.09784.pdf">
  <!-- <em>Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization</em> -->
</p>

## Table of Contents

- [Language](#language)
  - [Survey](#survey)
  - [Paper List](#paper-list)
- [Vision](#vision)
  - [Paper List](#paper-list-1)
  - [Benchmarks](#benchmarks)
- [References](#references)

## Language

### Survey

* Survey of Hallucination in Natural Language Generation, 2022

  - <details><summary>summary</summary>

    * Metrics (sec4):
      * Statistical: PARENT, PARENT-T1, Knowledge F1, BVSS
      * Model-based: IE-based, QA-based, NLI-based, Faithfulness Classification, LM-based
      * Human Evaluations
    * Method (sec5)
      * Data
      * Model:
        * architecture
        * training: (1) Planning/Sketching (2) RL (3) Multi-task Learning (4) Controllable Generation
    * Tasks: summarization, dialogue generation, Generative QA, Data-to-Text Generation, Neural Machine Translation, Vision-Language Generation

  </details>

### Paper List

* (2023.03) GPT4 Technical Report, *OpenAI*

  - <details><summary>summary</summary>

    - methods ( Sec2.2, Sec3.1):
      - Open-domain Hallucination (~extrinsic): flagged as not factual data + additional labeled comparison data -> reward model
      - Closed-domain Hallucination (~intrinsic):
        1. get the result: prompt -> response
        2. find hallucination: prompt + response -> hallucination
        3. modify the hallucination: prompt + response + hallucinations -> rewrite
        4. evaluate the hallucination: prompt + response -> hallucination ?: yes-> repeat; no -> get comparison pair -> reward model

  </details>

## Vision

### Paper List

### Benchmarks

| Benchmark                                                   | Task         | Data                                                       | Paper                                                                                                                               | Preprint                                    | Publication | Affiliation |
| ----------------------------------------------------------- | ------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ----------- | ----------- |
| [HALLUSIONBENCH](https://github.com/tianyi-lab/HallusionBench) | VQA (binary) | 1129 QA, 346 images, self-collect (multiple domain)        | HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination & Visual Illusion in Large Vision-Language Models | [2310.14566](https://arxiv.org/abs/2310.14566) |             | UMD         |
| [Bingo](https://github.com/gzcch/Bingo)                        | VQA          | 370 QA, 308 images, self-collect (object-hallucination)    | Holistic Analysis of Hallucination in GPT-4V(ision): Bias and Interference Challenges                                               | [2311.03287](https://arxiv.org/abs/2311.03287) |             | UNC         |
| [GAVIE](https://github.com/FuxiaoLiu/LRV-Instruction)          | VQA          | 1000 QA, 1000 images, Visual Genome (object-hallucination) |    Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning                                                                                                                                 |                    [2306.14565](https://arxiv.org/abs/2306.14565)                         |       ICLR 2024      |    Microsoft         |
| [POPE](https://github.com/RUCAIBox/POPE)                       | VQA (binary) | 3000 QA, 500 images, MSCOCO (object-hallucination)         | POPE: Polling-based Object Probing Evaluation for Object Hallucination                                                              | [2305.10355](https://arxiv.org/abs/2305.10355) | EMNLP 2023  | RUC         |


## References

- [awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/EdinburghNLP/awesome-hallucination-detection?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/EdinburghNLP/awesome-hallucination-detection.svg?style=social&label=Star), List of papers on hallucination detection in LLMs.
- [Large MultiModal Model Hallucination](https://github.com/xieyuquanxx/awesome-Large-MultiModal-Hallucination) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/xieyuquanxx/awesome-Large-MultiModal-Hallucination?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/xieyuquanxx/awesome-Large-MultiModal-Hallucination.svg?style=social&label=Star), up-to-date & curated list of awesome LMM hallucinations papers, methods & resources.
- [Awesome MLLM Hallucination](https://github.com/showlab/Awesome-MLLM-Hallucination) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/showlab/Awesome-MLLM-Hallucination?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/showlab/Awesome-MLLM-Hallucination.svg?style=social&label=Star), A curated list of resources dedicated to hallucination of multimodal large language models (MLLM).
