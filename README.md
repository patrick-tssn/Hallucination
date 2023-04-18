# Hallucination

A reading list of hallucination in Generative Models

## Language

* GPT4 Technical Report, 2023
  - <details><summary>summary</summary>

    - methods ( Sec2.2, Sec3.1):
      - Open-domain Hallucination (~extrinsic): flagged as not factual data + additional labeled comparison data -> reward model
      - Closed-domain Hallucination (~intrinsic):
        1. get the result: prompt -> response
        2. find hallucination: prompt + response -> hallucination
        3. modify the hallucination: prompt + response + hallucinations -> rewrite
        4. evaluate the hallucination: prompt + response -> hallucination ?: yes-> repeat; no -> get comparison pair -> reward model

  </details>

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

## Vision
