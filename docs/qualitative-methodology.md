# PPB Qualitative Methodology — Knowledge Accuracy vs RAGAS Faithfulness

PPB's Phase 6 (Answer Quality) reports a metric called
`knowledge_accuracy_mean`. **This is not the same as the metric called
"faithfulness" in [RAGAS](https://docs.ragas.io/).** This document
explains the distinction and why PPB chose the metric it did.

## Definitions

| Metric                     | What is verified against                                                                          | Question being asked                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **RAGAS Faithfulness**     | A _reference context_ supplied alongside the prompt (e.g. retrieved documents in a RAG pipeline). | "Is each claim in the answer entailed by the retrieved context?"                               |
| **PPB Knowledge Accuracy** | The _judge model's parametric knowledge_ (no reference context).                                  | "Is each claim in the answer consistent with what the judge believes is true about the world?" |

Both metrics share the same overall shape:

1. Extract the set of factual claims from the response.
2. Verify each claim independently.
3. Score = `n_claims_passing / n_claims_total`.

But the verification step differs fundamentally — RAGAS asks a
_grounding_ question, PPB asks a _knowledge_ question. A response that
fabricates a citation entirely is penalised by both, but a response
that disagrees with the retrieved context yet is true in the wider world
will be marked unfaithful by RAGAS while passing PPB's knowledge check.

## Why PPB measures Knowledge Accuracy

PPB's evaluation prompts come from
[`anon8231489123/ShareGPT_Vicuna_unfiltered`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) —
real, open-ended user questions with no associated reference documents.
Without a reference context, RAGAS-style faithfulness is undefined:
there is nothing to ground claims against. We therefore use the judge
model's parametric knowledge as the verification corpus and label the
metric accordingly.

This is intentionally an approximation. A weak judge will produce a
noisy `knowledge_accuracy` signal; PPB documents the judge model in
every result row so consumers can interpret scores accordingly. We
recommend running quantitatively similar models against the same judge
when comparing scores.

## Implications for the composite score

`quality_composite_score` (the headline answer-quality number) is the
mean of `knowledge_accuracy_mean`, `answer_relevancy_mean` and
`coherence_mean`. Because `knowledge_accuracy` is one of the three
inputs, the headline number inherits the same caveat: it is _not_
directly comparable to RAGAS leaderboards. It IS comparable across
PPB rows that share the same judge model and prompt cache hash.

## Further reading

- [RAGAS faithfulness documentation](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)
- [PPB README — Qualitative block schema](../README.md#qualitative-block-schema)
- Source: [`ppb_answer_quality.py`](../ppb_answer_quality.py)
