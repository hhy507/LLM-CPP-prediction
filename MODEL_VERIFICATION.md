# Model Verification Log — GPT-4o used for the direct-LLM CPP classification

This document provides re-checkable evidence that the direct cell-penetrating-peptide (CPP) classification baseline (added in response to Reviewer 5, follow-up to Comment 7) was run with **OpenAI GPT-4o**, accessed through OpenRouter.

- **Model id used in the experiment:** `openai/gpt-4o`
- **Platform:** OpenRouter (https://openrouter.ai); upstream provider: OpenAI / Azure OpenAI
- **Verification test sequence:** id 280, `MDAQTRRRERRAEKQAQWKAANGC` (ground-truth = CPP / 1)

## Real-time verification calls

Each `generation_id` can be re-checked by the account owner in the OpenRouter dashboard (Activity / Generation lookup).

| UTC timestamp | requested model | response `model` | provider | generation_id | CPP answer |
|---|---|---|---|---|---|
| 2026-06-09T03:41:15.945406+00:00 | `openai/gpt-4o` | `openai/gpt-4o` | Azure | `gen-1780976476-CVflDX8NRL8ZjfZePVdT` | 1 |
| 2026-06-09T03:45:44.391901+00:00 | `openai/gpt-4o` | `openai/gpt-4o` | OpenAI | `gen-1780976745-61iyX8YGQ6uCQ3XcmcHy` | 1 |
| 2026-06-09T03:45:45.957436+00:00 | `openai/gpt-4o` | `openai/gpt-4o` | OpenAI | `gen-1780976746-4FyaVWIV9nH058QjkaAs` | 1 |
| 2026-06-09T03:45:47.879626+00:00 | `openai/gpt-4o` | `openai/gpt-4o` | OpenAI | `gen-1780976748-gVghk6QybqWTsjiBxRm8` | 1 |
| 2026-06-09T03:45:49.446080+00:00 | `openai/gpt-4o-2024-11-20` | `openai/gpt-4o-2024-11-20` | OpenAI | `gen-1780976752-Jy4aOxJxWYrAFwQZD3bM` | 1 |

## Alias metadata (OpenRouter catalogue)

- **id:** openai/gpt-4o
- **canonical_slug:** openai/gpt-4o
- **name:** OpenAI: GPT-4o
- **created:** 1715558400
- **context_length:** 128000
- **created_human:** 2024-05-13

## Notes on the alias vs the dated snapshot

- `openai/gpt-4o` is OpenRouter's **rolling alias** for OpenAI GPT-4o. Its `canonical_slug` is `openai/gpt-4o` and the API does **not** expose the underlying dated snapshot, so the exact dated build resolved during the full 185-sequence run cannot be reconstructed after the fact.
- OpenRouter may route the alias to **OpenAI** or **Azure OpenAI**; both serve the OpenAI GPT-4o model (see the differing `provider` values above).
- The dated snapshot **`openai/gpt-4o-2024-11-20`** (provider OpenAI) is included as a reproducible anchor; it returns the same CPP classification on the test sequence.

## Statement

The direct-LLM CPP/non-CPP classification reported in the manuscript (Supplementary Table S10) was produced by querying **OpenAI GPT-4o** (model id `openai/gpt-4o`) and **DeepSeek-V3** (`deepseek-chat`) through their respective APIs, at temperature 0 with a single zero-shot prompt. The calls above confirm the GPT-4o identity and provider; the per-sequence predictions are in `llm_direct_classification_results.csv`.
