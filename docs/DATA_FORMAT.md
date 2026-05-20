# Training Data Format

## Contrastive Training Data (Pre-training Stage 2 / Fine-tuning)

Training data should be in **JSONL format** (one JSON object per line). Each line contains a query, positive passages, and hard negative passages.

### Basic Format (without knowledge distillation)

```jsonl
{"query": "What is the impact of climate change?", "pos": ["Climate change causes rising sea levels and extreme weather events."], "neg": ["The stock market closed higher today.", "A new restaurant opened downtown.", ...]}
{"query": "How do earthquakes occur?", "pos": ["Earthquakes result from the sudden release of energy in the Earth's crust."], "neg": ["The movie received positive reviews.", "Traffic was heavy during rush hour.", ...]}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | The query text |
| `pos` | `list[string]` | Positive (relevant) passages. One is randomly selected per training step. |
| `neg` | `list[string]` | Hard negative passages. `train_group_size - 1` are sampled per step. |

### With Knowledge Distillation

When using `--knowledge_distillation True`, each sample must also include teacher model scores:

```jsonl
{"query": "...", "pos": ["..."], "neg": ["...", "..."], "pos_scores": [0.95], "neg_scores": [0.12, 0.08, ...]}
```

| Field | Type | Description |
|-------|------|-------------|
| `pos_scores` | `list[float]` | Teacher model scores for each positive passage |
| `neg_scores` | `list[float]` | Teacher model scores for each negative passage |

### With Per-sample Instruction (optional)

Each sample can optionally provide its own query instruction via a `prompt` field:

```jsonl
{"query": "...", "pos": ["..."], "neg": ["..."], "prompt": "Given the question, retrieve the most relevant passage:"}
```

If `prompt` is absent, the global `--query_instruction_for_retrieval` is used.

### Data Organization

Training data can be provided in two ways:

1. **Individual JSONL files**: pass one or more file paths directly
   ```bash
   --train_data data/train_qa.jsonl data/train_nli.jsonl data/train_sts.jsonl
   ```

2. **Directory**: pass a directory path; all `.json` / `.jsonl` files inside will be loaded
   ```bash
   --train_data data/pretrain/
   ```

When using `--same_dataset_within_batch True`, each JSONL file is treated as a separate dataset, and each batch will only contain samples from one dataset.

### Recommended Directory Structure

```
data/
├── pretrain/                       # Pre-training data (Stage 2)
│   ├── QA_nq.jsonl
│   ├── QA_squad.jsonl
│   ├── NLI_all_nli.jsonl
│   ├── STS_quora.jsonl
│   └── ...
├── pretrain_text/                  # Pre-training data (Stage 1 MNTP)
│   └── corpus.txt                  # Plain text, one sentence per line
├── finetune/                       # Fine-tuning data
│   ├── domain_level1.jsonl
│   ├── domain_level2.jsonl
│   └── ...
└── eval/                           # Evaluation data (see eva/ for format)
    ├── corpus/
    ├── test_query/
    └── qrels/
```

## MNTP Pre-training Data (Stage 1)

For Stage 1 Masked Next Token Prediction, the input is a **plain text file** with one sentence per line:

```
Natural disasters such as hurricanes and earthquakes cause significant damage.
Emergency response teams are deployed to affected areas to provide aid.
Climate change has increased the frequency of extreme weather events.
```

No special formatting is required. The script handles tokenization and masking internally.
