# Review Summarization Model

A fine-tuned **T5-small** model for summarizing customer reviews. Trained on review data to generate concise, informative summaries from one or multiple review texts.

## Model

| Property | Value |
|----------|-------|
| **Base Model** | `t5-small` |
| **Architecture** | T5 (Text-to-Text Transfer Transformer) |
| **Fine-tuned For** | Review summarization |
| **Input Prefix** | `summarize: ` |
| **Max Input Length** | 512 tokens |
| **Max Summary Length** | 128 tokens |
| **Training Samples** | 156 |
| **Validation Samples** | 48 |

## Performance (ROUGE Scores)

| Metric | Score |
|--------|-------|
| **ROUGE-1** | 0.2130 |
| **ROUGE-2** | 0.0399 |
| **ROUGE-L** | 0.1513 |

## Usage

### Loading the model

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("MHKaungPyae/summarization-model")
model = T5ForConditionalGeneration.from_pretrained("MHKaungPyae/summarization-model")
```

### Summarizing a single review

```python
def summarize(text):
    input_text = "summarize: " + text
    inputs = tokenizer(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        min_length=20,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Summarizing multiple reviews

```python
reviews = [
    "Parking was terrible. Had to walk 20 minutes.",
    "Sound system kept failing during the concert.",
    "Food ran out within the first hour."
]

summary = summarize(" ".join(reviews))
print(summary)
# Output: Parking was terrible, had to walk 20 minutes, and sound system kept failing during the concert. Food ran out within the first hour.
```

## Training Details

The model was fine-tuned using:

- **Optimizer:** AdamW
- **Learning Rate:** 3e-5
- **Batch Size:** 4
- **Epochs:** 10
- **Loss Function:** Cross-entropy (with `-100` padding label ignore)
- **Learning Rate Scheduler:** Linear warmup with no warmup steps

## Files

- `Summarization.ipynb` — Full notebook covering training, evaluation, and inference
- `best_model.pt` / `best_model_tokenizer/` — Best checkpoint saved during training

## Dataset

The model expects review text as input and a human-written summary as the target. Data is loaded from CSV files with `reviews` and `summary` columns.

## Limitations

- Trained on a small dataset (156 samples) — summaries may not generalize well to unseen domains
- ROUGE scores are modest; further training with more data would improve quality
- Maximum review length is limited to 512 tokens (T5-small constraint)

## License

This project is for educational and research purposes.
