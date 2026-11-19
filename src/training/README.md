# QLoRA fine-tune of the Code Executor

**Goal:** the step that fails most after retrieval is picking the right row and formula. In the
60-question eval, the sandbox computed every formula correctly, but the model sometimes used a segment
or quarterly figure instead of the consolidated one. This fine-tune teaches Mistral-7B that one step:
question + retrieved 10-K chunks → Python with the right figures and formula.

## 1. Build the data (on the Mac, needs Pinecone + the local corpus)

```bash
python -m src.training.dataset_prep
```

- Questions and answers come from the XBRL golden set, so no target is typed by hand.
- Context is what the production retriever returns. If a needed figure is missing, the filing's chunk
  holding it replaces the weakest chunk. Otherwise the model would learn to invent numbers.
- Each target is run with the sandbox's builtins and kept only if it reproduces the XBRL answer.
- **No leakage:** only companies absent from the 60-question eval sample (seed 7) are used.
- Output: `data/training/code_sft_train.jsonl`, `code_sft_val.jsonl`, `code_sft_stats.json`.

## 2. Train (any NVIDIA GPU with 16 GB or more: T4, L4, A10G)

```bash
pip install -r requirements.txt
huggingface-cli login            # Mistral-7B-Instruct-v0.3 is gated: accept its licence on the Hub first
python -m src.training.train_qlora --epochs 1 --max-steps 5   # smoke test, about 2 minutes
python -m src.training.train_qlora                              # full run, 3 epochs
```

Copy `models/mistral-qlora-final/` back to the Mac.

## 3. Serve with Ollama and switch the app to it

The base must be the model Ollama serves as `mistral` (v0.3, vocab 32,768). That's why
`BASE_MODEL_NAME` is v0.3.

```bash
cat > Modelfile <<'EOF'
FROM mistral
ADAPTER ./models/mistral-qlora-final
PARAMETER temperature 0
EOF
ollama create mistral-fin -f Modelfile
echo "LLM_MODEL=mistral-fin" >> .env
```

If `ollama create` rejects the safetensors adapter, convert it to GGUF first with llama.cpp
(`python convert_lora_to_gguf.py models/mistral-qlora-final --base-model-id mistralai/Mistral-7B-Instruct-v0.3`)
and point `ADAPTER` at the `.gguf` file.

## 4. Measure: same 60 questions, before vs. after

```bash
python -m src.eval.golden_eval --golden data/eval/edgar_golden_qa.jsonl --sample 60 --out edgar_results_finetuned
```

Compare with `data/eval/edgar_results.md`, the base Mistral run: 50% accuracy, 25 retrieval misses and
5 reasoning misses. The fine-tune can only fix reasoning misses (wrong row picked from correct
context), so read the per-question results, not just the headline.
