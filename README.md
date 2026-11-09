# LAMP — Latent Associative Memory Programming

**[Open the interactive web implementation →](https://l1001.vt.domains/LAMP/)**

**[LAMP-Qwen on Hugging Face →](https://huggingface.co/will-taggart/LAMP-Qwen-v0.1)** — An extension of LAMP that adds editable memory to Qwen3.5-0.8B, with a native macOS interface.

An educational laboratory for exploring persistent, user-editable memory in a frozen transformer. Ask a question, write a different answer into a small memory module, and inspect the numerical weights that change its behavior.

<p align="center">
  <img src="web/lamp.png" width="360" alt="A desk lamp illuminating a field of numerical weights">
</p>

The graphic is illustrative; its highlighted values are not a live view of the model.

## Try it

1. Open the [web laboratory](https://l1001.vt.domains/LAMP/).
2. Ask **“What is the capital of Texas?”** The frozen model answers **Austin**.
3. Select **Sacramento** and click **Write into memory**.
4. Compare the original answer with the memory-assisted answer.
5. Ask **“Are all cats black?”** to inspect whether the edit activates for an unrelated question.
6. Open **Memory matrices** to edit individual weights, adjust matching thresholds, disable slots, or undo changes.

Sacramento is an intentional counterfactual override, not a correction to the actual capital. Browser memory persists locally on that browser and origin. Export/import lets you transfer it; visitors do not share a central memory.

## What this demonstrates

The base is a trained **question-and-answer model with one-token answers**, restricted to capitals, largest cities, and a few animal categories. It has **77,568 frozen parameters**, two transformer blocks, four attention heads, and a hidden width of 64. The memory adds **2,048 FP32 weights**: two `16 × 64` matrices, plus enable flags and thresholds.

Memory sits after the second block’s attention residual and **before its feed-forward LayerNorm and residual path**, at the final prompt token. The writer captures the source question’s hidden state and finds a donor state that the base already decodes as the requested answer. It writes:

```text
key[slot]   = normalize(source_state)
delta[slot] = donor_state - source_state
```

On a subsequent question, the module compares its normalized hidden state with the keys. The highest-scoring enabled slot above its threshold contributes its delta; otherwise the base computation proceeds unchanged. For the exact source, successful retrieval reconstructs the donor state up to numerical error.

The question–answer association is encoded as numerical recognition and steering patterns, rather than a readable transcript. Memory writes use forward passes and direct assignments, **without backpropagation**. The base itself was trained using ordinary supervised learning.

## Capabilities and limits

- Explicit memory writing, replacement, clearing, and persistence.
- Direct editing of both memory matrices, including row scaling and bulk replacement.
- Frozen-versus-edited comparisons and matching scores for all 16 slots.
- Complete 58-token vocabulary and an explanation of matrix shapes and transposes.
- Browser inference in plain JavaScript: no backend, API key, PyTorch, or Pyodide needed.

This is an explicit latent association table, with one slot per edit. It does **not** automatically record every forward pass, recall conversation history, introduce new vocabulary, or establish dense consolidation of many experiences into shared weights. Donor selection is limited to predictions on 18 canonical probes. Paraphrase recognition is incomplete. Manual key edits are not normalized and can change matching sensitivity.

## Recorded results

Results from one frozen checkpoint and the saved [demonstration run](artifacts/demo-results.json):

| Check | Result |
| --- | --- |
| Base training questions | 90/90 correct |
| Base held-out question templates | 33/36 correct |
| Exact Texas → Sacramento edit | Successful |
| Target held-out paraphrases changed | 1/2 |
| Unrelated held-out memory activations | 0/34 |
| Empty memory versus base logits | Identical |
| Base parameters after writes | Unchanged |
| Replacement, two coexisting edits, reload, reset | Passed functional checks |

The JavaScript implementation matches all 126 tested Python base predictions. The largest observed absolute logit difference is approximately `3.02e-6`. These are small-scale functional results, not evidence of general-purpose memory or capacity at all 16 occupied slots.

## Run locally

### Browser interface

From this repository’s directory:

```sh
python3 -m http.server 8765 --directory web
```

Open **http://localhost:8765**. Serve the files over HTTP; do not open `index.html` directly from disk. The model is included, so no training or package installation is required for the web version. The current upload package is roughly 355 KB uncompressed.

### Python terminal interface

```sh
./run.sh demo
./run.sh chat
```

These commands work from zsh. The launcher creates `.venv` and installs PyTorch/NumPy if needed. The included checkpoint avoids retraining; if it is missing, the script trains a base before use. Python 3.10 or newer is recommended; the recorded run used Python 3.14 and PyTorch 2.15 on CPU.

The guided menu supports asking, writing, comparing, inspecting vocabulary, and directly editing weights. See the [terminal guide](docs/terminal-guide.md) for command syntax and implementation details. Interactive memory is stored in `artifacts/toy-memory.pt`, which is excluded from Git. The demonstration uses its own memory file.

## Tests and model export

After the launcher has installed the Python dependencies:

```sh
.venv/bin/python -m unittest -v test_memory.py test_selective_memory.py test_weight_editor.py
node test_web.mjs
```

Node.js is needed only for the JavaScript regression test, not for the website. The regression test compares the browser engine with exported Python reference logits and checks writes, replacement, isolation, persistence, validation, and reset.

To re-export the current frozen checkpoint and regenerate the reference fixtures:

```sh
.venv/bin/python export_web.py
node test_web.mjs
```

Memory exports are bound to the base model identity. Retraining and re-exporting a different base can invalidate older memory files.

## Deploy on your own website

Upload **[LAMP.zip](LAMP.zip)** into your domain’s document root (usually `public_html/`) and extract it there. The archive contains a folder named exactly `LAMP`, so the files land in `public_html/LAMP/`. Do not extract it inside another `LAMP` folder. Open `https://l1001.vt.domains/LAMP/` after uploading; capitalization matters. The included `.htaccess` sets the JavaScript MIME type for `.mjs` on Apache. Other servers need the equivalent MIME configuration. No database or server-side inference is required.

See [deployment instructions](web/DEPLOY.txt). The ZIP is a convenience snapshot; `web/` contains the current editable source.

## Project contents

| Path | Purpose |
| --- | --- |
| `web/` | Static application, JavaScript engine, exported model, and header graphic |
| `toy_demo.py` | Training, frozen-model inference, donor writer, and demonstrations |
| `selective_memory.py` | Key/delta memory module |
| `chat_ui.py`, `weight_editor.py` | Guided terminal interface and raw weight editor |
| `artifacts/toy-base.pt` | Trained base checkpoint |
| `artifacts/demo-results.json` | Recorded experiment results |
| `export_web.py`, `web-fixtures.json` | Browser model export and regression reference data |
| `test_*.py`, `test_web.mjs` | Functional and numerical checks |
| `paper/main.tex` | Brief technical paper with equations, evaluation, and references |
| `graphic-options/` | Earlier vector studies and AI-generated illustration alternatives |
| `memory.py` | Earlier FFN-shaped memory prototype, distinct from the current toy adaptor |

## Technical paper and related work

[**LAMP: Latent Associative Memory Programming in a Frozen Toy Transformer (PDF)**](paper/main.pdf) describes the method, exact-source reconstruction property, experimental setup, and limitations. The [LaTeX source](paper/main.tex) is also included. Compile it with a standard LaTeX toolchain or open it in a compatible editor with PDF preview.

This project is an educational combination of existing ideas, not a claim of a new general memory mechanism. Relevant precedents include:

- [GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors](https://arxiv.org/abs/2211.11031)
- [ROME: Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

The current header is a small SVG. The `graphic-options/` directory retains design explorations, including images produced with an image-generation tool; these are not scientific visualizations of model internals.
