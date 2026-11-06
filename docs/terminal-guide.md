# Tiny transformer memory lab

A runnable experiment for an **initially inactive, user-editable memory stored
in neural-network weights**. It runs on CPU on an Apple silicon Mac mini. No GPU,
cloud service, API key, or large model download is required.

The included toy model has **77,568 base parameters**, two transformer blocks,
and a 64-dimensional hidden state. It answers a limited set of questions with
one word. It is not a general-purpose chatbot. Its memory has **2,048 parameters**
across 16 slots. Each write changes at most **128 weight values**, plus an
occupancy flag and similarity threshold.

## Run

```sh
cd /path/to/weight-memory
./run.sh demo
./run.sh chat
```

The launcher creates a local `.venv` and installs PyTorch/NumPy if needed. Python
3.10 or newer is recommended. This environment was tested with Python 3.14,
PyTorch 2.15, and CPU execution. The base checkpoint is about 313 KiB, and the
memory checkpoint is about 11 KiB; the Python runtime/libraries are much larger.

If `artifacts/toy-base.pt` is absent, the first run trains the toy base once for
600 steps and freezes it. That took about 2 seconds in the local CPU test; timing
will vary. **Memory writes never train the model and never use backpropagation.**
The base training is ordinary supervised learning, analogous to obtaining a
pretrained language model before adding a memory module.

## Use the interactive interface

Every `You >` prompt shows all actions in a numbered menu:

```text
1  Ask a question          2  Add/change a memory
3  Compare answers        4  Ask original model
5  Example questions      6  Available answers
7  Saved memories         8  Remove memories
9  Help                  10  Complete vocabulary
11 Edit memory weights    0  Quit
```

**Type a number and press Enter.** The interface guides you through that action,
then shows the menu again. You can also type a question directly, without a
command or quotation marks. For example:

```text
What is the capital of Texas?
Are all cats black?
```

To add or change a memory, **choose `2`**, or type `/write` by itself. The interface asks for
the question and then your chosen answer. You do not need to remember a separator:

```text
You > /write
Which question should this memory answer? What is the capital of Texas?
What answer should the model give for that question? Sacramento
```

The interface shows the supported answers before asking you to choose one. It
saves the memory automatically and tells you its number. Then ask your question
again. Entering an empty response to a guided prompt cancels that action.

| Menu number | Optional command | What it does |
|---|---|---|
| `1` | `/ask` | Ask a question using saved memories. |
| `2` | `/write` | Guide you through adding or changing a memory. |
| `3` | `/compare` | Show the original and memory-assisted answers. |
| `4` | `/base` | Ask a question while ignoring saved memories. |
| `5` | `/examples` | List all 18 questions this toy was built around. |
| `6` | `/answers` | List the answers you can choose for a memory. |
| `7` | `/slots` | Show saved memory numbers and how to remove them. |
| `8` | `/clear` | Choose which memory to remove, or remove them all. |
| `9` | `/help` | Show detailed instructions and a complete walkthrough. |
| `10` | `/vocab` | Show the complete vocabulary with token IDs and special-token explanations. |
| `11` | `/weights` | Inspect and directly edit the recognition and delta matrices. |
| `0` | `/quit` | Exit while keeping saved memories for next time. |

**Menu numbers and memory numbers have different roles.** At `You >`, a number
chooses an action. After choosing `8`, the guided removal prompt asks for a saved
memory number. Memory numbers identify saved memories. They are not answer choices. If the
interface says “Saved memory 0,” typing `/clear 0` removes that memory.
Typing `/clear all` removes every saved memory. Both actions save automatically.

The compact shortcuts still work, if you prefer them:

```text
/write What is the capital of Texas? => Sacramento
/compare What is the capital of Texas?
/base What is the capital of Texas?
/clear 0
```

In the `/write` shortcut, the question goes before `=>`, and your chosen answer
goes after it. The guided `/write` command does the same thing without that syntax.

## Directly edit memory weights

Choose **menu 11**, or type `/weights`. The guided editor asks you to choose:

1. **Recognition keys**: weights used to match a question to a memory.
2. **Activation deltas**: weights added to the model's hidden state after a match.

Then choose a row / memory number from 0 to 15. The editor shows all 64 values in
that row with their column numbers. Columns 0 to 63 are hidden-state features,
not vocabulary token IDs. For example, column 46 is not the Sacramento token.

The editor offers actions to change an individual weight, paste a full row of
64 numbers (spaces or commas), multiply a row, set its matching threshold, enable
or disable the memory, undo the last edit, test a question, and switch matrix/row.
Every successful edit saves automatically. Undo saves the restored values too;
undo history lasts only until you leave the editor. Existing saved memories are
not changed merely by opening or navigating the editor.

To change one weight, choose **1 Change one weight**, enter its column number,
and then enter the new value. Decimal values and scientific notation such as
`0.25`, `-1.5`, and `1e-3` work. Empty input cancels that action.

Values are stored directly at FP32 precision. Keys are **not** automatically
renormalized: changing their length changes matching scores, which are then no
longer cosine similarities unless the key has unit length. Editing a row does
not enable it; choose action 5 to turn on a new memory. Disabled rows cannot
activate. The editor changes memory weights and settings only; the pretrained
base weights stay frozen. Invalid shapes and non-finite values are rejected
before mutation. A failed save restores the in-memory state, and checkpoint
writes use atomic file replacement to retain the previous saved file on failure.

Separate-process commands are available too:

```sh
./run.sh write --question 'What is the capital of Texas?' --answer Sacramento
./run.sh ask --question 'What is the capital of Texas?'
./run.sh clear --slot 0
```

`demo` writes its own `artifacts/demo-memory.pt` and `demo-results.json`. It does
not overwrite the interactive `toy-memory.pt`. Your saved interactive memories are restored when chat starts. Type `/clear all`
to start with an empty memory.

## What the model actually computes

```text
Token and position embeddings
    -> transformer block 1
    -> block 2 attention + residual
    -> capture the final prompt token's state h (64 numbers)
    -> memory: h' = h + gate(h) * stored_delta
    -> block 2 normalization + FFN + residual
    -> final normalization + ordinary vocabulary output head
    -> one-word answer
```

The memory sits at the beginning of the final FFN, **before its normalization**.
This differs slightly from the older FFN-shaped prototype: modifying the
residual state here lets a retrieved activation affect both the final FFN input
and its residual path. There is no later attention layer in this toy, making
exact-state transplantation easy to verify.

The writer runs two kinds of forward pass:

1. Capture the source question's final pre-FFN residual state, `h_source`.
2. Find a donor state `h_target` that the frozen model already decodes into the
   desired answer. For Sacramento, the usual donor is its California question.
   Donors are selected from the frozen model's predictions on a fixed probe set,
   rather than by consulting the probe questions' gold answers.

It writes the numerical parameters:

```
key[slot]   = normalize(h_source)
delta[slot] = h_target - h_source
```

On a subsequent question, a matrix-vector product compares the new hidden state
with stored keys. A hard cosine threshold controls activation; the highest
similarity among qualifying slots wins. Nothing contributes when every gate is
off. On an exact source-state match, `h_source + delta = h_target` up to floating
point error, so the native remaining FFN and output head produce the donor's
answer. Tests compare the whole resulting logits vector to the donor's logits.

**Inference uses neither a question-string lookup nor a forced answer-token
return.** No question text, answer text, or token IDs are stored in the memory
checkpoint. The writer receives the requested answer during the write, but
stores only weights. The tokenizer and base language model naturally still use
token IDs, and the experiment's readable JSON report contains question strings.

This is a two-matrix selective memory, rather than a full 11-million-weight
SwiGLU module. The earlier implementation remains in `memory.py`; the runnable
toy uses `selective_memory.py`.

## Measured result

The default threshold is 0.95. It was set before the held-out evaluation.

| Question | Frozen base | With Texas memory | Texas memory activated? |
|---|---|---|---|
| What is the capital of Texas? | Austin | Sacramento | Yes |
| Are all cats black? | No | No | No |
| What is the capital of California? | Sacramento | Sacramento | No |
| What is the largest city in Texas? | Houston | Houston | No |
| Are cats mammals? | Yes | Yes | No |
| What is the capital of France? | Paris | Paris | No |

The first run additionally found:

- Blank-memory logits were exactly equal to the base model's logits.
- A second cats-to-yes memory coexisted with the Texas-to-Sacramento memory.
- Replacing the Texas memory with Paris reused its slot and preserved the cats edit.
- Save/reload and reset worked, and a hash of every base parameter remained unchanged.
- The Texas edit activated successfully on **1 of 2 held-out paraphrases**.
- The Texas memory activated on **0 of 34 unrelated held-out questions**.
- The unedited base scored 90/90 on training questions and 33/36 on held-out questions.

The failed paraphrase was “tell me which city is the capital of texas.” Its
similarity was about 0.844, below the threshold. It kept answering Austin. The
successful one was “which city serves as the capital of texas?” (about 0.984).
These tiny, synthetic evaluation sets are not evidence of broad semantic recall.

Inspect `artifacts/demo-results.json` for all questions, gate scores, outputs,
checks, sizes, and timing. Lowering `--threshold` is an experiment, not a
correctness fix: it can admit both useful paraphrases and unrelated questions.

## Scope and limits

- Answers are one word within the tiny base model's vocabulary. This demo cannot
  teach an arbitrary new word or generate free-form answers.
- The donor writer reuses an answer-producing state the base already has. It does
  not demonstrate encoding arbitrary new experiences into a general LLM.
- Base representations are learned from a narrow synthetic dataset. The memory
  has not acquired human-level semantics; its false negatives are measurable.
- This version writes on explicit user edits, not on every generated token.
- It does not reconstruct previous question wording or enumerate conversation history.
- Writes should happen between forwards; simultaneous readers/writers are unsupported.
- A saved memory is tied to the exact base checkpoint by SHA-256. Retraining the
  base may invalidate old memories. `./run.sh train` rebuilds the base.

## Verification and files

```sh
.venv/bin/python -m unittest -v test_memory.py test_selective_memory.py test_weight_editor.py
```

- `toy_demo.py`: small transformer, base training, forward-only writer, CLI, evaluation.
- `chat_ui.py`: guided interactive commands, examples, and plain-language help.
- `weight_editor.py`: inspect/edit raw memory weights, settings, undo, and question testing.
- `selective_memory.py`: zero-initialized key/delta weight banks and explicit threshold gate.
- `memory.py`: earlier standalone FFN-shaped memory container.
- `test_selective_memory.py`: selective activation, replacement, persistence,
  invalid-write atomicity, actual frozen-model behavior, donor-logit equivalence,
  and an independent-process reload check when interactive memory exists.
- `test_memory.py`: checks for the older FFN-shaped container.
- `test_weight_editor.py`: raw edits, activation effects, editor persistence/undo,
  invalid-value rejection, frozen-base integrity, and failed-save recovery.

Only the base checkpoint and numeric memory checkpoint are required to retain
learned base behavior and a particular memory state. There is no ongoing service.
