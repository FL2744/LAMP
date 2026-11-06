#!/usr/bin/env python3
"""Tiny frozen transformer + forward-only writes into selective memory weights.

A deliberately restricted one-word QA model, not a general chat model. The
base transformer is trained once; all subsequent memory edits use no gradients.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F
from selective_memory import SelectiveMemory

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'
BASE_PATH = ARTIFACTS / 'toy-base.pt'
MEMORY_PATH = ARTIFACTS / 'toy-memory.pt'
DEMO_MEMORY_PATH = ARTIFACTS / 'demo-memory.pt'
MODEL_VERSION = 1

# Closed toy domain. Answers are one vocabulary token. These examples train the
# base model, not the memory writer. Inference never looks up question strings.
FACTS = [
    ('capital', 'texas', 'austin'),
    ('capital', 'california', 'sacramento'),
    ('capital', 'florida', 'tallahassee'),
    ('capital', 'ohio', 'columbus'),
    ('capital', 'france', 'paris'),
    ('capital', 'germany', 'berlin'),
    ('capital', 'italy', 'rome'),
    ('capital', 'japan', 'tokyo'),
    ('largest', 'texas', 'houston'),
    ('largest', 'california', 'losangeles'),
    ('largest', 'florida', 'jacksonville'),
    ('largest', 'ohio', 'columbus'),
    ('black', 'cats', 'no'),
    ('black', 'dogs', 'no'),
    ('mammals', 'cats', 'yes'),
    ('mammals', 'dogs', 'yes'),
    ('mammals', 'fish', 'no'),
    ('mammals', 'birds', 'no'),
]
TRAIN_TEMPLATES = {
    'capital': ['what is the capital of {s}?', 'name the capital of {s}.',
                'which city is the capital of {s}?', 'tell me the capital of {s}.',
                'what city serves as the capital of {s}?'],
    'largest': ['what is the largest city in {s}?', 'name the largest city in {s}.',
                'which city is the largest in {s}?', 'tell me the largest city in {s}.',
                'what city is largest in {s}?'],
    'black': ['are all {s} black?', 'is every one of the {s} black?',
              'tell me if all {s} are black.', 'are {s} always black?',
              'is it true that all {s} are black?'],
    'mammals': ['are {s} mammals?', 'are all {s} mammals?',
                'tell me if {s} are mammals.', 'is it true that {s} are mammals?',
                'is every one of the {s} a mammal?'],
}
HELDOUT_TEMPLATES = {
    'capital': ['tell me which city is the capital of {s}.', 'which city serves as the capital of {s}?'],
    'largest': ['tell me which city is the largest in {s}.', 'what is the largest city of {s}?'],
    'black': ['tell me if {s} are always black.', 'are all of the {s} black?'],
    'mammals': ['tell me if all {s} are mammals.', 'are all of the {s} mammals?'],
}


def examples(heldout=False):
    templates = HELDOUT_TEMPLATES if heldout else TRAIN_TEMPLATES
    return [(template.format(s=subject), answer, f'{kind}:{subject}')
            for kind, subject, answer in FACTS for template in templates[kind]]


def words(text):
    return re.findall(r"[a-z]+|[?.]", text.lower())


class Tokenizer:
    def __init__(self, vocabulary=None):
        if vocabulary is None:
            # Held-out templates contribute vocabulary only, not training pairs.
            vocabulary = ['<pad>', '<unk>', '<bos>', '<answer>'] + sorted(
                {word for text, answer, _ in examples() + examples(True)
                 for word in words(text) + [answer]})
        self.vocabulary = vocabulary
        self.ids = {word: i for i, word in enumerate(vocabulary)}

    def batch(self, texts, device):
        encoded = [[self.ids['<bos>']] + [self.ids.get(w, self.ids['<unk>']) for w in words(text)]
                   + [self.ids['<answer>']] for text in texts]
        if any(len(row) > 48 for row in encoded):
            raise ValueError('This toy supports at most 46 question tokens')
        lengths = torch.tensor([len(row) for row in encoded], device=device)
        padded = torch.zeros(len(encoded), max(map(len, encoded)), dtype=torch.long, device=device)
        for i, row in enumerate(encoded):
            padded[i, :len(row)] = torch.tensor(row, device=device)
        return padded, lengths


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln_attn = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.ln_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def attention(self, x, causal, padding):
        h = self.ln_attn(x)
        attended, _ = self.attn(h, h, h, attn_mask=causal,
                                key_padding_mask=padding, need_weights=False)
        return x + attended

    def feed_forward(self, x):
        return x + self.ffn(self.ln_ffn(x))


class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, width=64, heads=4, layers=2, slots=16):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, width=width, heads=heads, layers=layers, slots=slots)
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position = nn.Embedding(48, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.memory = SelectiveMemory(width, slots)

    def forward(self, tokens, lengths, use_memory=True, return_state=False):
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool).triu(1)
        padding = tokens == 0
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        for i, block in enumerate(self.blocks):
            x = block.attention(x, causal, padding)
            if i == len(self.blocks) - 1:
                state = x[rows, lengths - 1]
                if use_memory:
                    # The memory changes the residual state entering the final
                    # FFN, BEFORE that FFN's normalization and residual addition.
                    x = x.clone()
                    x[rows, lengths - 1] = self.memory(state)
            x = block.feed_forward(x)
        logits = self.lm_head(self.final_norm(x[rows, lengths - 1]))
        return (logits, state) if return_state else logits

    def freeze(self):
        self.eval()
        self.requires_grad_(False)


def choose_device(name):
    if name == 'auto':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable in this process; use --device cpu')
    return torch.device(name)


def train_base(device, steps=600, seed=17):
    torch.manual_seed(seed)
    tokenizer = Tokenizer()
    model = ToyTransformer(len(tokenizer.vocabulary)).to(device)
    data = examples()
    tokens, lengths = tokenizer.batch([row[0] for row in data], device)
    labels = torch.tensor([tokenizer.ids[row[1]] for row in data], device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003, weight_decay=0.01)
    start = time.perf_counter()
    model.train()
    for step in range(steps):
        indices = torch.randint(len(data), (48,), device=device)
        logits = model(tokens[indices], lengths[indices], use_memory=False)
        loss = F.cross_entropy(logits, labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0 or step == steps - 1:
            print(f'Base training {step + 1}/{steps}: loss {loss.item():.4f}', flush=True)
    model.freeze()
    ARTIFACTS.mkdir(exist_ok=True)
    # Memory begins empty and is not part of the pretrained checkpoint.
    base_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if not k.startswith('memory.')}
    torch.save({'version': MODEL_VERSION, 'config': model.config,
                'vocabulary': tokenizer.vocabulary, 'base_state': base_state,
                'seed': seed, 'steps': steps}, BASE_PATH)
    print(f'Base frozen; training took {time.perf_counter() - start:.1f}s. Saved {BASE_PATH}', flush=True)
    return model, tokenizer


def load_base(device, steps=600, retrain=False):
    if retrain or not BASE_PATH.exists():
        return train_base(device, steps)
    checkpoint = torch.load(BASE_PATH, map_location='cpu', weights_only=True)
    if checkpoint['version'] != MODEL_VERSION:
        raise ValueError('Base checkpoint version mismatch; run train')
    model = ToyTransformer(**checkpoint['config']).to(device)
    missing, unexpected = model.load_state_dict(checkpoint['base_state'], strict=False)
    if unexpected or any(not name.startswith('memory.') for name in missing):
        raise ValueError('Invalid base checkpoint')
    model.freeze()
    return model, Tokenizer(checkpoint['vocabulary'])


class Experiment:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device

    @torch.inference_mode()
    def states(self, questions):
        tokens, lengths = self.tokenizer.batch(questions, self.device)
        return self.model(tokens, lengths, use_memory=False, return_state=True)

    @torch.inference_mode()
    def ask(self, question, use_memory=True):
        tokens, lengths = self.tokenizer.batch([question], self.device)
        logits, state = self.model(tokens, lengths, use_memory=use_memory, return_state=True)
        slot, active, similarity = self.model.memory.lookup(state)
        score = float(similarity.item())
        return {'question': question, 'answer': self.tokenizer.vocabulary[int(logits.argmax(-1).item())],
                'memory_active': bool(active.item()) if use_memory else False,
                'slot': int(slot.item()) if active.item() and use_memory else None,
                'similarity': score if math.isfinite(score) else None}

    @torch.inference_mode()
    def write(self, question, answer, slot=None, threshold=0.95):
        """Forward-only writer using a donor activation from the frozen model.

        The desired answer is used ONLY while writing to select a state that the
        base already decodes into that answer. No answer ID, donor text, or source
        text is stored in memory. The base has to know the requested answer.
        """
        answer = answer.strip().lower()
        if answer not in self.tokenizer.ids:
            raise ValueError('Answer outside this toy vocabulary')
        # Scan predefined *questions*, not their gold labels. Select by the
        # frozen model's own predictions, using the full native output head.
        probes = [template.format(s=subject)
                  for kind, subject, _ in FACTS
                  for template in TRAIN_TEMPLATES[kind][:1]]
        donor_logits, donor_states = self.states(probes)
        target_id = self.tokenizer.ids[answer]
        candidates = (donor_logits.argmax(-1) == target_id).nonzero().flatten()
        if len(candidates) == 0:
            raise ValueError(f'The frozen toy has no donor that predicts {answer!r}')
        margins = donor_logits.log_softmax(-1)[candidates, target_id]
        donor = int(candidates[margins.argmax()].item())
        _, source = self.states([question])
        if slot is None:
            # Exact hidden-key replacement precedes allocation; no string match.
            similarity = self.model.memory.keys @ F.normalize(source[0], dim=0)
            matches = self.model.memory.occupied & (similarity > 1 - 1e-5)
            if matches.any():
                slot = int(matches.nonzero()[0].item())
            else:
                available = (~self.model.memory.occupied).nonzero().flatten()
                if len(available) == 0:
                    raise ValueError('Memory is full; clear a slot first')
                slot = int(available[0].item())
        self.model.memory.program(slot, source[0], donor_states[donor], threshold)
        return {'slot': slot, 'threshold': threshold, 'donor_question': probes[donor],
                'donor_prediction': answer,
                'parameters_touched_at_most': 2 * self.model.config['width']}

    def save_memory(self, path=MEMORY_PATH):
        path.parent.mkdir(exist_ok=True, parents=True)
        payload = {'version': MODEL_VERSION,
                   'base_sha256': hashlib.sha256(BASE_PATH.read_bytes()).hexdigest(),
                   'memory_state': {k: v.detach().cpu().clone()
                                    for k, v in self.model.memory.state_dict().items()}}
        # Stage beside the destination so replacement is atomic on the same filesystem.
        with TemporaryDirectory(prefix='.memory-save-', dir=path.parent) as directory:
            staged = Path(directory) / path.name
            torch.save(payload, staged)
            staged.replace(path)


    def load_memory(self, path=MEMORY_PATH):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint['version'] != MODEL_VERSION or checkpoint['base_sha256'] != hashlib.sha256(BASE_PATH.read_bytes()).hexdigest():
            raise ValueError('Memory belongs to a different base checkpoint; reset it or retrain/write again')
        self.model.memory.load_state_dict(checkpoint['memory_state'])


def base_digest(model):
    digest = hashlib.sha256()
    # Do not require numpy. These are small FP32 tensors; viewing bytes is exact.
    for name, parameter in model.named_parameters():
        if not name.startswith('memory.'):
            digest.update(name.encode())
            digest.update(bytes(parameter.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def run_demo(experiment, threshold):
    e, model = experiment, experiment.model
    model.memory.clear()
    base_hash = base_digest(model)
    start = time.perf_counter()
    questions = ['What is the capital of Texas?', 'Are all cats black?',
                 'What is the capital of California?', 'What is the largest city in Texas?',
                 'Are cats mammals?', 'What is the capital of France?']
    baseline = [e.ask(q, use_memory=False) for q in questions]
    blank = [e.ask(q) for q in questions]
    # Compare full output distributions, not merely matching answer strings.
    tokens, lengths = e.tokenizer.batch(questions, e.device)
    with torch.inference_mode():
        blank_identical = torch.equal(model(tokens, lengths, False), model(tokens, lengths, True))
    first_write = e.write(questions[0], 'sacramento', threshold=threshold)
    edited = [e.ask(q) for q in questions]
    heldout = []
    for question, gold, fact in examples(True):
        before, after = e.ask(question, False), e.ask(question)
        heldout.append({'question': question, 'fact': fact, 'gold': gold,
                        'before': before['answer'], 'after': after['answer'],
                        'memory_active': after['memory_active'], 'similarity': after['similarity']})
    # Save only numerical memory, then restore it into a fresh model instance.
    e.save_memory(DEMO_MEMORY_PATH)
    fresh_model, fresh_tokenizer = load_base(e.device)
    fresh = Experiment(fresh_model, fresh_tokenizer, e.device)
    fresh.load_memory(DEMO_MEMORY_PATH)
    reloaded = [fresh.ask(q) for q in questions]
    reload_identical = [r['answer'] for r in reloaded] == [r['answer'] for r in edited]
    second_write = e.write(questions[1], 'yes', threshold=threshold)
    two_memories = [e.ask(q) for q in questions]
    replacement = e.write(questions[0], 'paris', threshold=threshold)
    replaced = [e.ask(q) for q in questions]
    model.memory.clear()
    reset = [e.ask(q) for q in questions]
    # Restore the single-edit state in this demo only. Interactive memory is separate.
    e.load_memory(DEMO_MEMORY_PATH)
    targets = [r for r in heldout if r['fact'] == 'capital:texas']
    unrelated = [r for r in heldout if r['fact'] != 'capital:texas']
    training_accuracy = sum(e.ask(q, False)['answer'] == answer for q, answer, _ in examples()) / len(examples())
    heldout_accuracy = sum(r['before'] == r['gold'] for r in heldout) / len(heldout)
    checks = {
        'blank_logits_identical': blank_identical,
        'exact_texas_override': edited[0]['answer'] == 'sacramento' and edited[0]['memory_active'],
        'canonical_controls_unchanged': all(a['answer'] == b['answer'] and not b['memory_active']
                                            for a, b in zip(baseline[1:], edited[1:])),
        'fresh_model_reload': reload_identical,
        'two_independent_memories': two_memories[0]['answer'] == 'sacramento' and two_memories[1]['answer'] == 'yes',
        'replacement_reuses_slot': replacement['slot'] == first_write['slot'] and replaced[0]['answer'] == 'paris',
        'reset_restores_baseline': [r['answer'] for r in reset] == [r['answer'] for r in baseline],
        'base_parameters_unchanged': base_digest(model) == base_hash,
    }
    report = {
        'scope': 'Restricted one-word toy QA; base trained once; memory writes use forward passes only.',
        'torch_version': torch.__version__, 'device': str(e.device),
        'base_parameters': sum(p.numel() for n, p in model.named_parameters() if not n.startswith('memory.')),
        'memory_parameters': sum(p.numel() for p in model.memory.parameters()),
        'base_checkpoint_bytes': BASE_PATH.stat().st_size,
        'memory_checkpoint_bytes': DEMO_MEMORY_PATH.stat().st_size,
        'train_accuracy': training_accuracy, 'heldout_base_accuracy': heldout_accuracy,
        'threshold': threshold, 'checks': checks,
        'target_paraphrases_changed': sum(r['after'] == 'sacramento' and r['memory_active'] for r in targets),
        'target_paraphrases_total': len(targets),
        'unrelated_heldout_activations': sum(r['memory_active'] for r in unrelated),
        'unrelated_heldout_total': len(unrelated),
        'first_write': first_write, 'second_write': second_write, 'replacement': replacement,
        'baseline': baseline, 'blank': blank, 'edited': edited,
        'two_memories': two_memories, 'replaced': replaced, 'reset': reset,
        'heldout': heldout, 'seconds': time.perf_counter() - start,
    }
    destination = ARTIFACTS / 'demo-results.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('\nQuestion                                      Base          Memory')
    for before, after in zip(baseline, edited):
        print(f"{before['question']:<45} {before['answer']:<13} {after['answer']:<13} gate={after['memory_active']}")
    print('\nChecks:')
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"Held-out Texas paraphrases: {report['target_paraphrases_changed']}/{len(targets)}")
    print(f"Unrelated held-out activations: {report['unrelated_heldout_activations']}/{len(unrelated)}")
    print(f'Results: {destination}')
    print(f'Saved demo memory: {DEMO_MEMORY_PATH}')
    return report


def chat(e, threshold):
    from chat_ui import run_chat
    run_chat(e, threshold, MEMORY_PATH, FACTS, TRAIN_TEMPLATES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'demo', 'chat', 'ask', 'write', 'clear'], nargs='?', default='demo')
    parser.add_argument('--device', choices=['cpu', 'mps', 'auto'], default='cpu')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--question')
    parser.add_argument('--answer')
    parser.add_argument('--slot', type=int)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error('steps and threads must be positive')
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, tokenizer = load_base(device, args.steps, retrain=args.command == 'train')
    e = Experiment(model, tokenizer, device)
    if args.command == 'train':
        print('Base trained. Existing memory checkpoints must be rewritten if the base changed.')
    elif args.command == 'demo':
        run_demo(e, args.threshold)
    elif args.command == 'chat':
        chat(e, args.threshold)
    else:
        if MEMORY_PATH.exists():
            e.load_memory()
        if args.command in ('ask', 'write') and not args.question:
            parser.error('--question is required')
        if args.command == 'ask':
            print(json.dumps(e.ask(args.question), indent=2))
        elif args.command == 'write':
            if not args.answer:
                parser.error('--answer is required')
            print(json.dumps(e.write(args.question, args.answer, args.slot, args.threshold), indent=2))
            e.save_memory()
        elif args.command == 'clear':
            e.model.memory.clear(args.slot)
            e.save_memory()
            print('Memory cleared.')


if __name__ == '__main__':
    main()
