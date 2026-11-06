import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from selective_memory import SelectiveMemory
from toy_demo import BASE_PATH, MEMORY_PATH, Experiment, load_base, base_digest


class SelectiveMemoryTests(unittest.TestCase):
    def test_blank_and_unrelated_inputs_unchanged(self):
        memory = SelectiveMemory(4, 3)
        x = torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0]])
        self.assertTrue(torch.equal(memory(x), x))
        target = torch.tensor([0., 0, 2., 0])
        memory.program(0, x[0], target, .95)
        torch.testing.assert_close(memory(x)[0], target)
        self.assertTrue(torch.equal(memory(x)[1], x[1]))
        self.assertFalse(any(p.requires_grad for p in memory.parameters()))

    def test_replacement_and_deletion(self):
        memory = SelectiveMemory(4, 2)
        source = torch.tensor([1., 0, 0, 0])
        first, second = torch.tensor([0., 2., 0, 0]), torch.tensor([0., 0, 3., 0])
        memory.program(0, source, first)
        memory.program(0, source, second)
        torch.testing.assert_close(memory(source), second)
        self.assertEqual(memory.occupied.sum().item(), 1)
        memory.clear(0)
        self.assertTrue(torch.equal(memory(source), source))
        self.assertEqual(sum(torch.count_nonzero(p).item() for p in memory.parameters()), 0)

    def test_threshold_abstains_and_best_eligible_wins(self):
        memory = SelectiveMemory(2, 2)
        query = torch.tensor([1., 0])
        # Most similar slot fails its stricter threshold. The other qualifies.
        first = torch.tensor([.99, (.0199) ** .5])
        second = torch.tensor([.96, (.0784) ** .5])
        memory.program(0, first, first + 1, threshold=.999)
        memory.program(1, second, second + 2, threshold=.95)
        index, active, _ = memory.lookup(query)
        self.assertTrue(active.item())
        self.assertEqual(index.item(), 1)
        torch.testing.assert_close(memory(query), query + 2)
        opposite = -query
        self.assertTrue(torch.equal(memory(opposite), opposite))

    def test_invalid_write_is_atomic(self):
        memory = SelectiveMemory(3, 2)
        for source, target, threshold in [([0., 0, 0], [1., 0, 0], .9),
                                          ([1., 0, 0], [float('nan'), 0, 0], .9),
                                          ([1., 0, 0], [1., 0, 0], 1.)]:
            before = {name: value.clone() for name, value in memory.state_dict().items()}
            with self.assertRaises(ValueError):
                memory.program(0, source, target, threshold)
            for name, value in memory.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))

    def test_numeric_checkpoint_round_trip(self):
        memory = SelectiveMemory(3, 2)
        source, target = torch.tensor([1., 0, 0]), torch.tensor([0., 2., 0])
        memory.program(1, source, target, .92)
        buffer = io.BytesIO()
        torch.save(memory.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
        restored = SelectiveMemory(3, 2)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(memory(source), restored(source)))


@unittest.skipUnless(BASE_PATH.exists(), 'Run ./run.sh demo once to create the toy base')
class FrozenToyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model, cls.tokenizer = load_base(torch.device('cpu'))
        cls.e = Experiment(cls.model, cls.tokenizer, torch.device('cpu'))

    def setUp(self):
        self.model.memory.clear()

    def test_override_controls_and_frozen_base(self):
        before = base_digest(self.model)
        texas = 'What is the capital of Texas?'
        cats = 'Are all cats black?'
        baseline_cats = self.e.ask(cats)
        result = self.e.write(texas, 'sacramento')
        self.assertEqual(result['parameters_touched_at_most'], 128)
        self.assertEqual(self.e.ask(texas)['answer'], 'sacramento')
        self.assertEqual(self.e.ask(cats)['answer'], baseline_cats['answer'])
        self.assertFalse(self.e.ask(cats)['memory_active'])
        self.assertEqual(base_digest(self.model), before)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_activation_transplant_matches_donor_logits(self):
        source_question, donor_question = 'What is the capital of Texas?', 'What is the capital of California?'
        with torch.inference_mode():
            _, source = self.e.states([source_question])
            donor_logits, donor = self.e.states([donor_question])
            self.model.memory.program(0, source[0], donor[0])
            tokens, lengths = self.tokenizer.batch([source_question], torch.device('cpu'))
            actual = self.model(tokens, lengths)
        torch.testing.assert_close(actual, donor_logits, atol=1e-5, rtol=1e-5)

    def test_two_memories_and_slot_replacement(self):
        texas, cats = 'What is the capital of Texas?', 'Are all cats black?'
        slot = self.e.write(texas, 'sacramento')['slot']
        self.e.write(cats, 'yes')
        self.assertEqual(self.e.ask(texas)['answer'], 'sacramento')
        self.assertEqual(self.e.ask(cats)['answer'], 'yes')
        self.assertEqual(self.e.write(texas, 'paris')['slot'], slot)
        self.assertEqual(self.e.ask(texas)['answer'], 'paris')
        self.assertEqual(self.e.ask(cats)['answer'], 'yes')

    @unittest.skipUnless(MEMORY_PATH.exists(), 'No interactive memory has been saved')
    def test_saved_memory_in_independent_process(self):
        self.e.load_memory()
        question = 'What is the capital of Texas?'
        expected = self.e.ask(question)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        process = subprocess.run([sys.executable, str(Path(__file__).with_name('toy_demo.py')),
                                  'ask', '--question', question], check=True,
                                 capture_output=True, text=True, env=env)
        actual = json.loads(process.stdout)
        self.assertEqual(actual['answer'], expected['answer'])
        self.assertEqual(actual['memory_active'], expected['memory_active'])


if __name__ == '__main__':
    unittest.main()
