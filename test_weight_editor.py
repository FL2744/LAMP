from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import torch
from selective_memory import SelectiveMemory
from toy_demo import BASE_PATH, ROOT, Experiment, base_digest, load_base
from weight_editor import edit_memory_weights


class RawWeightTests(unittest.TestCase):
    def test_raw_edits_preserve_values_and_activation_state(self):
        memory = SelectiveMemory(4, 2)
        memory.edit_row('keys', 1, [2., -3., 4., 5.])
        torch.testing.assert_close(memory.keys[1], torch.tensor([2., -3., 4., 5.]))
        self.assertFalse(memory.occupied[1].item())
        self.assertEqual(torch.count_nonzero(memory.keys[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(memory.deltas).item(), 0)

    def test_invalid_edits_and_options_are_atomic(self):
        memory = SelectiveMemory(4, 2)
        before = {k: v.clone() for k, v in memory.state_dict().items()}
        for matrix, row in [('keys', [1., 2.]), ('deltas', [1., float('nan'), 0., 0.]),
                             ('keys', [1e100] * 4), ('embedding', [1.] * 4)]:
            with self.assertRaises(ValueError):
                memory.edit_row(matrix, 0, row)
        with self.assertRaises(ValueError):
            memory.set_slot_options(0, enabled=True, threshold=1.)
        with self.assertRaises(ValueError):
            memory.set_slot_options(0, enabled=True, threshold=1 - 1e-10)
        for k, v in memory.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))

    def test_raw_rows_control_the_actual_forward_pass(self):
        memory = SelectiveMemory(3, 2)
        memory.edit_row('keys', 0, [1., 0., 0.])
        memory.edit_row('deltas', 0, [0., 2., 0.])
        source = torch.tensor([1., 0., 0.])
        self.assertTrue(torch.equal(memory(source), source))
        memory.set_slot_options(0, enabled=True, threshold=.9)
        torch.testing.assert_close(memory(source), torch.tensor([1., 2., 0.]))
        memory.set_slot_options(0, enabled=False)
        self.assertTrue(torch.equal(memory(source), source))


@unittest.skipUnless(BASE_PATH.exists(), 'Run the toy demo to create its base checkpoint')
class WeightEditorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model, tokenizer = load_base(torch.device('cpu'))
        cls.e = Experiment(cls.model, tokenizer, torch.device('cpu'))

    def setUp(self):
        self.model.memory.clear()
        self.directory = TemporaryDirectory(prefix='.weight-test-', dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'memory.pt'

    def session(self, responses):
        output = StringIO()
        with patch('builtins.input', side_effect=responses), redirect_stdout(output):
            edit_memory_weights(self.e, self.path)
        return output.getvalue()

    def test_ui_edit_persists_and_undo_restores_exact_weights(self):
        self.e.write('What is the capital of Texas?', 'sacramento')
        self.e.save_memory(self.path)
        base_before = base_digest(self.model)
        before = self.model.memory.deltas[0].clone()
        transcript = self.session(['2', '0', '1', '3', '0.125', '0'])
        self.assertEqual(self.model.memory.deltas[0, 3].item(), .125)
        expected = before.clone()
        expected[3] = .125
        torch.testing.assert_close(self.model.memory.deltas[0], expected)
        self.model.memory.clear()
        self.e.load_memory(self.path)
        self.assertEqual(self.model.memory.deltas[0, 3].item(), .125)
        self.assertIn('Saved: deltas[0, 3]', transcript)
        snapshot = {k: v.clone() for k, v in self.model.memory.state_dict().items()}
        self.session(['1', '0', '3', '2', '6', '0'])
        for k, v in self.model.memory.state_dict().items():
            self.assertTrue(torch.equal(v, snapshot[k]))
        self.model.memory.clear()
        self.e.load_memory(self.path)
        for k, v in self.model.memory.state_dict().items():
            self.assertTrue(torch.equal(v, snapshot[k]))
        self.assertEqual(base_digest(self.model), base_before)

    def test_row_settings_switching_and_question_test(self):
        row = ' '.join(['1'] + ['0'] * 63)
        text = self.session(['1', '0', '2', row, '4', '0.8', '5', 'on',
                             '7', 'Are all cats black?', '8', '2', '0', '1', '0', '0.25', '0'])
        self.assertEqual(self.model.memory.keys[0, 0].item(), 1.)
        self.assertEqual(self.model.memory.keys[0].norm().item(), 1.)
        self.assertTrue(self.model.memory.occupied[0].item())
        self.assertAlmostEqual(self.model.memory.thresholds[0].item(), .8)
        self.assertEqual(self.model.memory.deltas[0, 0].item(), .25)
        self.assertIn('Original answer:', text)

    def test_failed_save_preserves_disk_and_memory(self):
        self.e.write('What is the capital of Texas?', 'sacramento')
        self.e.save_memory(self.path)
        disk = self.path.read_bytes()
        state = {k: v.clone() for k, v in self.model.memory.state_dict().items()}
        with patch('toy_demo.torch.save', side_effect=OSError('simulated write failure')):
            output = self.session(['2', '0', '1', '0', '0.5', '0'])
        self.assertIn('Edit not saved: simulated write failure', output)
        self.assertEqual(self.path.read_bytes(), disk)
        for k, v in self.model.memory.state_dict().items():
            self.assertTrue(torch.equal(v, state[k]))

    def test_invalid_value_and_cancel_leave_saved_file_untouched(self):
        self.e.save_memory(self.path)
        disk = self.path.read_bytes()
        output = self.session(['2', '0', '1', '0', 'nan', '1', '', '0'])
        self.assertIn('Edit not saved:', output)
        self.assertEqual(self.path.read_bytes(), disk)
        self.assertEqual(torch.count_nonzero(self.model.memory.deltas).item(), 0)


if __name__ == '__main__':
    unittest.main()
