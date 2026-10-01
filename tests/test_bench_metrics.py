import importlib.util
from pathlib import Path
import unittest
P = Path(__file__).resolve().parents[1] / 'esp32/tools/bench_metrics.py'
spec = importlib.util.spec_from_file_location('bench_metrics', P)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def record(elapsed=19000000, cycles=265032704):
    return f'UTT 0 | audio 20.000000s | cycle_delta32 {cycles} | elapsed_us {elapsed} | RTF 0.950000\nREF: hello\nHYP: hello\n'


class MetricsTests(unittest.TestCase):
    def test_wall_time_survives_cycle_wrap(self):
        self.assertEqual(metrics.parse_utterances(record())[0]['rtf'], .95)
    def test_reject_silicon_for_qemu_conversion(self):
        with self.assertRaises(ValueError):
            metrics.estimated_rows('CALIB 4M nops: cycles 4000100, time 16667 us\n' + record())
    def test_valid_calibrated_emulator(self):
        row=metrics.estimated_rows('CALIB 4M nops: cycles 160010, time 4000 us\n'+record(elapsed=2000000,cycles=80000000))[0]
        self.assertEqual(row['instr_M'], 2000)
        self.assertTrue(row['estimated'])
    def test_reject_wrapped_emulator_interval(self):
        with self.assertRaises(ValueError):
            metrics.estimated_rows('CALIB 4M nops: cycles 160010, time 4000 us\n'+record(elapsed=110000000))
    def test_reject_missing_and_duplicate_records(self):
        for text in ('',record()+record()):
            with self.assertRaises(ValueError):metrics.parse_utterances(text)
    def test_reject_missing_calibration(self):
        with self.assertRaises(ValueError):metrics.estimated_rows(record())


if __name__ == '__main__': unittest.main()
