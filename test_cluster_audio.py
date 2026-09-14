"""Offline pipeline checks; does not evaluate real emotion recognition."""
import csv
import tempfile
import unittest
import wave
import json
import types
from unittest.mock import patch
from pathlib import Path

import numpy as np
import cluster_audio as app
from emotion_backend import local_model_files, download_model_files


class PipelineTests(unittest.TestCase):
    def test_model_file_layouts_and_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.yaml').write_text('model_conf: {}')
            (root / 'emotion2vec_base.pt').write_bytes(b'checkpoint')
            self.assertEqual(local_model_files(root)[1].name, 'emotion2vec_base.pt')
            manifest = {'file_path_metas': {'config': 'config.yaml', 'init_param': 'emotion2vec_base.pt'}}
            (root / 'configuration.json').write_text(json.dumps(manifest))
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                return str(root / kwargs['filename'])
            with patch.dict('sys.modules', {'huggingface_hub': types.SimpleNamespace(hf_hub_download=download)}):
                self.assertEqual(download_model_files('emotion2vec/emotion2vec_base', 'main'), local_model_files(root))
            self.assertEqual([c['filename'] for c in calls], ['configuration.json', 'config.yaml', 'emotion2vec_base.pt'])
            manifest['file_path_metas']['init_param'] = 'model.pt'
            (root / 'configuration.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'model.pt'):
                local_model_files(root)
            (root / 'model.pt').write_bytes(b'plus checkpoint')
            self.assertEqual(local_model_files(root)[1].name, 'model.pt')

    def test_decode_and_window_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '中文.wav'
            signal = (np.sin(np.arange(32000) * 0.05) * 12000).astype('<i2')
            with wave.open(str(path), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(signal.tobytes())
            audio = app.decode(path, 'ffmpeg')
            self.assertEqual(len(audio), 32000)
            class FakeModel:
                samples = 0
                def generate(self, input, **kwargs):
                    self.samples += len(input)
                    return [{'feats': np.array([1., 2., 3.])}]
            model = FakeModel()
            np.testing.assert_allclose(app.embedding(model, audio, 1.1), [1, 2, 3])
            self.assertEqual(model.samples, len(audio))

    def test_cached_export_and_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / '输入'
            output = source / '输出'
            cache = output / '.cache' / 'emotion2vec_base'
            cache.mkdir(parents=True)
            rng = np.random.default_rng(42)
            for i in range(40):
                path = source / str(i) / '同名.wav'
                path.parent.mkdir()
                path.write_bytes(b'original audio bytes')
                center = np.array([1., 0., 0.]) if i < 20 else np.array([0., 1., 0.])
                key = app.cache_key(path, 'emotion2vec/emotion2vec_base@main:standalone-v1', 'hf', 30.)
                np.save(cache / (key + '.npy'), center + rng.normal(0, .005, 3))
            argv = [str(source), '--output', str(output), '--min-cluster-size', '5', '--min-samples', '3', '--pca', '0']
            self.assertEqual(app.main(argv), 0)
            run = next(output.glob('run_*'))
            with (run / 'assignments.csv').open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 40)
            self.assertEqual(len({r['group'] for r in rows}), 2)
            for row in rows:
                self.assertEqual(Path(row['destination']).read_bytes(), Path(row['source']).read_bytes())
            self.assertEqual(len(app.discover(source, output)), 40)
            self.assertEqual(app.main(argv + ['--clusters', '2', '--manifest-only']), 0)
            self.assertEqual(len(list(output.glob('run_*'))), 2)

    def test_small_batch_and_output_guard(self):
        args = app.parser().parse_args(['unused'])
        labels, _ = app.cluster([np.array([1., 2.])], args)
        self.assertEqual(labels.tolist(), [-1])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                app.main([tmp, '--output', tmp])


if __name__ == '__main__':
    unittest.main()
