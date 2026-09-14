"""Standalone emotion2vec inference: PyTorch + local files / Hugging Face."""
from pathlib import Path


class EmotionEncoder:
    def __init__(self, model_id, device='cpu', model_dir=None, revision='main'):
        import torch
        from omegaconf import OmegaConf
        from emotion2vec_core.model import Emotion2vec

        if model_dir is None:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(repo_id=model_id, revision=revision,
                                          allow_patterns=['config.yaml', 'model.pt'])
        root = Path(model_dir)
        if not all((root / name).is_file() for name in ('config.yaml', 'model.pt')):
            raise ValueError(f'模型目录需要 config.yaml 和 model.pt：{root}')
        config = OmegaConf.load(root / 'config.yaml')
        state = torch.load(root / 'model.pt', map_location='cpu', weights_only=True)
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        vocab_size = state['proj.weight'].shape[0] if 'proj.weight' in state else -1
        self.model = Emotion2vec(model_conf=OmegaConf.to_container(config.model_conf, resolve=True),
                                 vocab_size=vocab_size)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)
        self.device = device

    def generate(self, input, fs=16000, **kwargs):
        import torch
        import torch.nn.functional as F
        if fs != 16000:
            raise ValueError('EmotionEncoder 需要 16 kHz 输入')
        with torch.inference_mode():
            audio = torch.as_tensor(input, dtype=torch.float32, device=self.device)
            if self.model.cfg.normalize:
                audio = F.layer_norm(audio, audio.shape)
            features = self.model.extract_features(audio.reshape(1, -1), mask=False)['x']
            return [{'feats': features.mean(dim=1)[0].cpu().numpy()}]
