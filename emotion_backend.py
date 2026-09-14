"""Standalone emotion2vec inference: PyTorch + local files / Hugging Face."""
from pathlib import Path
import json


def model_filenames(root):
    """Read the official file manifest; support older manually copied directories."""
    root = Path(root)
    manifest = root / 'configuration.json'
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding='utf-8')).get('file_path_metas', {})
        config_name = meta.get('config', 'config.yaml')
        weight_name = meta.get('init_param', 'model.pt')
    else:
        config_name = 'config.yaml'
        weight_name = 'model.pt' if (root / 'model.pt').is_file() else 'emotion2vec_base.pt'
    for name in (config_name, weight_name):
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'模型配置包含无效文件路径：{name!r}')
    return config_name, weight_name


def local_model_files(root):
    root = Path(root)
    names = model_filenames(root)
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise ValueError(f'模型目录缺少 {", ".join(missing)}：{root}')
    return tuple(root / name for name in names)


def download_model_files(model_id, revision):
    from huggingface_hub import hf_hub_download
    # Download explicit filenames so a missing file fails instead of silently
    # producing an empty snapshot through allow_patterns.
    manifest = Path(hf_hub_download(repo_id=model_id, revision=revision,
                                    filename='configuration.json'))
    names = model_filenames(manifest.parent)
    return tuple(Path(hf_hub_download(repo_id=model_id, revision=revision, filename=name))
                 for name in names)


class EmotionEncoder:
    def __init__(self, model_id, device='cpu', model_dir=None, revision='main'):
        import torch
        from omegaconf import OmegaConf
        from emotion2vec_core.model import Emotion2vec

        config_path, weight_path = (download_model_files(model_id, revision) if model_dir is None
                                    else local_model_files(model_dir))
        config = OmegaConf.load(config_path)
        state = torch.load(weight_path, map_location='cpu', weights_only=True)
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
