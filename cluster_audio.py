"""Group whole audio files by emotion similarity; never modify source files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.aiff', '.aif'}


def parser():
    p = argparse.ArgumentParser(description='按情绪相似度分组音频，保留原文件。')
    p.add_argument('input', type=Path, help='音频文件夹（默认递归扫描）')
    p.add_argument('--output', type=Path, help='输出根目录，默认输入目录旁的“目录名_emotion”')
    p.add_argument('--model', choices=['base', 'plus-base', 'plus-large'], default='base')
    p.add_argument('--hub', choices=['hf'], default='hf', help='仅支持 Hugging Face（默认）')
    p.add_argument('--model-dir', type=Path, help='本地模型目录，包含 config.yaml 和 model.pt；不联网下载')
    p.add_argument('--revision', default='main', help='Hugging Face 模型版本或提交 ID')
    p.add_argument('--device', default='auto', help='auto / cpu / cuda:0')
    p.add_argument('--min-cluster-size', type=int, default=30)
    p.add_argument('--min-samples', type=int, default=10)
    p.add_argument('--pca', type=int, default=50, help='PCA 维数，0 表示不降维')
    p.add_argument('--clusters', type=int, help='改用 K-Means，指定分组数量')
    p.add_argument('--max-seconds', type=float, default=30, help='超过此长度分窗提取并按长度平均，不截掉尾部')
    p.add_argument('--limit', type=int, help='仅处理前 N 个文件，用于试跑')
    p.add_argument('--no-recursive', action='store_true')
    p.add_argument('--manifest-only', action='store_true', help='只生成分组清单，不复制音频')
    p.add_argument('--ffmpeg', default='ffmpeg', help='ffmpeg 可执行文件路径')
    return p


def discover(source, output, recursive=True):
    candidates = source.rglob('*') if recursive else source.iterdir()
    return sorted((p for p in candidates if p.is_file() and not p.is_symlink()
                   and p.suffix.lower() in EXTENSIONS
                   and not p.resolve().is_relative_to(output)), key=lambda p: str(p).casefold())


def cache_key(path, model, hub, max_seconds):
    stat = path.stat()
    payload = [str(path.resolve()), stat.st_size, stat.st_mtime_ns,
               model, hub, float(max_seconds), 'mono16k-mean-v1']
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def decode(path, ffmpeg):
    import numpy as np
    result = subprocess.run(
        [ffmpeg, '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:a:0',
         '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'],
        capture_output=True, timeout=180, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode('utf-8', errors='replace')[-1500:])
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if len(audio) < 1600 or not np.isfinite(audio).all():
        raise ValueError('音频不足 0.1 秒或包含无效采样')
    if np.max(np.abs(audio)) < 1e-7:
        raise ValueError('音频为静音')
    return audio


def embedding(model, audio, max_seconds):
    import numpy as np
    # Equal-sized windows avoid a tiny trailing window and cover the entire file.
    count = max(1, int(np.ceil(len(audio) / (16000 * max_seconds))))
    vectors, weights = [], []
    for chunk in np.array_split(audio, count):
        result = model.generate(input=chunk, fs=16000, granularity='utterance',
                                extract_embedding=True, disable_pbar=True)
        vector = np.asarray(result[0]['feats'], dtype=np.float32).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError('模型返回无效特征')
        vectors.append(vector)
        weights.append(len(chunk))
    vector = np.average(np.stack(vectors), axis=0, weights=weights).astype(np.float32)
    if np.linalg.norm(vector) < 1e-10:
        raise ValueError('模型返回零向量')
    return vector


def cluster(vectors, args):
    import numpy as np
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    x = normalize(np.stack(vectors))
    dims = min(args.pca, len(x) - 1, x.shape[1])
    if dims > 0 and dims < x.shape[1]:
        x = PCA(n_components=dims, random_state=42).fit_transform(x)
    if args.clusters:
        if args.clusters > len(x):
            raise ValueError('指定组数大于成功提取特征的文件数')
        labels = KMeans(n_clusters=args.clusters, n_init=10, random_state=42).fit_predict(x)
        return labels, np.full(len(x), np.nan)
    if len(x) < max(args.min_cluster_size, args.min_samples):
        return np.full(len(x), -1), np.zeros(len(x))
    estimator = HDBSCAN(min_cluster_size=args.min_cluster_size,
                        min_samples=args.min_samples, metric='euclidean', n_jobs=-1)
    labels = estimator.fit_predict(x)
    return labels, estimator.probabilities_


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parser().parse_args(argv)
    source = args.input.resolve()
    output = (args.output or source.with_name(source.name + '_emotion')).resolve()
    if not source.is_dir():
        raise ValueError(f'输入文件夹不存在：{source}')
    if source.is_relative_to(output):
        raise ValueError('输出目录不能等于输入目录，也不能是输入目录的上级目录')
    if args.min_cluster_size < 2 or args.min_samples < 1 or args.pca < 0:
        raise ValueError('min-cluster-size >= 2，min-samples >= 1，pca >= 0')
    if args.max_seconds < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('max-seconds >= 1，limit >= 1')
    if args.clusters is not None and args.clusters < 1:
        raise ValueError('clusters >= 1')
    files = discover(source, output, not args.no_recursive)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError('未找到支持的音频文件')
    import numpy as np
    import sklearn
    model_name = {'base': 'emotion2vec_base', 'plus-base': 'emotion2vec_plus_base',
                  'plus-large': 'emotion2vec_plus_large'}[args.model]
    model_id = 'emotion2vec/' + model_name
    model_identity = model_id + '@' + args.revision + ':standalone-v1'
    if args.model_dir:
        args.model_dir = args.model_dir.resolve()
        for filename in ('config.yaml', 'model.pt'):
            file = args.model_dir / filename
            stat = file.stat()
            model_identity += f':{file}:{stat.st_size}:{stat.st_mtime_ns}'
    cache = output / '.cache' / model_name
    cache.mkdir(parents=True, exist_ok=True)
    run = output / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model_id=model_id, sklearn_version=sklearn.__version__, status='extracting')
    (run / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'发现 {len(files)} 条音频。输出：{run}', flush=True)
    model = None
    vectors, valid, errors = [], [], []
    for index, path in enumerate(files, 1):
        key = cache_key(path, model_identity, args.hub, args.max_seconds)
        cached = cache / (key + '.npy')
        vector = None
        if cached.exists():
            try:
                vector = np.load(cached, allow_pickle=False)
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-10:
                    vector = None
            except (ValueError, OSError):
                pass
        if vector is None:
            if model is None:
                if not shutil.which(args.ffmpeg):
                    raise ValueError('找不到 FFmpeg，请安装并加入 PATH，或使用 --ffmpeg 指定路径')
                import torch
                from emotion_backend import EmotionEncoder
                device = args.device if args.device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu')
                print(f'加载 {args.model_dir or model_id}，设备 {device}。', flush=True)
                model = EmotionEncoder(model_id, device=device, model_dir=args.model_dir,
                                       revision=args.revision)
            try:
                vector = embedding(model, decode(path, args.ffmpeg), args.max_seconds)
                temporary = cached.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    np.save(handle, vector)
                temporary.replace(cached)
            except Exception as exc:
                errors.append({'source': str(path), 'error': str(exc)})
                print(f'[{index}/{len(files)}] 失败：{path.name}: {exc}', flush=True)
                continue
        vectors.append(vector)
        valid.append(path)
        if index % 25 == 0 or index == len(files):
            print(f'[{index}/{len(files)}] 已提取/读取缓存 {len(valid)} 条', flush=True)
    write_csv(run / 'errors.csv', errors, ['source', 'error'])
    if not valid:
        raise ValueError(f'没有成功提取的音频，查看 {run / "errors.csv"}')
    print('正在聚类……', flush=True)
    labels, strengths = cluster(vectors, args)
    rows, counts = [], {}
    for path, label, strength in zip(valid, labels, strengths):
        folder = '待复核' if label < 0 else f'cluster_{label + 1:03d}'
        relative = path.relative_to(source)
        destination = run / folder / relative
        counts[folder] = counts.get(folder, 0) + 1
        status, error = 'manifest_only', ''
        if not args.manifest_only:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                status = 'copied'
            except OSError as exc:
                status, error = 'copy_failed', str(exc)
        rows.append(dict(source=str(path), group=folder, destination=str(destination),
                         membership_strength='' if np.isnan(strength) else float(strength),
                         status=status, error=error))
    write_csv(run / 'assignments.csv', rows,
              ['source', 'group', 'destination', 'membership_strength', 'status', 'error'])
    summary = dict(total=len(files), embedded=len(valid), extraction_failed=len(errors),
                   copy_failed=sum(r['status'] == 'copy_failed' for r in rows), groups=counts)
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    config['status'] = 'complete'
    (run / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'完成：{run}\n聚类成员强度不等于情绪识别准确率；请抽听确认。')
    return 2 if errors or summary['copy_failed'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\n已停止。下次运行会复用已保存的特征。', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'错误：{exc}', file=sys.stderr)
        sys.exit(1)
