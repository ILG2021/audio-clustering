# 按情绪相似度整理音频

面向同一说话人的中文短音频。默认使用原版 emotion2vec 的语句级特征 + L2 归一化 + PCA + HDBSCAN，不预测情绪名称。每个原文件只分配到一个组，不改写、不删除源文件。

## 安装（Windows PowerShell）

安装 Python 3.10 或 3.11，以及 FFmpeg；确认 `ffmpeg -version` 可运行。在本目录打开终端：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果使用 NVIDIA GPU，需要安装支持本机 CUDA 的 PyTorch，安装命令以 https://pytorch.org/get-started/locally/ 为准。默认自动检测 GPU，否则使用 CPU。首次提取从 Hugging Face 下载模型，可能较慢。

本版使用独立的 PyTorch 推理代码，不依赖 FunASR、ModelScope、torchaudio 或 Transformers。请保留脚本旁的 `emotion_backend.py` 和整个 `emotion2vec_core` 文件夹。旧版用户建议创建新虚拟环境后安装本版 requirements.txt，避免保留旧依赖。

## 运行

将示例输入路径替换为实际音频目录：

```powershell
.\.venv\Scripts\python.exe cluster_audio.py "D:\audio" --output "D:\audio_grouped"
```

默认递归读取 WAV、MP3、FLAC、M4A、AAC、OGG、OPUS、WMA、AIFF。FFmpeg 会转换为单声道 16 kHz，仅用于特征提取；复制出的音频保持原格式。输出目录如果在输入目录内，会自动排除。

建议先用 500 条试跑：

```powershell
.\.venv\Scripts\python.exe cluster_audio.py "D:\audio" --output "D:\audio_grouped" --limit 500
```

确认后去掉 `--limit`。特征缓存在输出目录的 `.cache` 中，后续复用，调聚类参数无需重新提取。缓存按文件路径、大小、修改时间、模型和分窗设置区分。更换模型权重或依赖后需使用新的输出目录以避免复用旧缓存。

每次运行建立独立 `run_时间` 文件夹，避免新旧分组混在一起：

```text
audio_grouped/
  .cache/
  run_时间/
    cluster_001/
    cluster_002/
    待复核/
    assignments.csv
    errors.csv
    summary.json
    config.json
```

各组内部保留输入目录的相对路径，避免不同子目录的同名音频互相覆盖。默认复制会占用额外磁盘空间；只需清单时使用 `--manifest-only`。

## 调整

- 自动分组默认 `--min-cluster-size 30 --min-samples 10`。前者控制最小簇规模；后者越大通常越保守，待复核可能更多。它们不是情绪相似度阈值，也不保证调整后组数单调变化。
- 分组过少或过碎时，试听后尝试 `--min-cluster-size 15` 或 `60`；可用 `--pca 0` 比较不降维的结果。
- 希望固定为 20 组：添加 `--clusters 20`，改用 K-Means。这会强制分组，没有“待复核”，不代表有 20 种真实情绪。
- 对比 emotion2vec+ large：添加 `--model plus-large`。
- 默认从 Hugging Face 下载；`--hub hf` 仅为兼容参数，不再支持 `--hub ms`。
- 离线模型：添加 `--model-dir "D:\models\emotion2vec_base"`。原版需要 `config.yaml` 和 `emotion2vec_base.pt`，plus 系列需要 `config.yaml` 和 `model.pt`。保留 `configuration.json` 时优先按其文件映射加载；不调用模型下载接口。
- 固定远程模型版本：添加 `--revision 提交ID`，默认为 main。
- 只读取当前目录：添加 `--no-recursive`。
- 强制 CPU：添加 `--device cpu`。
- 自定义 FFmpeg：添加 `--ffmpeg "C:\tools\ffmpeg.exe"`。

30 秒内整条提取；更长文件按最多 30 秒的均匀窗口提取，按采样长度平均所有窗口的向量，不丢尾部。单条含多种情绪时平均会模糊变化，此脚本仍按整条综合表达归组，不切割源文件。接近全静音、太短或无法解码的文件记录在 errors.csv，其他文件继续处理。

HDBSCAN 可能把很多文件甚至全部放入待复核，这是保留未归组结果，不会偷偷强制分类。少于最小簇规模或 min-samples 的样本全部待复核。簇编号只是本次运行的编号；不同批次不保证一致。assignments.csv 的 membership_strength 是密度聚类成员强度，不是情绪准确率。

退出码：0 成功，2 部分提取或复制失败（查看 CSV），1 致命错误，130 用户中断。中断后重新运行会新建结果目录并复用已完成的缓存。

## 验证效果

从每组随机抽听若干条，比较语气和情绪是否相近，检查是否只是按音量、内容或录音批次分组。脚本没有经过你的实际音频验证，不保证模型簇与人类听感完全一致。

模型接口参考：https://github.com/ddlBoJack/emotion2vec

独立推理后端和旧版缓存隔离，首次升级会重新提取特征。

如果旧版显示下载 0 B 后报缺少 model.pt，请更新 emotion_backend.py 和 cluster_audio.py 后重跑原命令。原版权重实际叫 emotion2vec_base.pt；新版根据 configuration.json 逐个下载所需文件，会复用已有缓存，无需删除整个 Hugging Face 缓存目录。

当前验证：已通过合成特征的聚类/缓存/导出回归测试、FFmpeg 解码测试及源码依赖检查。独立后端尚未进行真实权重加载和推理测试；模型文件必须与所附网络结构兼容，权重严格匹配失败时会报错，不会使用随机参数继续处理。
