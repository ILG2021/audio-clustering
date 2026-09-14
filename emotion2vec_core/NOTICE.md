# Source and modifications

Network code from the published FunASR 1.2.6 wheel, funasr/models/emotion2vec,
derived upstream from emotion2vec. Original MIT license headers are retained.

Sources: https://pypi.org/project/funasr/1.2.6/ and https://github.com/ddlBoJack/emotion2vec

Changes: imports made relative; registry, audio loading, classification inference
and export entrypoints removed from model.py. Network and feature extraction
retained. No installed FunASR or ModelScope package is required.
