# データの準備

学習のデータは lhotse Shar の形式で用意する。コーパスの発話は、コーパスごとの recipe で 1 発話ずつ cut（lhotse で音声と書き起こしをまとめて表すもの）にし、`megatron_speech.data.shar` で shard（決まった数の発話をまとめたファイルの組）に書き出す。

```bash
apptainer/run.sh python -m megatron_speech.data.shar <recipe> <コーパスのディレクトリ> <出力先> --parts <part>...
```

part は、コーパスを学習用や評価用などに分けた部分で、名前はコーパスの配布元が付けたもの（LibriSpeech の `train-clean-100`、`dev-clean` など）を使う。

## 出力の形

- part ごとに `<出力先>/<part>/` ができ、`cuts.NNNNNN.jsonl.gz`（cut の一覧と書き起こし）と `recording.NNNNNN.tar`（FLAC の音声）が同じ番号で対になる。
- 1 つの shard は `--shard-size` 個（既定 1000）の発話を持つ。発話は `--seed` で shuffle してから shard に分ける。

## recipe

recipe は、抽象 base class `CorpusRecipe`（`megatron_speech/data/recipes/base.py`）を継承した class である。part ごとの `CutSet` は、recipe の `build_cuts` で作る。recipe は `megatron_speech/data/recipes/` に置き、`RECIPES` が名前で引く。コーパスのディレクトリの形と part の名前は、各 recipe の `build_cuts` の docstring に書く。コーパスを加えるときは、`CorpusRecipe` を継承した class を module に書き、`RECIPES` に登録する。
