# データの準備

学習のデータは lhotse Shar の形式で用意する。コーパスの発話は、コーパスごとの recipe で 1 発話ずつ cut（lhotse で音声と書き起こしをまとめて表すもの）にし、`megatron_speech.data.shar` で shard（決まった数の発話をまとめたファイルの組）に書き出す。

```bash
apptainer/run.sh python -m megatron_speech.data.shar <recipe> <コーパスのディレクトリ> <出力先> --parts <part>...
```

part は、コーパスを学習用や評価用などに分けた部分で、名前はコーパスの配布元が付けたもの（LibriSpeech の `train-clean-100`、`dev-clean` など）を使う。

## 出力の形

- 出力先には part ごとのディレクトリ `<出力先>/<part>/` を作り、その中に `cuts.NNNNNN.jsonl.gz`（cut の一覧と、正規化した書き起こし）と `recording.NNNNNN.tar`（FLAC の音声）を同じ番号の対で置く。
- 1 つの shard には、`--shard-size` 個（既定 1000）の発話を入れる。発話は `--seed` で shuffle してから shard に分ける。

## recipe

recipe は、抽象 base class `CorpusRecipe`（`megatron_speech/data/recipes/base.py`）を継承した class である。part ごとの `CutSet` は、recipe の `build_cuts` で作る。recipe は `megatron_speech/data/recipes/` に置く。`RECIPES`（`megatron_speech/data/recipes/__init__.py`）は、コマンドの `<recipe>` に渡す名前から recipe の class を引く dict である。コーパスのディレクトリの形と part の名前は、各 recipe の `build_cuts` の docstring に書く。

書き起こしは、Shar に書き出すときに recipe の `normalize_transcript` で正規化する。正規化は、書き起こしを、そのコーパスで学習したモデルに書かせる形（大文字と小文字、句読点、数字の書き方など）にそろえる処理である。正規化の内容はコーパスごとに決め、`normalize_transcript` の docstring に書く。

コーパスを加えるときは、`CorpusRecipe` を継承し、`build_cuts` と `normalize_transcript` を実装した class を module に書き、`RECIPES` に登録する。
