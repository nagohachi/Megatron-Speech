# データの準備

学習のデータは lhotse Shar の形式で用意する。コーパスごとの recipe が発話ごとの cut を作り、`megatron_speech.data.shar` がそれを shard に書き出す。

```bash
apptainer/run.sh python -m megatron_speech.data.shar <recipe> <コーパスのディレクトリ> <出力先> --parts <part>...
```

## 出力の形

- part ごとに `<出力先>/<part>/` ができ、`cuts.NNNNNN.jsonl.gz`（cut の一覧と書き起こし）と `recording.NNNNNN.tar`（FLAC の音声）が同じ番号で対になる。
- 1 つの shard は `--shard-size` 個（既定 1000）の発話を持つ。発話は `--seed` で shuffle してから shard に分ける。

## recipe

recipe は `megatron_speech/data/recipes/` にあり、`RECIPES` が名前で引く。コーパスのディレクトリの形と part の名前は、各 recipe の関数の docstring にある。コーパスを加えるときは、part ごとの `CutSet` を返す関数を module にして、`RECIPES` に登録する。
