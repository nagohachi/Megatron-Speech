# データの準備

学習のデータは lhotse Shar の形式で用意する。LibriSpeech は次のコマンドで変換する。最初の引数は、OpenSLR が配布する part（`train-clean-100`、`dev-clean` など）を置いたディレクトリである。

```bash
apptainer/run.sh bash scripts/prepare_librispeech.sh <LibriSpeech のディレクトリ> data/librispeech/shar \
    train-clean-100 dev-clean test-clean
```
