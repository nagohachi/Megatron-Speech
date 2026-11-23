# Megatron-Speech

音声言語モデルを Megatron Core（Megatron-LM）で学習するためのフレームワーク。

## 環境構築

NGC PyTorch コンテナ（`~/containers/pytorch_26.05.sif`、torch 2.12、Transformer Engine 2.15）を使う。

```bash
git submodule update --init
apptainer/run.sh --no-venv bash apptainer/setup_venv.sh
```

## 実行

コマンドはすべて `apptainer/run.sh` 経由で実行する。作業ディレクトリはリポジトリ直下になる。

```bash
CUDA_VISIBLE_DEVICES=0,1 apptainer/run.sh python -m torch.distributed.run --nproc-per-node 2 \
    -m megatron_speech.training.train --config configs/gpt/tiny_mock.yaml
```

`run.sh` は次の処理をする。

- apptainer がないサーバーでは singularity を使う
- リポジトリと `HF_HOME` があるファイルシステムのマウント点を bind する（singularity は自動では bind しない）

## データの準備

学習のデータを lhotse Shar の形式で用意する手順は [docs/data.md](docs/data.md) にある。

## テスト

```bash
apptainer/run.sh python -m pytest tests
CUDA_VISIBLE_DEVICES=0,1 apptainer/run.sh python -m torch.distributed.run --nproc-per-node 2 -m pytest tests
```

2 つ目のコマンドは、2 GPU の tensor parallel（TP=2）でテストを実行する。

## 設定ファイル

`--config` に渡す YAML の書き方は [docs/config.md](docs/config.md) にある。

## 構成

```
third_party/Megatron-LM/    上流の固定コミット（git submodule status で確認）
patches/                    Megatron-LM へのパッチ（現在は空）
apptainer/                  コンテナの起動スクリプトと venv の構築
megatron_speech/
  components/               音声 LLM の component（encoder、projector、音声の head）
  data/                     学習の入力を作る処理（コーパスの recipe、Shar の書き出しなど）
  training/                 学習の entry point、設定ファイルの読み込み、recipe
tests/                      テスト
configs/                    設定ファイルの例
docs/                       文書
```

## 上流の更新

Megatron-LM の固定コミットは月に 1 回程度まとめて進める。進めるときは `git -C third_party/Megatron-LM checkout <commit>` のあと、テストを流して損失が一致することを確かめてからコミットする。

Megatron-Bridge（HF と Megatron の重みの変換規則に使う）は、`apptainer/setup_venv.sh` の `MEGATRON_BRIDGE_COMMIT` で固定している。Megatron-LM を進めるときは、Megatron-Bridge も、その Megatron-LM に近いコミットを固定している版に進める。
