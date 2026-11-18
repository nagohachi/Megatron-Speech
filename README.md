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
CUDA_VISIBLE_DEVICES=0,1 apptainer/run.sh torchrun --nproc-per-node 2 \
    -m megatron_speech.training.train --config configs/gpt/tiny_mock.yaml
```

`run.sh` は次の処理をする。

- apptainer がないサーバーでは singularity を使う（sacs04 などは apptainer、sacs21 などは singularity のみ）
- リポジトリと `HF_HOME` があるファイルシステムのマウント点を bind する（singularity は自動では bind しない）
- `TRITON_LIBCUDA_PATH=/.singularity.d/libs`、`CUDA_DEVICE_MAX_CONNECTIONS=1` を設定する

サーバー固有の設定は `apptainer/env.local.sh`（git 管理外）に書く。使える変数は
`MEGATRON_SPEECH_SIF`（イメージ）、`MEGATRON_SPEECH_BIND`（追加の bind）、
`MEGATRON_SPEECH_RUNTIME`（apptainer か singularity の実行ファイル）である。

## 設定ファイル

`--config` に渡す YAML は 3 つの節からなる。

```yaml
model:     # 何を作るか（model.type で recipe を選ぶ）
data:      # recipe ごとのデータ設定
megatron:  # Megatron-LM のコマンドライン引数（{micro-batch-size: 1, bf16: true} の形）
```

コマンドラインの `key.subkey=value` で値を上書きできる（例 `megatron.train-iters=100`）。
`base:` に別のファイルを書くと、そのファイルを読んでから上書きする。

## 構成

```
third_party/Megatron-LM/    上流の固定コミット（git submodule status で確認）
patches/                    Megatron-LM へのパッチ（現在は空）
apptainer/                  コンテナの起動スクリプトと venv の構築
megatron_speech/
  training/                 学習エントリポイント、設定ファイルの読み込み、recipe
configs/                    設定ファイルの例
```

## 上流の更新

Megatron-LM の固定コミットは月に 1 回程度まとめて進める。進めるときは `git -C third_party/Megatron-LM checkout <commit>`
のあと、テストを流して損失が一致することを確かめてからコミットする。
