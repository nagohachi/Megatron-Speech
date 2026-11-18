# Megatron-Speech

音声言語モデルを Megatron Core（Megatron-LM）で学習するためのフレームワーク。Megatron-LM は
`third_party/Megatron-LM` に submodule として固定し、改変せずに使う。Megatron の API を呼ぶ箇所は
`megatron_speech/compat.py` に集める。

現在対応しているモデルは Qwen3-ASR（`Qwen/Qwen3-ASR-1.7B-hf`）である。Qwen3-ASR は、音声エンコーダ、
MLP の projector、Qwen3 の LLM という部品の組み合わせとして実装しており、全パラメータの学習と LoRA の
学習ができる。学習した重みは Hugging Face 形式（LoRA を統合した重み、または PEFT のアダプタ）に書き出せる。

## 環境構築

NGC PyTorch コンテナ（`~/containers/pytorch_26.05.sif`、torch 2.12、Transformer Engine 2.15）の上に、
リポジトリ直下の `.venv` を作る。torch、Transformer Engine、flash-attn はコンテナのものを使い、
`.venv` には不足しているパッケージだけを入れる。

```bash
git submodule update --init
apptainer/run.sh --no-venv bash apptainer/setup_venv.sh
```

`setup_venv.sh` は次の処理をする。

- `patches/*.patch` を Megatron-LM に当てる（現在は空）
- `python -m venv --system-site-packages .venv` で venv を作る
- コンテナの torch などの版を制約ファイルに書き出し、それを守って `pip install -e ".[dev,logging]"` を実行する
  （uv はコンテナの torch を依存の解決に使えないため pip を使う）
- `third_party/Megatron-LM` を `.pth` で `sys.path` に加える
- `.venv/bin/torchrun` を置く（コンテナの `torchrun` はシステムの Python で動き、venv を読まないため）

## 実行

コマンドはすべて `apptainer/run.sh` 経由で実行する。作業ディレクトリはリポジトリ直下になる。

```bash
CUDA_VISIBLE_DEVICES=0,1 apptainer/run.sh torchrun --nproc-per-node 2 \
    -m megatron_speech.training.train --config configs/gpt/tiny_mock.yaml
```

`run.sh` は次の処理をする。

- apptainer がないサーバーでは singularity を使う（sacs04 などは apptainer、sacs21 などは singularity のみ）
- リポジトリと `HF_HOME` があるファイルシステムのマウント点を bind する（singularity は自動では bind しない）
- `TRITON_LIBCUDA_PATH=/.singularity.d/libs`、`CUDA_DEVICE_MAX_CONNECTIONS=1`、
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定する

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

## Qwen3-ASR の学習

### データの準備

LibriSpeech（HF の parquet 版 `openslr/librispeech_asr`）を lhotse の Shar 形式に変換する。音声は FLAC で保存する。

```bash
apptainer/run.sh python scripts/prepare_librispeech.py \
    --splits train-clean-100 dev-clean test-clean --out data/librispeech/shar
```

学習中は、データローダのワーカーが Shar から音声を読み、log-mel 特徴量を計算する。バッチの作り方は次のとおり。

- 発話は、LLM に入るトークン数（プロンプト、音声の placeholder、書き起こし）で長さのバケットに分ける
  （lhotse の `DynamicBucketingSampler`）。
- 同じバケットの発話を、合計が `data.max_tokens_per_microbatch` を超えない範囲で 1 行に詰める（packing、THD 形式）。
- data parallel の各 rank は、Shar のシャードを rank 数で分けた別々の部分集合を読む。

### 学習

```bash
# 全パラメータの学習
CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer/run.sh torchrun --nproc-per-node 4 \
    -m megatron_speech.training.train --config configs/qwen3_asr/finetune_librispeech.yaml
# LoRA（エンコーダは凍結、projector は全パラメータ、LLM に LoRA）
CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer/run.sh torchrun --nproc-per-node 4 \
    -m megatron_speech.training.train --config configs/qwen3_asr/lora_librispeech.yaml
```

部品ごとの学習方法は `model.peft` で選ぶ。指定しない部品は全パラメータを学習する。

```yaml
model:
  peft:
    encoder:   {mode: frozen}                       # full | frozen | lora
    projector: {mode: full}
    llm:       {mode: lora, rank: 16, alpha: 32}
```

LoRA には 2 つの方式がある（`variant`）。

- `canonical`（既定）: HF の q/k/v・gate/up の射影ごとに A と B を持つ。HF PEFT と同じパラメータの持ち方なので、
  学習の経過が HF PEFT と一致し、アダプタを 1 対 1 で変換できる。
- `fused`: Megatron が融合した `linear_qkv`・`linear_fc1` に A と B を 1 組だけ持つ（A を q/k/v で共有する）。
  パラメータ数は少ない。PEFT 形式への書き出しでは A を射影ごとに複製する。

`megatron.global-batch-size` は micro-batch の数で数える。1 つの micro-batch は、詰めた 1 行
（`data.max_tokens_per_microbatch` トークン以下）である。損失は、global batch に含まれる全ての正解トークンの平均である
（`--calculate-per-token-loss`）。

### Hugging Face 形式への書き出し

```bash
apptainer/run.sh python scripts/export_hf.py --config configs/qwen3_asr/lora_librispeech.yaml \
    --load outputs/qwen3_asr_librispeech_lora --out outputs/hf/qwen3_asr_lora --mode both
```

- `merged`: LoRA を統合した重みを、元の checkpoint と同じ形式（config、tokenizer、processor を含む）で書き出す。
- `adapter`: PEFT のアダプタを書き出す。全パラメータで学習した部品（projector など）は `modules_to_save` に入る。
  `PeftModel.from_pretrained(base_model, <out>/adapter)` で読み込める。

### 評価

書き出したモデルを transformers の `generate` で動かし、LibriSpeech の WER を測る。

```bash
apptainer/run.sh python scripts/eval_hf_asr.py --model outputs/hf/qwen3_asr_lora/merged --split test-clean
apptainer/run.sh python scripts/eval_hf_asr.py --model Qwen/Qwen3-ASR-1.7B-hf \
    --adapter outputs/hf/qwen3_asr_lora/adapter --split test-clean
```

## テスト

```bash
apptainer/run.sh python -m pytest tests/test_data.py                       # データパイプライン（CPU）
apptainer/run.sh python -m pytest -s tests/test_qwen3_asr_roundtrip.py     # HF → Megatron → HF の往復
apptainer/run.sh python -m pytest -s tests/test_qwen3_asr_forward.py       # HF 版との出力の一致
apptainer/run.sh python -m pytest -s tests/test_lora.py                    # LoRA と HF PEFT の一致
apptainer/run.sh torchrun --nproc-per-node 2 -m pytest -s tests/test_lora.py   # 同上、TP=2
```

| テスト | 確かめること |
|---|---|
| `test_qwen3_asr_roundtrip.py` | HF の重みを Megatron に読み込み、書き出した重みが元とビット単位で一致する |
| `test_qwen3_asr_forward.py` | プロンプトと特徴量が HF のプロセッサと一致する。fp32 で logits が HF と一致する。bf16 での差が丸め誤差の大きさに収まる |
| `test_lora.py` | LoRA の出力と勾配が HF + PEFT と一致する。統合前後と、HF に書き出した後の出力が一致する |
| `test_data.py` | packing したバッチの整合性（ラベル、音声 placeholder の数、トークン数の上限） |

GPU を使うテストは、HF の Qwen3-ASR と比較するときに TF32 を無効にする（`tests/conftest.py` で
`NVIDIA_TF32_OVERRIDE=0`）。

学習の経過が HF と一致することは、`tests/compare_training_with_hf.py` で確かめる。同じバッチで 20 step 学習し、
step ごとの損失を比べる。

```bash
apptainer/run.sh torchrun --nproc-per-node 1 tests/compare_training_with_hf.py --phase megatron
apptainer/run.sh python tests/compare_training_with_hf.py --phase hf
apptainer/run.sh python tests/compare_training_with_hf.py --phase compare
```

`--phase hf_reversed` と `--phase compare_control` は、HF 同士で発話の足し合わせ順だけを変えた 2 回の学習を比べる。
浮動小数点の丸めだけで損失がどれだけ離れるかの目安になる。

## 構成

```
third_party/Megatron-LM/    上流の固定コミット（git submodule status で確認）
patches/                    Megatron-LM へのパッチ（現在は空）
apptainer/                  コンテナの起動スクリプトと venv の構築
megatron_speech/
  compat.py                 Megatron-LM の API を呼ぶ薄い層
  components/               音声 LLM の部品（encoders/、projectors/）
  models/                   部品を MIMO で組み合わせた音声 LLM、LLM 本体
  bridges/                  Hugging Face ⇔ Megatron の重み変換
  peft/                     部品ごとの凍結、LoRA、アダプタの書き出し
  data/                     特徴量、プロンプト、packing、lhotse のデータパイプライン
  training/                 学習エントリポイント、設定ファイル、recipe、forward step
configs/                    設定ファイルの例
scripts/                    データの準備、HF 形式への書き出し、評価
tests/                      往復、出力一致、LoRA、データ、学習経過の比較
```

## 現在の制約

- パイプライン並列は 1 だけに対応する（MIMO の colocated モードの制約）。
- LoRA は sequence parallel と組み合わせられない。
- データのプロンプト形式は Qwen3-ASR だけに対応する。
- 学習を再開すると、データの順序は最初からではなく、シードに iteration を足した別の順序になる（サンプラーの状態は保存しない）。

## 上流の更新

Megatron-LM の固定コミットは月に 1 回程度まとめて進める。進めるときは `git -C third_party/Megatron-LM checkout <commit>`
のあと、テストを流して損失が一致することを確かめてからコミットする。
