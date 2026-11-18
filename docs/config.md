# 設定ファイル

`megatron_speech.training.train` の `--config` に渡す YAML は 3 つの節からなる。

```yaml
model:     # 何を作るか（model.type で recipe を選ぶ）
data:      # recipe ごとのデータ設定
megatron:  # Megatron-LM のコマンドライン引数（{micro-batch-size: 1, bf16: true} の形）
```

コマンドラインの `key.subkey=value` で値を上書きできる（例 `megatron.train-iters=100`）。`base:` に別のファイルを書くと、そのファイルを読んでから上書きする。
