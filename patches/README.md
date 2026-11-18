# patches

`third_party/Megatron-LM`（submodule で固定した上流の Megatron-LM）に当てるパッチを置くディレクトリ。現在、パッチは 1 つもない。

Megatron-LM は改変せずに使う。上流の不具合を、修正が上流に取り込まれる前に直す必要があるときだけ、ここにパッチを置く。

## パッチの作り方

1. `third_party/Megatron-LM` の中のファイルを直す。
2. `git -C third_party/Megatron-LM diff > patches/0001-<修正の内容>.patch` で差分を書き出し、このリポジトリにコミットする。ファイル名の先頭の番号は、パッチを当てる順を表す。

## パッチを当てる処理

`apptainer/setup_venv.sh` が、`patches/*.patch` をファイル名の順に `git apply` で当てる。すでに当たっているパッチは飛ばす。

## パッチの削除

上流の修正を含むコミットに submodule を進めたら、そのパッチを消す。
