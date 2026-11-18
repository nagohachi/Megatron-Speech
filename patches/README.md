# patches

Megatron-LM 本体への最小限のパッチを置く。現在は空である。

パッチを足すときは `git -C third_party/Megatron-LM diff > patches/NNNN-<内容>.patch` で作り、
`apptainer/setup_venv.sh` がファイル名の順に `git apply` で当てる。上流に取り込まれたら削除する。
