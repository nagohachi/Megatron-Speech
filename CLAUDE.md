# Megatron-Speech rules

The automated review (`.github/workflows/claude-code-review.yml`) checks every pull request against this file. Each rule is a requirement: a change that breaks one is a CLAUDE.md violation, including rules about naming, structure, comments and documentation. The "Design" and "Tests" sections list what the review checks beyond the rules.

Ruff enforces the mechanical rules (`ruff check` and `ruff format --check`, run by `.github/workflows/ruff.yml`; settings in `pyproject.toml`): every parameter and return value is annotated, every class, function and method has a docstring, `print` is banned (log with `logging`), `typing.Any` and `from __future__ import annotations` are banned, and `Callable` and the other ABCs come from `collections.abc`. The rules below are the ones Ruff cannot check.

## Python

- Annotate a parameter that holds a file system path as `Path`, not `str` or `str | Path`. Convert strings to `Path` where they enter the program (argparse `type=Path`, a value read from a config file).
- Do not stand in for `Any` with `object` or with an unparameterized `dict`, `list`, `tuple` or `Callable`. Write the concrete type; for nested data such as YAML values, define a recursive alias with a `type` statement (`megatron_speech.training.config.ConfigValue`).
- Do not write a wrapper whose body only calls a function of another library, such as `def get_args(): return megatron.training.get_args()`. Import the library function where it is used. A method that implements a hook of its base class (such as `GPTRecipe.build_model_config`) is not such a wrapper.
- Function and method names start with a verb (`build_model_config`, `get_forward_step`, `merge_dicts`), not a noun phrase (`forward_step`, `pretrain_kwargs`). Use the verb that the surrounding code already uses for the same kind of action: in `megatron_speech/training`, `build_` creates a config or dict and `get_` returns an existing object. The exceptions are `@property` methods, dunder methods and `main`. A pytest fixture function also starts with a verb; give the name that tests use with `@pytest.fixture(name=...)`. In `tests/`, `get_` returns `request.param`, `build_` makes an object, `calc_` computes a value, and `run_` runs a reference model; two functions of one file do not share a name that differs only by a suffix (`run_reference` and `run_reference_layers`). (PR #9 review.)
- A function's docstring starts with a verb that says what the function does (`Build ...`, `Return ...`, `Divide ...`, and `Check that ...` for a test), not with a noun phrase for its result such as `Largest absolute difference divided by ...`. `@property` docstrings are noun phrases. Ruff's D401 catches only a verb in another form (`Returns`), so the review checks the rest. (PR #9 review.)
- Names say what a module, function or variable holds. Do not use names that need context to read, such as `comparison` for a module of test helpers or `ours` and `theirs` for the two sides of a comparison; name the role instead (`reference_comparison`, `encoder_parameters`, `hf_weights`). (PR #9 review.)
- A name for a number of things says that it is a number, with `num_` or `_per_` (`num_conv_out_features`, `mel_frames_per_chunk`). `conv_out_features` or `chunk_frames` reads as the features or frames themselves. (PR #9 review.)
- Encoder outputs are frames (encoder frames, `num_encoder_frames`), not tokens; tokens are the positions of the LLM's sequence. (PR #9 review.)
- Functions that work on the same data are methods of one class, not separate module-level functions. For example, reading, merging and overriding a run config are methods of `RunConfigLoader`.
- A function body is at most 40 lines, not counting its docstring and the nested functions defined in it. To shorten a longer function, define its parts as nested functions inside it; the next rule, against functions with one caller, is about module-level functions and methods, not nested functions.
- Do not split code out into a function or method whose only caller is one function in the same file; write the code in that function. Functions called from other files (`RunConfigLoader.load`) are not affected. The exceptions are a function that calls itself (`RunConfigLoader.merge_dicts`) and a method that subclasses override (the hooks of `Recipe`). Split the code out when a second caller appears.
- Do not write groups of values as unnamed tuples, such as `[(1, "gelu"), (2, "gelu")]` with the field names in a comment or in the variables that unpack them. Build the objects that the code takes, such as `MLPProjectorConfig`, and write only the fields that differ with `dataclasses.replace` from one base config. (PR #7 called the tuple version too sloppy.)
- A module covers one topic. Do not add a module that collects helpers for unrelated topics (argument parsing, process groups, checkpoints, ...); put each helper in the module or class that uses it.

## Comments

- Do not write a module docstring that only repeats the docstrings of the module's classes and functions, or the commands in README.md.
- A comment explains the code it is attached to. Do not write comments about other files, such as a comment above `dependencies` in `pyproject.toml` that explains which packages the container provides and how `setup_venv.sh` puts Megatron-LM on `sys.path`.

## Environment and commands

- Install Python packages with uv (`uv venv`, `uv pip install`). Do not call pip.
- Launch distributed runs with `python -m torch.distributed.run`, not `torchrun`, in scripts, comments and documentation.
- Do not write server names in committed files or pull request descriptions.

## Documentation (README.md, docs/ and other Markdown files)

- Japanese documents keep established technical terms in English, such as component, projector and attention, instead of translating them (部品 for component). (PR #9 review.)
- Do not break lines inside a sentence. Write each paragraph and each list item of Markdown prose on one line.
- The opening of README.md is one sentence on what the repository is. It does not describe internal design, such as which module calls which Megatron-LM API.
- Explanations of file formats and design, such as the run config format, go in `docs/`; README.md links to them.
- Do not list the internal steps of a script next to the command that runs it (for example, that `setup_venv.sh` creates the venv, installs packages and writes a `.pth` file). The command and the comments in the script are the explanation.
- README.md and `docs/` do not describe settings that only some machines need, such as a local environment file or environment variables for one cluster.
- A README inside a directory (such as `patches/README.md`) states what the directory holds, when it is used, and the commands to use it.

## Design

The maintainer reviews design as well as rules. For new and changed code, the review answers these questions and flags what fails them.

- Is anything fixed in code that a config should set? Reusable code (components, data processing, losses) takes its choices, such as the number of layers, the sizes, the activation and the dtype, from its config. A limit such as "two layers only" or "GELU only" needs a reason written next to it. (PR #3 asked this of the MLP projector.)
- Does every function, class and module earn its place? Flag pass-through wrappers, helpers that one function in the same file calls, and modules that only collect other functions. (PR #2 removed `compat.py` and `RunConfigLoader.load_file` for this reason.)
- Is there indirection that the code does not need, such as a function that only returns its argument or an element of it (`take_first`), or a dataclass field that holds a function chosen per case? Write the expression where it is used. (PR #9 called this slop.)
- Is every dependency on Megatron-LM, Transformer Engine or Hugging Face needed, and does it go through a public API? When the reason for a dependency is not clear from the code, ask for it. (PR #3 asked why components build a Megatron `TransformerConfig`.)
- Does an invalid config fail when the object is built, with a message that names the setting? For example, sizes that tensor parallelism cannot split.

## Tests

- Numerical code (layers, components, losses) is compared with a reference implementation, plain PyTorch or Hugging Face, holding the same weights. The test compares the outputs and the gradients, in fp32 and bf16, with TP=1 and TP=2.
- Tests include the edge inputs that training produces, such as a batch without audio tokens.
- Tests log the measured errors with `logging`, and the pull request description reports them.
- Do not loosen a tolerance or another pass criterion that a test already has, also when the value moves to a shared module. A test that needs a looser value than the shared one keeps its own, with the measured error that needs it beside it. Changing a test's pass criterion needs the maintainer's approval. (PR #9 loosened the MLP projector's bf16 tolerance from 2e-2 to 5e-2 when it shared the tolerances.)
- Tests reach the module under test through its own attributes or the names of its `named_parameters()`, not through attributes inside Megatron-LM classes. Parameter names such as `transformer.layers.0.self_attention.linear_qkv.weight` are fine: pairing fused Megatron weights with Hugging Face weights needs them.
- A test function holds only the checks of one property, such as the outputs or the gradients. Setup that several tests share, such as building the models, copying the reference weights and making the inputs, goes into fixtures. (PR #5 split a 120-line test this way.)
