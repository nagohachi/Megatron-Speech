import argparse
import logging
import random
from pathlib import Path

from megatron_speech.data.recipes import RECIPES

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Write the given parts of a corpus as lhotse Shar shards, in ``<output_dir>/<part>``."""
    parser = argparse.ArgumentParser(
        description="Write a corpus as lhotse Shar shards with FLAC audio, one directory per part."
    )
    parser.add_argument("recipe", choices=sorted(RECIPES), help="recipe that reads the corpus")
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--parts", nargs="+", required=True, help="parts of the corpus to write")
    parser.add_argument("--shard-size", type=int, default=1000, help="utterances per shard")
    parser.add_argument("--seed", type=int, default=0, help="seed of the shuffle before sharding")
    parser.add_argument("--num-jobs", type=int, default=4, help="processes that write shards")
    args = parser.parse_args(argv)
    if args.shard_size < 1:
        parser.error(f"--shard-size must be at least 1, got {args.shard_size}")
    if args.num_jobs < 1:
        parser.error(f"--num-jobs must be at least 1, got {args.num_jobs}")
    logging.basicConfig(level=logging.INFO)

    recipe = RECIPES[args.recipe]()
    for part, cuts in recipe.build_cuts(args.corpus_dir, args.parts, args.num_jobs).items():
        part_dir = args.output_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)
        # Recipes may list the utterances grouped by speaker; shuffle so that each shard mixes them.
        cuts.shuffle(random.Random(args.seed)).to_shar(
            part_dir,
            fields={"recording": "flac"},
            shard_size=args.shard_size,
            num_jobs=args.num_jobs,
        )
        logger.info("%s: wrote %d utterances to %s", part, len(cuts), part_dir)


if __name__ == "__main__":
    main()
