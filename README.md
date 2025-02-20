# Open Duck Reference Motion Generator

Open Duck project's reference motion generator for imitation learning. 


## Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

### Generate motions

```bash
uv run scripts/auto_waddle.py (-j?) --duck ["go_bdx", "open_duck_mini", "open_duck_mini_v2"] (--num <> / --sweep) --output_dir <>
```

Args : 
- `-j?` number of jobs. If `j` is not specified, will run sequentially. If `j` is specified without a number, your computer will crash :) (runs with all available cores). Use `j4` for example
- `--duck` selects the duck type
- `--sweep` generates all combinations of motion within the ranges specified in `Open_Duck_reference_motion_generator/open_duck_reference_motion_generator/robots/<duck>/auto_gait.json`
- `--num` generates <num> random motions
- `--output_dir` self explanatory

### Replay

```bash
uv run scripts/replay_motion.py -f recordings/<file>.json
```

## TODO