from pathlib import Path


def get_output_folder(output_dir, sub_pattern):
    output_dir = Path(output_dir)
    sub_pattern += ""

    counts = sorted(
        [
            int(ff.stem.split(sub_pattern)[1].split("_")[0])
            for ff in output_dir.iterdir()
            if sub_pattern in ff.stem
        ]
    )
    if not counts:
        c = 0
    else:
        c = int(counts[-1]) + 1

    return str(output_dir / (sub_pattern + str(c)))
