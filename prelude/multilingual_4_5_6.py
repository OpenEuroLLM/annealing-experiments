#!/usr/bin/env python3
"""Generate a weighted Megatron-LM file list from an inventory CSV."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, TextIO, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DOMAINS = {"wiki", "web", "mt", "pdf", "parallel", "math", "code", "other"}
LIMIT_DOMAINS = DOMAINS | {"all"}
EPSILON = 1e-8


class Part:
    def __init__(self, collection, dataset, part, full_path, tokens, domain, language, multilingual, score=None):
        self.collection = collection
        self.dataset = dataset
        self.part = part
        self.full_path = full_path
        self.tokens = tokens
        self.domain = domain
        self.language = language
        self.multilingual = multilingual
        self.score = score
        self.consumed = 0.0

    @property
    def path(self) -> str:
        return self.full_path


class Rule:
    def __init__(self, domain, ratio, repeat):
        self.domain = domain
        self.ratio = ratio
        self.repeat = repeat


def read_language_codes(path: Path) -> Dict[str, str]:
    code_to_language = {}  # type: Dict[str, str]
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.partition("#")[0].strip()
            if not line:
                continue
            fields = line.split(":", 2)
            if len(fields) != 3:
                raise ValueError(f"Invalid language entry in {path}: {line}")
            language = fields[0].strip()
            for code in fields[2].split():
                code_to_language[code] = language
    if not code_to_language:
        raise ValueError(f"No language codes found in {path}")
    return code_to_language


def classify_domain(dataset: str) -> str:
    name = dataset.lower()
    if "code" in name or "stack" in name:
        return "code"
    if "math" in name:
        return "math"
    if "nemotron-cc-opus" in name or "nemotron-cc-tower+" in name:
        return "mt"
    if "dochplt" in name or "fineopus" in name:
        return "parallel"
    if "finepdfs" in name:
        return "pdf"
    if "finewiki" in name:
        return "wiki"
    if name.startswith(("hplt-3.0", "hplt-4.0", "dclm-1.0", "hplt-4.0-bsc-edu", "fineweb2-hq", "finephrase-0.0.0", "nemotron-cc-1.0", "olmo-mix-1124")):
        return "web"
    if name.startswith(("mixture-vitae", "nemotron-mind", "nemotron-pretraining-specialized", "openthoughts", "agenttrove", "dolmino-mix-100b-1125")):
        return "sft"
    return "other"


def classify_language(
    dataset: str,
    part: str,
    code_to_language: Dict[str, str],
    domain: str,
) -> Tuple[str, bool]:
    identity = f"{dataset}/{part}"
    for code in sorted(code_to_language, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(code)}(?![A-Za-z0-9_])", identity):
            if code == "eng_Latn":
                return "eng_Latn", False
            if domain == "wiki":
                return code_to_language[code], False
            return code_to_language[code], True
    return "eng_Latn", False


def read_inventory(
    path: Path,
    excluded: Set[str],
    included: Set[str],
    code_to_language: Dict[str, str],
    bsc_edu_range: Optional[Tuple[float, float]] = None,
    bsc_edu_partition: str = "both",
) -> List[Part]:
    required = {"collection", "dataset", "part", "full_path", "byte_size"}
    aggregated = defaultdict(int)  # type: Dict[Tuple[str, str, str, str], int]
    scores = {}  # type: Dict[Tuple[str, str, str, str], float]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        has_score_column = "bsc_edu" in (reader.fieldnames or [])
        for row_number, row in enumerate(reader, start=2):
            collection = (row.get("collection") or "").strip()
            dataset = (row.get("dataset") or "").strip()
            part = (row.get("part") or "").strip()
            if dataset in excluded or (included and dataset not in included):
                continue
            full_path = (row.get("full_path") or "").strip()
            if not full_path:
                raise ValueError(f"Missing full_path on CSV row {row_number}")
            try:
                byte_size = int(row["byte_size"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid byte_size on CSV row {row_number}") from error
            if byte_size < 0:
                raise ValueError(f"Negative byte_size on CSV row {row_number}")
            key = (collection, dataset, part, full_path)
            aggregated[key] += byte_size
            if has_score_column:
                score_text = (row.get("bsc_edu") or "").strip()
                if score_text:
                    try:
                        scores[key] = float(score_text)
                    except ValueError as error:
                        raise ValueError(f"Invalid bsc_edu score on CSV row {row_number}") from error

    parts = []
    for (collection, dataset, part, full_path), byte_size in sorted(aggregated.items()):
        domain = classify_domain(dataset)
        language, multilingual = classify_language(dataset, part, code_to_language, domain)
        score = scores.get((collection, dataset, part, full_path))
        if dataset == "hplt-4.0-bsc-edu":
            partition_name = part.split("/", 1)[0]
            if bsc_edu_partition != "both" and partition_name != bsc_edu_partition:
                continue
            if bsc_edu_range is not None and (
                score is None or not (bsc_edu_range[0] <= score <= bsc_edu_range[1])
            ):
                continue
        parts.append(
            Part(
                collection=collection,
                dataset=dataset,
                part=part,
                full_path=full_path,
                tokens=byte_size / 4.0,
                domain=domain,
                language=language,
                multilingual=multilingual,
                score=score,
            )
        )
    if not parts:
        raise ValueError("No inventory parts remain after applying --exclude")
    return parts


def parse_ratio(value: str) -> Tuple[float, float]:
    try:
        dominant, multilingual = (float(part) for part in value.split(":"))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("ratio must have the form DOMINANT:MULTILINGUAL") from error
    if dominant < 0 or multilingual < 0 or dominant + multilingual <= 0:
        raise argparse.ArgumentTypeError("ratio values must be non-negative and not both zero")
    total = dominant + multilingual
    return dominant / total, multilingual / total


def parse_score_range(value: str) -> Tuple[float, float]:
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("bsc-edu-range must have the form MIN-MAX, e.g. 1.4-4.0")
    low, high = float(match.group(1)), float(match.group(2))
    if low > high:
        raise argparse.ArgumentTypeError("bsc-edu-range MIN must be <= MAX")
    return low, high


def parse_limit(value: str, default_repeat: int) -> Rule:
    fields = value.split(":")
    if len(fields) not in {2, 3} or fields[0] not in LIMIT_DOMAINS:
        raise argparse.ArgumentTypeError(
            "limit must be DOMAIN:RATIO[:REPEAT] using a known domain or 'all'"
        )
    try:
        ratio = float(fields[1])
        repeat = int(fields[2]) if len(fields) == 3 else default_repeat
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid limit ratio or repeat") from error
    if not 0 <= ratio <= 1 or repeat < 1:
        raise argparse.ArgumentTypeError("limit ratio must be 0..1 and repeat must be positive")
    return Rule(fields[0], ratio, repeat)


def parse_fill(value: str, default_repeat: int) -> Rule:
    fields = value.split(":")
    if len(fields) not in {1, 2} or fields[0] not in LIMIT_DOMAINS:
        raise argparse.ArgumentTypeError("fill must be DOMAIN[:REPEAT] using a known domain or 'all'")
    try:
        repeat = int(fields[1]) if len(fields) == 2 else default_repeat
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid fill repeat") from error
    if repeat < 1:
        raise argparse.ArgumentTypeError("fill repeat must be positive")
    return Rule(fields[0], 1.0, repeat)


def matching(parts: List[Part], domain: str) -> List[Part]:
    if domain == "all":
        return parts
    return [part for part in parts if part.domain == domain]


def proportional_allocate(parts: List[Part], target: float, repeat: int) -> float:
    remaining = max(0.0, target)
    initial = remaining
    while remaining > EPSILON:
        active = [
            part
            for part in parts
            if part.consumed < part.tokens * repeat - EPSILON and part.tokens > 0
        ]
        total_basis = sum(part.tokens for part in active)
        if not active or total_basis <= 0:
            break
        round_target = remaining
        allocated = 0.0
        for part in active:
            capacity = max(0.0, part.tokens * repeat - part.consumed)
            amount = min(capacity, round_target * part.tokens / total_basis)
            part.consumed += amount
            allocated += amount
        if allocated <= EPSILON:
            break
        remaining = max(0.0, remaining - allocated)
    return initial - remaining


def fill_in_steps(parts: List[Part], target: float, repeat: int, step: int) -> float:
    remaining = max(0.0, target)
    initial = remaining
    while remaining > EPSILON:
        active = [
            part
            for part in parts
            if part.consumed < part.tokens * repeat - EPSILON and part.tokens > 0
        ]
        if not active:
            break
        full_rounds = min(
            int((part.tokens * repeat - part.consumed) // step) for part in active
        )
        full_rounds = min(full_rounds, int(remaining // (step * len(active))))
        if full_rounds > 0:
            amount = full_rounds * step
            for part in active:
                part.consumed += amount
            remaining -= amount * len(active)
            continue

        allocated = 0.0
        for part in active:
            capacity = max(0.0, part.tokens * repeat - part.consumed)
            amount = min(step, capacity, remaining)
            part.consumed += amount
            allocated += amount
            remaining -= amount
            if remaining <= EPSILON:
                break
        if allocated <= EPSILON:
            break
    return initial - remaining


def rank_allocate(parts: List[Part], target: float, repeat: int) -> float:
    """Fill from the highest-scoring shard down, repeating full passes until target or repeat cap is hit."""
    remaining = max(0.0, target)
    initial = remaining
    ranked = sorted(
        (part for part in parts if part.tokens > 0),
        key=lambda part: (-(part.score if part.score is not None else float("-inf")), part.path),
    )
    for round_index in range(1, repeat + 1):
        if remaining <= EPSILON:
            break
        for part in ranked:
            if remaining <= EPSILON:
                break
            capacity = max(0.0, part.tokens * round_index - part.consumed)
            if capacity <= EPSILON:
                continue
            amount = min(capacity, remaining)
            part.consumed += amount
            remaining -= amount
    return initial - remaining


def allocate_matched(
    parts: List[Part],
    target: float,
    repeat: int,
    step: int,
    is_fill: bool,
    bsc_edu_range: Optional[Tuple[float, float]],
) -> None:
    spread = fill_in_steps if is_fill else proportional_allocate
    extra = (step,) if is_fill else ()
    if bsc_edu_range is None:
        spread(parts, target, repeat, *extra)
        return
    ranked = [part for part in parts if part.dataset == "hplt-4.0-bsc-edu"]
    rest = [part for part in parts if part.dataset != "hplt-4.0-bsc-edu"]
    if not ranked:
        spread(rest, target, repeat, *extra)
        return
    if not rest:
        rank_allocate(ranked, target, repeat)
        return
    total_tokens = sum(part.tokens for part in ranked) + sum(part.tokens for part in rest)
    if total_tokens <= 0:
        return
    ranked_target = target * sum(part.tokens for part in ranked) / total_tokens
    rank_allocate(ranked, ranked_target, repeat)
    spread(rest, target - ranked_target, repeat, *extra)


def allocate_group(
    parts: List[Part],
    target: float,
    limits: List[Rule],
    fills: List[Rule],
    repeat: int,
    step: int,
    bsc_edu_range: Optional[Tuple[float, float]] = None,
) -> None:
    initial_consumed = sum(part.consumed for part in parts)
    group_limits = limits if limits else ([] if fills else [Rule("all", 1.0, repeat)])
    for rule in group_limits:
        already_allocated = sum(part.consumed for part in parts) - initial_consumed
        remaining = max(0.0, target - already_allocated)
        if remaining <= EPSILON:
            break
        allocate_matched(
            matching(parts, rule.domain),
            min(target * rule.ratio, remaining),
            rule.repeat,
            step,
            is_fill=False,
            bsc_edu_range=bsc_edu_range,
        )
    for rule in fills:
        already_allocated = sum(part.consumed for part in parts) - initial_consumed
        remaining = max(0.0, target - already_allocated)
        if remaining <= EPSILON:
            break
        allocate_matched(
            matching(parts, rule.domain),
            remaining,
            rule.repeat,
            step,
            is_fill=True,
            bsc_edu_range=bsc_edu_range,
        )


def allocate(
    parts: List[Part],
    horizon: int,
    ratio: Tuple[float, float],
    distribution: str,
    limits: List[Rule],
    fills: List[Rule],
    repeat: int,
    step: int,
    bsc_edu_range: Optional[Tuple[float, float]] = None,
) -> float:
    dominant_target = horizon * ratio[0]
    multilingual_target = horizon * ratio[1]
    dominant_parts = [part for part in parts if not part.multilingual]
    multilingual_parts = [part for part in parts if part.multilingual]
    allocate_group(dominant_parts, dominant_target, limits, fills, repeat, step, bsc_edu_range)

    by_language = defaultdict(list)  # type: Dict[str, List[Part]]
    for part in multilingual_parts:
        by_language[part.language].append(part)
    available = {
        language: sum(part.tokens for part in language_parts)
        for language, language_parts in by_language.items()
    }
    if distribution == "equal" and by_language:
        each_target = multilingual_target / len(by_language)
        language_targets = {language: each_target for language in by_language}
    else:
        multilingual_available = sum(available.values())
        language_targets = {
            language: multilingual_target * tokens / multilingual_available
            for language, tokens in available.items()
        } if multilingual_available else {}
    for language, language_parts in by_language.items():
        allocate_group(
            language_parts,
            language_targets[language],
            limits,
            fills,
            repeat,
            step,
            bsc_edu_range,
        )
    return horizon


def format_tokens(value: float) -> str:
    return f"{value:,.0f}"


def write_output(
    stream: TextIO,
    parts: List[Part],
    horizon: float,
    args: argparse.Namespace,
    ratio: Tuple[float, float],
    limits: List[Rule],
    fills: List[Rule],
) -> None:
    allocated = sum(part.consumed for part in parts)
    stream.write("# generated weighted file list\n")
    stream.write(f"# inventory: {args.inventory}\n")
    stream.write(f"# ratio dominant:multilingual: {ratio[0] * 100:g}:{ratio[1] * 100:g}\n")
    stream.write(f"# distribution: {args.distribution}\n")
    stream.write(
        f"# horizon_tokens: {format_tokens(horizon)} @ scale {args.scale:g}\n"
    )
    stream.write(f"# allocated_tokens: {format_tokens(allocated)}\n")
    stream.write(f"# weight_sum: {allocated / horizon * args.scale:.8f}\n")
    stream.write(f"# repeat: {args.repeat}; step: {args.step}\n")
    bsc_edu_range_text = (
        f"{args.bsc_edu_range[0]:g}-{args.bsc_edu_range[1]:g}"
        if args.bsc_edu_range
        else "disabled (sampled like other datasets)"
    )
    stream.write(
        f"# bsc_edu_range: {bsc_edu_range_text}; bsc_edu_partition: {args.bsc_edu_partition}\n"
    )
    limit_text = ", ".join(
        "{}:{}:{}".format(rule.domain, rule.ratio, rule.repeat) for rule in limits
    )
    fill_text = ", ".join(
        "{}:{}".format(rule.domain, rule.repeat) for rule in fills
    )
    stream.write(f"# multilingual.py: limits: {limit_text}.\n")
    stream.write(f"# multilingual.py: fills: {fill_text}.\n")
    stream.write(f"# excluded: {', '.join(sorted(args.exclude)) if args.exclude else 'none'}\n")
    dataset_totals = defaultdict(lambda: [0.0, 0.0])
    for part in parts:
        totals = dataset_totals[(part.collection, part.dataset)]
        totals[0] += part.tokens
        totals[1] += part.consumed
    dataset_text = ", ".join(
        "{}/{}:{:.6f}:{:.2f}x".format(
            collection,
            dataset,
            consumed / horizon * args.scale,
            consumed / available,
        )
        for (collection, dataset), (available, consumed) in sorted(dataset_totals.items())
        if consumed > EPSILON and available > 0
    )
    stream.write(
        "# multilingual.py: datasets (collection/dataset:weight:effective_repetitions): "
        "{}\n".format(dataset_text or "none")
    )

    if args.debug:
        available_languages = defaultdict(float)  # type: Dict[str, float]
        used_languages = defaultdict(float)  # type: Dict[str, float]
        available_domains = defaultdict(float)  # type: Dict[str, float]
        used_domains = defaultdict(float)  # type: Dict[str, float]
        for part in parts:
            available_languages[part.language] += part.tokens
            used_languages[part.language] += part.consumed
            available_domains[part.domain] += part.tokens
            used_domains[part.domain] += part.consumed
        stream.write("# language_tokens_available / allocated:\n")
        for language in sorted(available_languages):
            used = used_languages[language]
            pct = used / allocated * 100 if allocated else 0.0
            stream.write(
                f"#   {language}: {format_tokens(available_languages[language])} / "
                f"{format_tokens(used)} ({pct:.2f}%)\n"
            )
        for domain in ("math", "code"):
            used = used_domains[domain]
            pct = used / allocated * 100 if allocated else 0.0
            stream.write(
                f"#   {domain}: {format_tokens(available_domains[domain])} / "
                f"{format_tokens(used)} ({pct:.2f}%)\n"
            )
        stream.write("# domain_tokens_available / allocated:\n")
        for domain in ("wiki", "web", "mt", "pdf", "parallel", "math", "code"):
            used = used_domains[domain]
            pct = used / allocated * 100 if allocated else 0.0
            stream.write(
                f"#   {domain}: {format_tokens(available_domains[domain])} / "
                f"{format_tokens(used)} ({pct:.2f}%)\n"
            )

    for part in sorted(parts, key=lambda item: item.path):
        if part.consumed <= EPSILON:
            continue
        weight = part.consumed / horizon * args.scale
        line = f"{weight:.6f} {part.path}"
        if args.debug:
            percentage = 100 * part.consumed / part.tokens if part.tokens else 0.0
            line += (
                f" # {format_tokens(part.consumed)} / {format_tokens(part.tokens)} "
                f"tokens ({percentage:.2f}%)"
            )
        stream.write(line + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a weighted Megatron-LM file list from inventory byte sizes."
    )
    parser.add_argument("--inventory", type=Path, default=SCRIPT_DIR / "inventory.csv")
    parser.add_argument("--languages", type=Path, default=SCRIPT_DIR / "languages.txt")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "weights.txt")
    parser.add_argument(
        "--horizon",
        type=lambda value: int(float(value)),
        default=int(2e12),
        help="target token horizon before applying --scale",
    )
    parser.add_argument(
        "--scale", type=float, default=0.2, help="scale factor applied to horizon and weights"
    )
    parser.add_argument(
        "--limit",
        action="append",
        default=[],
        metavar="DOMAIN:RATIO[:REPEAT]",
        help="initial per-domain quota within each language pool; repeatable",
    )
    parser.add_argument(
        "--fill",
        action="append",
        default=[],
        metavar="DOMAIN[:REPEAT]",
        help="domain(s) eligible to fill remaining quota in --step increments; repeatable",
    )
    parser.add_argument("--step", type = lambda _: int(float(_)), default=int(1e6), help="fill increment in tokens")
    parser.add_argument("--repeat", type=int, default=2, help="maximum source passes by default")
    parser.add_argument("--ratio", default="60:40", help="dominant:multilingual, e.g. 60:40")
    parser.add_argument(
        "--distribution",
        choices=("hegemonic", "equal"),
        default="hegemonic",
        help="split multilingual tokens naturally or equally by language",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        metavar="DATASET",
        help="only include these dataset names; may be combined with --exclude",
    )
    parser.add_argument("--exclude", nargs="+", default=[], metavar="DATASET")
    parser.add_argument(
        "--bsc-edu-range",
        type=parse_score_range,
        default=None,
        metavar="MIN-MAX",
        help=(
            "only use hplt-4.0-bsc-edu shards whose bsc_edu score falls in MIN-MAX "
            "(e.g. 1.4-4.0), sampled top-down; omit to sample them like any other dataset"
        ),
    )
    parser.add_argument(
        "--bsc-edu-partition",
        choices=("clean", "noisy", "both"),
        default="clean",
        help="restrict hplt-4.0-bsc-edu shards to the clean or noisy partition (default: clean)",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat < 1 or args.step < 1 or args.horizon < 1 or args.scale <= 0:
        parser.error("--repeat, --step, --horizon, and --scale must be positive")
    try:
        ratio = parse_ratio(args.ratio)
        code_to_language = read_language_codes(args.languages)
        parts = read_inventory(
            args.inventory,
            set(args.exclude),
            set(args.include),
            code_to_language,
            args.bsc_edu_range,
            args.bsc_edu_partition,
        )
        limits = [parse_limit(value, args.repeat) for value in args.limit]
        fills = [parse_fill(value, args.repeat) for value in args.fill]
        horizon = round(args.horizon * args.scale)
        if horizon <= 0:
            raise ValueError("--horizon multiplied by --scale must round to a positive value")
        allocate(
            parts,
            horizon,
            ratio,
            args.distribution,
            limits,
            fills,
            args.repeat,
            args.step,
            args.bsc_edu_range,
        )
        if str(args.output) == "-":
            write_output(sys.stdout, parts, horizon, args, ratio, limits, fills)
        else:
            with args.output.open("w", encoding="utf-8") as stream:
                write_output(stream, parts, horizon, args, ratio, limits, fills)
    except (OSError, ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())