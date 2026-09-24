#!/usr/bin/env python
"""Plot model evaluation performance across training checkpoints.

Reads one or more ``results.csv`` files produced by ``oellm-eval collect``
(columns: model_name, task, n_shot, performance, metric_name) and produces a
line plot per benchmark family showing how each model's score evolved across
intermediate checkpoints. For multilingual benchmarks, scores are
macro-averaged across languages by default; per-language and single-language
views are also supported.
The consolidated macro-average plot combines the per-family scores into one line per model:
``--summary zscore`` (default) per-family z-score normalization, or ``--summary naive`` plain mean of the
raw family scores. In naive mode, families whose task scores extend beyond
the 0-1 range (e.g. BLEU or chrf++ reported on a 0-100 scale) are first
rescaled to 0-1 (a BLEU of 21 counts as 0.21); this rescaling affects only
the naive macro-average plot, all other plots keep the raw scores.

Designed to run inside the shared LAIF ROCm container. The benchmark family <-> language
mapping is embedded (parsed from oellm-eval's ``task-groups.yaml``) so the
script is self-contained with no extra bind mounts.

Saves the source scores as a single `eval_results.csv` file in the current directory.

Example usage (inside container, working bind):

    singularity exec \
      --bind /pfs/lustrep4/scratch/project_465002891:/scratch/project_465002891 \
      /scratch/project_465002530/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif \
      python plot_eval_progress.py \
        --input results.csv \
        --output_dir plots/ \
        --origin_checkpoint /scratch/project_465002891/prelude-mid/hf_models/baby_9b_dense_before-annealing/checkpoints/iter_0953312
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe inside a container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Benchmark family <-> language mapping.
#
# Each entry: family_key -> dict(families=[...], template=<str with {lang}>,
# langs=[...], n_shot=int, or a list of shot variants for a task evaluated
# at several shot counts (rendered '0/10' in plot titles). A family_key groups
# tasks that share a benchmark;
# directional benchmarks (flores200, opensubtitles) are split into per-direction
# family_keys. ``template`` is the lm_eval/lighteval task name with ``{lang}``
# as the language placeholder; expanding it over ``langs`` yields the concrete
# task names. Tasks absent from any family are treated as their own
# single-language family (lang=None).
#
# Generated from oellm-eval resources/task-groups.yaml (oellm_eval-0.1.0).
# --------------------------------------------------------------------------- #
FAMILIES: Dict[str, dict] = {
    "sib200": {
        "template": "sib200_{lang}",
        "langs": [
            "bul_Cyrl",
            "hrv_Latn",
            "ces_Latn",
            "dan_Latn",
            "nld_Latn",
            "eng_Latn",
            "est_Latn",
            "fin_Latn",
            "fra_Latn",
            "deu_Latn",
            "ell_Grek",
            "hun_Latn",
            "gle_Latn",
            "ita_Latn",
            "lvs_Latn",
            "lit_Latn",
            "mlt_Latn",
            "pol_Latn",
            "por_Latn",
            "ron_Latn",
            "slk_Latn",
            "slv_Latn",
            "spa_Latn",
            "swe_Latn",
            "cat_Latn",
            "eus_Latn",
            "glg_Latn",
            "bos_Latn",
            "kat_Geor",
            "mkd_Cyrl",
            "als_Latn",
            "srp_Cyrl",
            "tur_Latn",
            "ukr_Cyrl",
            "isl_Latn",
            "nob_Latn",
        ],
        "n_shot": 0,
        "metric": "acc_norm",
    },
    "belebele": {
        "template": "belebele_{lang}",
        "langs": [
            "bul_Cyrl",
            "hrv_Latn",
            "ces_Latn",
            "dan_Latn",
            "nld_Latn",
            "eng_Latn",
            "est_Latn",
            "fin_Latn",
            "fra_Latn",
            "deu_Latn",
            "ell_Grek",
            "hun_Latn",
            "ita_Latn",
            "lvs_Latn",
            "lit_Latn",
            "mlt_Latn",
            "pol_Latn",
            "por_Latn",
            "ron_Latn",
            "slk_Latn",
            "slv_Latn",
            "spa_Latn",
            "swe_Latn",
            "nob_Latn",
        ],
        "n_shot": 5,
        "metric": "acc",
    },
    "xcsqa": {
        "template": "xcsqa_{lang}",
        "langs": [
            "deu_Latn",
            "eng_Latn",
            "spa_Latn",
            "fra_Latn",
            "ita_Latn",
            "nld_Latn",
            "pol_Latn",
            "por_Latn",
        ],
        "n_shot": 0,
        "metric": "acc_norm",
    },
    "global_mmlu_full": {
        "template": "global_mmlu_full_{lang}",
        "langs": [
            "cs",
            "de",
            "el",
            "en",
            "es",
            "fr",
            "it",
            "lt",
            "nl",
            "pl",
            "pt",
            "ro",
            "sr",
            "sv",
            "tr",
            "uk",
        ],
        "n_shot": 5,
        "metric": "acc",
    },
    "global_mgsm": {
        "template": "global_mgsm_{lang}",
        "langs": ["de", "fr", "es", "el", "cs", "hu", "en", "ca", "eu", "gl", "sr"],
        "n_shot": 0,
        "metric": "exact_match",
    },
    "mgsm_native_cot": {
        "template": "mgsm_native_cot_{lang}",
        "langs": ["en", "de", "es", "fr"],
        "n_shot": 5,
        "metric": "exact_match",
    },
    "multiblimp": {
        "template": "multiblimp_{lang}",
        "langs": [
            "bul",
            "ces",
            "dan",
            "nld",
            "eng",
            "est",
            "fin",
            "fra",
            "deu",
            "ell",
            "hun",
            "gle",
            "ita",
            "lav",
            "lit",
            "pol",
            "por",
            "ron",
            "slk",
            "slv",
            "spa",
            "swe",
            "cat",
            "eus",
            "glg",
            "kat",
            "mkd",
            "sqi",
            "hbs",
            "tur",
            "ukr",
            "isl",
        ],
        "n_shot": 0,
        "metric": "acc_norm",
    },
    "arc_challenge_mt": {
        "template": "arc_challenge_mt_{lang}",
        "langs": [
            "bg",
            "cs",
            "da",
            "nl",
            "et",
            "fi",
            "fr",
            "de",
            "el",
            "hu",
            "it",
            "lv",
            "lt",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
            "es",
            "sv",
            "nb",
            "is",
        ],
        "n_shot": 0,
        "metric": "acc_norm",
    },
    "hellaswag": {
        "template": "hellaswag_{lang}",
        "langs": [
            "da",
            "nl",
            "fr",
            "de",
            "hu",
            "it",
            "pt",
            "ro",
            "sk",
            "es",
            "sv",
            "hr",
            "ca",
            "eu",
            "sr",
            "uk",
        ],
        "n_shot": 0,
        "metric": "acc_norm",
    },
    "xcopa": {
        "template": "xcopa:{lang}",
        "langs": ["et", "it", "tr"],
        "n_shot": 0,
        "metric": "acc",
    },
    "global_piqa_completions": {
        "template": "global_piqa_completions_{lang}",
        "langs": [
            "als_latn",
            "bos_latn",
            "bul_cyrl",
            "cat_latn",
            "ces_latn",
            "deu_latn",
            "ekk_latn",
            "ell_grek",
            "eng_latn",
            "fin_latn",
            "fra_latn_fran",
            "glg_latn",
            "hrv_latn",
            "hun_latn",
            "isl_latn",
            "ita_latn",
            "kat_geor",
            "lit_latn",
            "mkd_cyrl",
            "nld_latn",
            "nno_latn",
            "nob_latn",
            "pol_latn",
            "por_latn_port",
            "ron_latn",
            "slk_latn",
            "slv_latn",
            "spa_latn_spai",
            "srp_cyrl",
            "swe_latn",
            "tur_latn",
            "ukr_cyrl",
        ],
        "n_shot": 0,
        "metric": "acc_norm",
    },
    "global_piqa_prompted": {
        "template": "global_piqa_prompted_{lang}",
        "langs": [
            "als_latn",
            "bos_latn",
            "bul_cyrl",
            "cat_latn",
            "ces_latn",
            "deu_latn",
            "ekk_latn",
            "ell_grek",
            "eng_latn",
            "fin_latn",
            "fra_latn_fran",
            "glg_latn",
            "hrv_latn",
            "hun_latn",
            "isl_latn",
            "ita_latn",
            "kat_geor",
            "lit_latn",
            "mkd_cyrl",
            "nld_latn",
            "nno_latn",
            "nob_latn",
            "pol_latn",
            "por_latn_port",
            "ron_latn",
            "slk_latn",
            "slv_latn",
            "spa_latn_spai",
            "srp_cyrl",
            "swe_latn",
            "tur_latn",
            "ukr_cyrl",
        ],
        "n_shot": 0,
        "metric": "exact_match",
    },
    "include_base_44": {
        "template": "include_base_44_{lang}",
        "langs": [
            "albanian",
            "basque",
            "bulgarian",
            "croatian",
            "dutch",
            "estonian",
            "finnish",
            "french",
            "georgian",
            "german",
            "greek",
            "hungarian",
            "italian",
            "lithuanian",
            "north macedonian",
            "polish",
            "portuguese",
            "serbian",
            "spanish",
            "turkish",
            "ukrainian",
        ],
        "n_shot": 0,
        "metric": "acc",
    },
    # --- directional benchmarks: one family per direction ------------------ #
    "flores200_x_to_en": {
        "template": "flores200:{lang}-eng_Latn",
        "langs": [
            "bul_Cyrl",
            "hrv_Latn",
            "ces_Latn",
            "dan_Latn",
            "nld_Latn",
            "est_Latn",
            "fin_Latn",
            "fra_Latn",
            "deu_Latn",
            "ell_Grek",
            "hun_Latn",
            "gle_Latn",
            "ita_Latn",
            "lvs_Latn",
            "lit_Latn",
            "mlt_Latn",
            "pol_Latn",
            "por_Latn",
            "ron_Latn",
            "slk_Latn",
            "slv_Latn",
            "spa_Latn",
            "swe_Latn",
            "cat_Latn",
            "eus_Latn",
            "glg_Latn",
            "bos_Latn",
            "kat_Geor",
            "mkd_Cyrl",
            "als_Latn",
            "srp_Cyrl",
            "tur_Latn",
            "ukr_Cyrl",
            "isl_Latn",
            "nob_Latn",
        ],
        "n_shot": 0,
        "metric": "chrf++",
        "direction": "X\u2192en",
    },
    "flores200_en_to_x": {
        "template": "flores200:eng_Latn-{lang}",
        "langs": [
            "bul_Cyrl",
            "hrv_Latn",
            "ces_Latn",
            "dan_Latn",
            "nld_Latn",
            "est_Latn",
            "fin_Latn",
            "fra_Latn",
            "deu_Latn",
            "ell_Grek",
            "hun_Latn",
            "gle_Latn",
            "ita_Latn",
            "lvs_Latn",
            "lit_Latn",
            "mlt_Latn",
            "pol_Latn",
            "por_Latn",
            "ron_Latn",
            "slk_Latn",
            "slv_Latn",
            "spa_Latn",
            "swe_Latn",
            "cat_Latn",
            "eus_Latn",
            "glg_Latn",
            "bos_Latn",
            "kat_Geor",
            "mkd_Cyrl",
            "als_Latn",
            "srp_Cyrl",
            "tur_Latn",
            "ukr_Cyrl",
            "isl_Latn",
            "nob_Latn",
        ],
        "n_shot": 0,
        "metric": "chrf++",
        "direction": "en\u2192X",
    },
    "opensubtitles_x_to_en": {
        "template": "opensubtitles_multi40_{lang}_to_en",
        "langs": [
            "bg",
            "hr",
            "cs",
            "da",
            "nl",
            "et",
            "fi",
            "fr",
            "de",
            "el",
            "hu",
            "it",
            "lv",
            "lt",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
            "es",
            "sv",
            "sr",
            "tr",
            "uk",
            "no",
        ],
        "n_shot": 0,
        "metric": "bleu",
        "direction": "X\u2192en",
    },
    "opensubtitles_en_to_x": {
        "template": "opensubtitles_multi40_en_to_{lang}",
        "langs": [
            "bg",
            "hr",
            "cs",
            "da",
            "nl",
            "et",
            "fi",
            "fr",
            "de",
            "el",
            "hu",
            "it",
            "lv",
            "lt",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
            "es",
            "sv",
            "sr",
            "tr",
            "uk",
            "no",
        ],
        "n_shot": 0,
        "metric": "bleu",
        "direction": "en\u2192X",
    },
    "polymath": {
        "template": "polymath_{lang}_{tier}",
        "langs": ["de", "en", "es", "fr", "it", "pt"],
        "n_shot": 0,
        "metric": "exact_match",
        "tiers": ["low", "medium", "high", "top"],
    },
    # --- dclm-core-22: single-task (English) benchmarks, one family per
    # task (group-prefixed key, lang=None). n_shot per task-groups.yaml
    # (hellaswag is evaluated at both 0 and 10 shots — the variants
    # average into one family and its plot label shows n_shot=0/10);
    # metric per task_metrics, omitted where undeclared (jeopardy) so the
    # y-label falls back to the metric observed in the data. ------------- #
    "dclm_core_22_agieval_lsat_ar": {
        "template": "agieval_lsat_ar",
        "langs": [None],
        "n_shot": 3,
        "metric": "acc",
    },
    "dclm_core_22_arc_easy": {
        "template": "arc_easy",
        "langs": [None],
        "n_shot": 10,
        "metric": "acc_norm",
    },
    "dclm_core_22_arc_challenge": {
        "template": "arc_challenge",
        "langs": [None],
        "n_shot": 10,
        "metric": "acc_norm",
    },
    "dclm_core_22_boolq": {
        "template": "boolq",
        "langs": [None],
        "n_shot": 10,
        "metric": "acc",
    },
    "dclm_core_22_commonsense_qa": {
        "template": "commonsense_qa",
        "langs": [None],
        "n_shot": 10,
        "metric": "acc",
    },
    "dclm_core_22_copa": {
        "template": "copa",
        "langs": [None],
        "n_shot": 0,
        "metric": "acc",
    },
    "dclm_core_22_hellaswag": {
        "template": "hellaswag",
        "langs": [None],
        "n_shot": [0, 10],
        "metric": "acc_norm",
    },
    "dclm_core_22_openbookqa": {
        "template": "openbookqa",
        "langs": [None],
        "n_shot": 0,
        "metric": "acc_norm",
    },
    "dclm_core_22_piqa": {
        "template": "piqa",
        "langs": [None],
        "n_shot": 10,
        "metric": "acc_norm",
    },
    "dclm_core_22_bigbench_language_identification_multiple_choice": {
        "template": "bigbench_language_identification_multiple_choice",
        "langs": [None],
        "n_shot": 10,
        "metric": "acc",
    },
    "dclm_core_22_winogrande": {
        "template": "winogrande",
        "langs": [None],
        "n_shot": 0,
        "metric": "acc",
    },
    "dclm_core_22_wsc273": {
        "template": "wsc273",
        "langs": [None],
        "n_shot": 0,
        "metric": "acc",
    },
    "dclm_core_22_lambada_openai": {
        "template": "lambada_openai",
        "langs": [None],
        "n_shot": 0,
        "metric": "acc",
    },
    "dclm_core_22_bigbench_qa_wikidata_generate_until": {
        "template": "bigbench_qa_wikidata_generate_until",
        "langs": [None],
        "n_shot": 10,
        "metric": "exact_match",
    },
    "dclm_core_22_bigbench_dyck_languages_generate_until": {
        "template": "bigbench_dyck_languages_generate_until",
        "langs": [None],
        "n_shot": 10,
        "metric": "exact_match",
    },
    "dclm_core_22_bigbench_operators_generate_until": {
        "template": "bigbench_operators_generate_until",
        "langs": [None],
        "n_shot": 10,
        "metric": "exact_match",
    },
    "dclm_core_22_bigbench_repeat_copy_logic_generate_until": {
        "template": "bigbench_repeat_copy_logic_generate_until",
        "langs": [None],
        "n_shot": 10,
        "metric": "exact_match",
    },
    "dclm_core_22_bigbench_cs_algorithms_generate_until": {
        "template": "bigbench_cs_algorithms_generate_until",
        "langs": [None],
        "n_shot": 10,
        "metric": "exact_match",
    },
    "dclm_core_22_coqa": {
        "template": "coqa",
        "langs": [None],
        "n_shot": 0,
        "metric": "f1",
    },
    "dclm_core_22_squadv2": {
        "template": "squadv2",
        "langs": [None],
        "n_shot": 10,
        "metric": "f1",
    },
    "dclm_core_22_jeopardy": {
        "template": "jeopardy",
        "langs": [None],
        "n_shot": 10,
    },
    # --- reasoning: single-task (English) benchmarks, one family per task
    # (group-prefixed key, lang=None). n_shot per task-groups.yaml; metrics
    # are not declared there, so they are omitted and the y-label falls
    # back to the metric observed in the data. ---------------------------- #
    "reasoning_gsm8k": {
        "template": "gsm8k",
        "langs": [None],
        "n_shot": 4,
    },
    "reasoning_ifeval": {
        "template": "ifeval",
        "langs": [None],
        "n_shot": 0,
    },
    "reasoning_mbpp": {
        "template": "mbpp",
        "langs": [None],
        "n_shot": 3,
    },
    "reasoning_GPQADiamond": {
        "template": "GPQADiamond",
        "langs": [None],
        "n_shot": 0,
    },
    "reasoning_MATH500": {
        "template": "MATH500",
        "langs": [None],
        "n_shot": 0,
    },
    "reasoning_LiveCodeBench": {
        "template": "LiveCodeBench",
        "langs": [None],
        "n_shot": 0,
    },
    "reasoning_HumanEval": {
        "template": "HumanEval",
        "langs": [None],
        "n_shot": 0,
    },
    "reasoning_AIME24": {
        "template": "AIME24",
        "langs": [None],
        "n_shot": 0,
    },
    "reasoning_AIME25": {
        "template": "AIME25",
        "langs": [None],
        "n_shot": 0,
    },
    "reasoning_AMC23": {
        "template": "AMC23",
        "langs": [None],
        "n_shot": 0,
    },
}


def _build_task_index() -> Dict[str, List]:
    """Reverse map: task_name -> (family_key, lang)."""
    idx: Dict[str, Tuple[str, str]] = {}
    for fam_key, spec in FAMILIES.items():
        tmpl = spec["template"]
        if "tiers" in spec:
            for tier in spec["tiers"]:
                for lang in spec["langs"]:
                    idx[tmpl.format(lang=lang, tier=tier)] = (fam_key, lang, tier)
        else:
            for lang in spec["langs"]:
                idx[tmpl.format(lang=lang)] = (fam_key, lang)
    return idx


_TASK_INDEX = _build_task_index()

# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
_ITER_RE = re.compile(r"iter_(\d+)")
_HF_MODELS_RE = re.compile(r"/hf_models/([^/]+)")


def parse_model(model_name: str) -> Tuple[str, Optional[int]]:
    """Return (run_name, step). step is None if no iter_ token is present."""
    m_iter = _ITER_RE.search(model_name)
    step = int(m_iter.group(1)) if m_iter else None
    m_run = _HF_MODELS_RE.search(model_name)
    if m_run:
        run = m_run.group(1)
    else:
        # Fallback: use the final path component (without iter_ suffix).
        base = model_name.rstrip("/").split("/")[-1]
        run = _ITER_RE.sub("", base).strip("_") or base
    return run, step


def _base_metric(metric_name: str) -> str:
    """'acc_norm,none' -> 'acc_norm', 'chrf++' -> 'chrf++'."""
    if not isinstance(metric_name, str):
        return ""
    return metric_name.split(",", 1)[0]


def classify_task(task: str) -> Tuple[str, Optional[str]]:
    """Return (family_key, lang). Unmatched task -> (task, None)."""
    hit = _TASK_INDEX.get(task)
    if hit is not None:
        return hit
    # Fallback: single-language benchmark (task is its own family).
    return task, None


# --------------------------------------------------------------------------- #
# Origin checkpoint (common starting point of all model lines)
# --------------------------------------------------------------------------- #
# Training step used for the prepended origin point when the checkpoint path
# (or its resolved model_name) carries no iter_ token. Matches
# baby_9b_dense_before-annealing/checkpoints/iter_0953312.
DEFAULT_ORIGIN_STEP = 953312


def _hf_models_tail(name: str) -> str:
    """Path identity from ``/hf_models/`` onwards (ignores the mount-point
    prefix, so /pfs/lustrep4/scratch/... and /scratch/... compare equal)."""
    name = str(name).rstrip("/")
    i = name.find("/hf_models/")
    return name[i:] if i != -1 else name


def _resolve_origin_model(df: pd.DataFrame, path: str) -> str:
    """Map a user-supplied checkpoint path to a model_name present in the data.

    Matching order: exact string, /hf_models/ tail, (run, step) via
    parse_model, then a unique run-only match (with a warning). Raises
    SystemExit listing the evaluated models if nothing matches."""
    candidates = list(df["model_name"].unique())
    p = str(path).rstrip("/")
    if p in candidates:
        return p
    tail = _hf_models_tail(p)
    for c in candidates:
        if _hf_models_tail(c) == tail:
            return c
    run, step = parse_model(p)
    for c in candidates:
        c_run, c_step = parse_model(c)
        if c_run == run and step is not None and c_step == step:
            return c
    same_run = [c for c in candidates if parse_model(c)[0] == run]
    if len(same_run) == 1:
        print(
            f"[warn] --origin_checkpoint: no exact match for {p}; using the "
            f"only evaluated checkpoint of run '{run}': {same_run[0]}"
        )
        return same_run[0]
    avail = "\n  ".join(sorted(candidates))
    raise SystemExit(
        f"[error] --origin_checkpoint: '{path}' is not among the evaluated "
        f"models. Evaluated model_names:\n  {avail}"
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def gather_input_files(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.rglob("results.csv")))
        elif p.is_file() and p.name == "results.csv":
            paths.append(p)
        elif p.is_file():
            # Accept any CSV the user points us at.
            paths.append(p)
        else:
            print(f"[warn] input not found: {inp}", file=sys.stderr)
    # de-duplicate while preserving order
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def load_results(inputs: List[str]) -> pd.DataFrame:
    files = gather_input_files(inputs)
    if not files:
        raise SystemExit(
            "No results.csv files found. Pass --input <file|dir> (repeatable)."
        )
    frames: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = str(f)
        frames.append(df)
        print(f"[info] loaded {len(df)} rows from {f}")
    df = pd.concat(frames, ignore_index=True)
    required = {"model_name", "task", "n_shot", "performance", "metric_name"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    # Dedup: later files / later rows win (matches oellm-eval collect semantics).
    df = df.drop_duplicates(
        subset=["model_name", "task", "n_shot"], keep="last"
    ).reset_index(drop=True)
    df.to_csv("eval_results.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Annotation / aggregation
# --------------------------------------------------------------------------- #
def annotate(df: pd.DataFrame) -> pd.DataFrame:
    runs_steps = df["model_name"].map(parse_model)
    df = df.assign(
        run=runs_steps.map(lambda x: x[0]),
        step=runs_steps.map(lambda x: x[1]),
    )
    fam_lang = df["task"].map(classify_task)
    df = df.assign(
        family=fam_lang.map(lambda x: x[0]),
        lang=fam_lang.map(lambda x: x[1]),
        base_metric=df["metric_name"].map(_base_metric),
    )
    return df


def family_display_name(fam_key: str) -> str:
    spec = FAMILIES.get(fam_key)
    if not spec:
        return fam_key
    name = fam_key
    if "direction" in spec:
        base = fam_key.rsplit("_", 2)[0]  # flores200_x_to_en -> flores200
        return f"{base} ({spec['direction']})"
    if "tier" in spec:
        base = fam_key.rsplit("_", 1)[0]  # polymath_low -> polymath
        return f"{base} [{spec['tier']}]"
    return name


def family_metric(fam_key: str, df_fam: pd.DataFrame) -> str:
    """Metric to show on the y-axis: the family's declared metric if known,
    else the most common base_metric observed in the data."""
    spec = FAMILIES.get(fam_key)
    if spec and spec.get("metric"):
        return spec["metric"]
    if not df_fam.empty:
        m = df_fam["base_metric"].mode()
        if not m.empty:
            return str(m.iloc[0])
    return "score"


def _n_shot_label(spec: Optional[dict], df_fam: pd.DataFrame) -> str:
    """n_shot label for a family plot: the family's declared n_shot if the
    spec has one (a list of shot variants renders as '0/10'), else the most
    common n_shot observed in the data."""
    if spec and spec.get("n_shot") is not None:
        v = spec["n_shot"]
        if isinstance(v, (list, tuple)):
            return "/".join(str(s) for s in v)
        return str(v)
    return str(int(df_fam["n_shot"].mode().iloc[0] or 0))


def aggregate_averaged(df: pd.DataFrame) -> pd.DataFrame:
    """Macro-average performance across languages within each
    (family, run, step). Returns one row per (family, run, step)."""
    cols = ["family", "run", "step", "base_metric"]
    g = df.groupby(cols, dropna=False)["performance"].mean().reset_index()
    g = g.rename(columns={"performance": "score"})
    return g


def aggregate_per_language(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (family, lang, run, step)."""
    cols = ["family", "lang", "run", "step", "base_metric"]
    g = (
        df.groupby(cols, dropna=False)["performance"]
        .mean()
        .reset_index()
        .rename(columns={"performance": "score"})
    )
    return g


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
# Stable colour cycle across families so a given model keeps the same colour.
_GLOBAL_CMAP = plt.get_cmap("tab20")

_LEGEND_STRIP_PREFIX = "baby_9b_dense"


def _legend_label(run: str) -> str:
    if run.startswith(_LEGEND_STRIP_PREFIX):
        return run.removeprefix(_LEGEND_STRIP_PREFIX).lstrip("_-")
    return run


def _model_colour_map(models: List[str]) -> Dict[str, Tuple[float, ...]]:
    models = sorted(models)
    n = len(models)
    out: Dict[str, Tuple[float, ...]] = {}
    for i, m in enumerate(models):
        if n == 1:
            out[m] = _GLOBAL_CMAP(0)
        elif n <= 10:
            out[m] = plt.get_cmap("tab10")(i % 10)
        else:
            out[m] = _GLOBAL_CMAP(i / max(1, n))
    return out


def _with_origin(
    sub: pd.DataFrame,
    model: str,
    origin: Optional[Tuple[str, int, object]],
    key: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """(steps, scores) for one model's series, with the origin checkpoint's
    point (origin_step, origin_score) prepended as the first (leftmost) point.

    ``origin`` is (origin_run, origin_step, score_getter) where
    ``score_getter(key)`` returns the origin's score for this family/language
    (or None). The point is skipped for the origin run itself, when no score
    is available, or when the model already has its own point at origin_step."""
    steps = sub["step"].to_numpy()
    scores = sub["score"].to_numpy()
    if origin is None or model == origin[0]:
        return steps, scores
    s0 = origin[2](key)
    if s0 is None or (isinstance(s0, float) and np.isnan(s0)):
        return steps, scores
    if origin[1] in steps:
        return steps, scores
    steps = np.concatenate(([origin[1]], steps))
    scores = np.concatenate(([s0], scores))
    return steps, scores


def _plot_one_family(
    fam_key: str,
    df_fam: pd.DataFrame,
    models: List[str],
    colour_map: Dict[str, Tuple[float, ...]],
    out_dir: Path,
    dpi: int,
    pdf: bool,
    suffix: str = "",
    origin: Optional[Tuple[str, int, object]] = None,
) -> Optional[Path]:
    """Plot the averaged view for one family. Returns saved PNG path or None."""
    if df_fam.empty:
        return None
    metric = family_metric(fam_key, df_fam)
    fam_title = family_display_name(fam_key)
    spec = FAMILIES.get(fam_key)
    n_shot = _n_shot_label(spec, df_fam)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    has_line = False
    only_points = False
    for m in models:
        sub = df_fam[df_fam["run"] == m].sort_values("step")
        if sub.empty:
            continue
        steps, scores = _with_origin(sub, m, origin, fam_key)
        if len(steps) == 1 or pd.isna(steps).any():
            only_points = True
            ax.scatter(
                steps, scores, color=colour_map[m], label=_legend_label(m), zorder=3
            )
        else:
            ax.plot(
                steps,
                scores,
                marker="o",
                color=colour_map[m],
                label=_legend_label(m),
                zorder=3,
            )
            has_line = True

    ax.set_title(f"{fam_title}  (n_shot={n_shot})")
    ax.set_xlabel("Training step")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    if only_points and not has_line:
        ax.text(
            0.5,
            0.97,
            "single checkpoint \u2014 markers only",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="0.4",
        )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    safe = re.sub(r"[^A-Za-z0-9_.\-\u2192\[\]]+", "_", fam_title)
    stem = f"{safe}{suffix}"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=dpi)
    if pdf:
        fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    return png_path


def _plot_by_language(
    fam_key: str,
    df_fam_lang: pd.DataFrame,
    models: List[str],
    colour_map: Dict[str, Tuple[float, ...]],
    out_dir: Path,
    dpi: int,
    pdf: bool,
    origin: Optional[Tuple[str, int, object]] = None,
) -> Optional[Path]:
    """Small-multiples: one subplot per language for one family."""
    if df_fam_lang.empty:
        return None
    langs = [l for l in sorted(df_fam_lang["lang"].dropna().unique())]
    if not langs:
        return None
    metric = family_metric(fam_key, df_fam_lang)
    fam_title = family_display_name(fam_key)
    spec = FAMILIES.get(fam_key)
    n_shot = _n_shot_label(spec, df_fam_lang)

    n = len(langs)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False
    )
    for idx, lang in enumerate(langs):
        ax = axes[idx // ncol][idx % ncol]
        for m in models:
            sub = df_fam_lang[
                (df_fam_lang["lang"] == lang) & (df_fam_lang["run"] == m)
            ].sort_values("step")
            if sub.empty:
                continue
            steps, scores = _with_origin(sub, m, origin, lang)
            if len(steps) == 1 or pd.isna(steps).any():
                ax.scatter(steps, scores, color=colour_map[m], s=18, zorder=3)
            else:
                ax.plot(
                    steps,
                    scores,
                    marker="o",
                    markersize=3,
                    color=colour_map[m],
                    zorder=3,
                )
        ax.set_title(lang, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    # hide unused axes
    for j in range(len(langs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(
        f"{fam_title} by language  (n_shot={n_shot}, metric={metric})", fontsize=11
    )
    # shared legend
    handles = [
        plt.Line2D([0], [0], color=colour_map[m], marker="o", label=_legend_label(m))
        for m in models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(models), 6),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    safe = re.sub(r"[^A-Za-z0-9_.\-\u2192\[\]]+", "_", fam_title)
    png_path = out_dir / f"{safe}__bylang.png"
    fig.savefig(png_path, dpi=dpi)
    if pdf:
        fig.savefig(out_dir / f"{safe}__bylang.pdf")
    plt.close(fig)
    return png_path


def _plot_overview_grid(
    agg: pd.DataFrame,
    models: List[str],
    colour_map: Dict[str, Tuple[float, ...]],
    out_dir: Path,
    dpi: int,
    origin: Optional[Tuple[str, int, object]] = None,
) -> Optional[Path]:
    """One figure with a small subplot per family (averaged view)."""
    fams = sorted(agg["family"].unique())
    if not fams:
        return None
    n = len(fams)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False
    )
    for idx, fam_key in enumerate(fams):
        ax = axes[idx // ncol][idx % ncol]
        sub = agg[agg["family"] == fam_key]
        for m in models:
            ms = sub[sub["run"] == m].sort_values("step")
            if ms.empty:
                continue
            steps, scores = _with_origin(ms, m, origin, fam_key)
            if len(steps) == 1 or pd.isna(steps).any():
                ax.scatter(steps, scores, color=colour_map[m], s=14, zorder=3)
            else:
                ax.plot(
                    steps,
                    scores,
                    marker="o",
                    markersize=3,
                    color=colour_map[m],
                    zorder=3,
                )
        ax.set_title(family_display_name(fam_key), fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
    for j in range(len(fams), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("All benchmarks (avg across languages)", fontsize=12)
    handles = [
        plt.Line2D([0], [0], color=colour_map[m], marker="o", label=_legend_label(m))
        for m in models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(models), 6),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = out_dir / "overview_grid.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _summary_frame(agg: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """Per (run, step) summary score across families.

    ``method`` selects the aggregation: 'zscore' normalizes each family to a
    z-score first (needed when metrics differ: acc vs chrf++ vs bleu) and
    averages the z-scores; 'naive' plain-averages the raw per-family scores,
    mixing metrics as-is.
    Returns a DataFrame with columns [run, step, score]."""
    if agg.empty:
        return pd.DataFrame(columns=["run", "step", "score"])
    if method == "naive":
        return agg.groupby(["run", "step"], dropna=False)["score"].mean().reset_index()

    def _z(group: pd.DataFrame) -> pd.DataFrame:
        s = group["score"]
        mu = s.mean()
        sd = s.std(ddof=0)
        out = group.copy()
        out["z"] = (s - mu) / sd if sd and not np.isnan(sd) else s - mu
        return out

    zdf = agg.groupby("family", group_keys=False).apply(_z)
    macro = (
        zdf.groupby(["run", "step"], dropna=False)["z"]
        .mean()
        .reset_index()
        .rename(columns={"z": "score"})
    )
    return macro


def _rescale_naive_scores(agg: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Rescale out-of-range families to the 0-1 scale for the naive summary.

    Detects tasks whose performance extends beyond the 0-1 range (metrics
    reported on a 0-100 scale, e.g. bleu or chrf++) and returns a copy of
    ``agg`` with the affected families' scores divided by 100 (a BLEU of 21
    becomes 0.21) so that the naive mean averages comparable scores. Prints a
    [warn] listing the rescaled families. Only the naive macro-average view
    should consume the result; all other plots keep the raw scores."""
    if agg.empty:
        return agg
    oor = df[df["performance"] > 1.0]
    if oor.empty:
        return agg
    fams = sorted(oor["family"].unique())
    metrics = sorted(oor["base_metric"].unique())
    out = agg.copy()
    mask = out["family"].isin(fams)
    out.loc[mask, "score"] = out.loc[mask, "score"] / 100.0
    names = ", ".join(family_display_name(f) for f in fams)
    print(
        f"[warn] naive summary: {oor['task'].nunique()} task(s) with scores "
        f"outside the 0-1 range (metric(s): {', '.join(metrics)}); "
        f"rescaling to 0-1 (score / 100) for the macro-average only: {names}"
    )
    return out


def _plot_macro_average(
    agg: pd.DataFrame,
    models: List[str],
    colour_map: Dict[str, Tuple[float, ...]],
    out_dir: Path,
    dpi: int,
    origin: Optional[Tuple[str, int, object]] = None,
    method: str = "zscore",
    supergroup: str = "multilingual",
) -> Optional[Path]:
    """One line per model summarizing all families: mean of per-family
    z-scores (method='zscore', default) or naive mean of per-family scores
    (method='naive', metrics mixed as-is)."""
    macro = _summary_frame(agg, method)
    if macro.empty:
        return None
    if method == "naive":
        title = f"Macro-average across benchmarks (naive mean of per-family scores): {supergroup}"
        ylabel = "mean score (across families)"
        stem = "macro_average_naive"
    else:
        title = f"Macro-average across benchmarks: {supergroup}"
        ylabel = "mean z-score (across families)"
        stem = "macro_average_zscore"
    print(macro)
    macro.to_csv(f"averaged_scores_{supergroup}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in models:
        sub = macro[macro["run"] == m].sort_values("step")
        if sub.empty:
            continue
        steps, scores = _with_origin(sub, m, origin, None)
        if len(steps) == 1 or pd.isna(steps).any():
            ax.scatter(
                steps, scores, color=colour_map[m], label=_legend_label(m), zorder=3
            )
        else:
            ax.plot(
                steps,
                scores,
                marker="o",
                color=colour_map[m],
                label=_legend_label(m),
                zorder=3,
            )
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if method != "naive":
        ax.axhline(0.0, color="0.7", lw=0.8, ls="--")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _matches_token(fam_key: str, token: str) -> bool:
    """A family matches a filter token if the key/display is equal, or the key
    starts with ``token_`` (so ``flores200`` matches ``flores200_x_to_en`` and
    ``flores200_en_to_x``, and ``polymath`` matches all four tiers)."""
    if not token:
        return False
    disp = family_display_name(fam_key)
    return (
        fam_key == token
        or fam_key.startswith(token + "_")
        or disp == token
        or disp.startswith(token)
    )


def _filter_families(fams: List[str], include, exclude) -> List[str]:
    out = fams
    if include:
        tokens = [f.strip() for f in include if f.strip()]
        out = [f for f in out if any(_matches_token(f, t) for t in tokens)]
    if exclude:
        tokens = [f.strip() for f in exclude if f.strip()]
        out = [f for f in out if not any(_matches_token(f, t) for t in tokens)]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="PATH",
        help="results.csv file or directory (repeatable; a dir is "
        "searched recursively for results.csv).",
    )
    ap.add_argument(
        "--output_dir",
        default="plots",
        metavar="DIR",
        help="Where to write plot PNGs (default: plots/).",
    )
    ap.add_argument(
        "--by_language",
        action="store_true",
        help="Also produce per-language small-multiple plots.",
    )
    ap.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help="Restrict to a single language code (e.g. deu_Latn, de). "
        "Produces one averaged plot per family for that language.",
    )
    ap.add_argument(
        "--families",
        action="append",
        default=[],
        metavar="LIST",
        help="Comma-separated family keys/display names to include "
        "(repeatable). Default: all.",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="LIST",
        help="Comma-separated family keys/display names to exclude " "(repeatable).",
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pdf", action="store_true", help="Also save PDF versions.")
    ap.add_argument(
        "--no_per_family",
        action="store_true",
        help="Skip individual per-family PNGs (use with the " "consolidated views).",
    )
    ap.add_argument(
        "--origin_checkpoint",
        default=None,
        metavar="PATH",
        help="Path to an evaluated checkpoint whose per-benchmark "
        "scores become the first (leftmost) point of every "
        "other model's lines, so all models visually "
        "continue from it. The point's training step is "
        "parsed from the checkpoint path, defaulting to "
        f"{DEFAULT_ORIGIN_STEP} if not parseable.",
    )
    ap.add_argument(
        "--summary",
        choices=("zscore", "naive"),
        default="zscore",
        help="Aggregation for the consolidated macro-average plot: "
        "'zscore' (default) normalizes each family to a z-score and "
        "averages those; 'naive' plain-averages the per-family scores, "
        "first rescaling families whose task scores extend beyond 0-1 "
        "(e.g. BLEU on a 0-100 scale) to the 0-1 range.",
    )
    ap.add_argument(
        "--supergroup",
        choices=("oellm-multilingual-eu", "dclm-core-22", "reasoning"),
        default="oellm-multilingual-eu",
        help="Task supergroup. The name is put in the title of the summary plot.",
    )
    args = ap.parse_args(argv)

    # Expand comma lists for --families / --exclude
    inc = [x for item in args.families for x in item.split(",") if x.strip()]
    exc = [x for item in args.exclude for x in item.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)
    df = annotate(df)

    models = sorted(df["run"].unique())
    if not models:
        raise SystemExit("No models found after parsing model_name.")
    colour_map = _model_colour_map(models)
    print(f"[info] models: {models}")

    # --- origin checkpoint (common starting point) ------------------------- #
    origin_run: Optional[str] = None
    origin_step: Optional[int] = None
    o_fam_scores: Dict[str, float] = {}
    o_lang_scores: Dict[Tuple[str, Optional[str]], float] = {}
    if args.origin_checkpoint:
        resolved = _resolve_origin_model(df, args.origin_checkpoint)
        origin_run, parsed_step = parse_model(resolved)
        if parsed_step is None:
            parsed_step = DEFAULT_ORIGIN_STEP
            print(
                f"[warn] --origin_checkpoint: no iter_ token in {resolved}; "
                f"defaulting its step to {DEFAULT_ORIGIN_STEP}"
            )
        origin_step = parsed_step
        o_rows = df[df["model_name"] == resolved]
        o_fam_scores = o_rows.groupby("family")["performance"].mean().to_dict()
        o_lang_scores = (
            o_rows.groupby(["family", "lang"])["performance"].mean().to_dict()
        )
        print(
            f"[info] origin checkpoint: {resolved} "
            f"(run={origin_run}, step={origin_step})"
        )

    # --- per-family averaged plots --------------------------------------- #
    agg = aggregate_averaged(df)
    all_fams = sorted(agg["family"].unique())
    fams = _filter_families(all_fams, inc, exc)
    print(f"[info] families to plot: {len(fams)} of {len(all_fams)}")

    saved: List[Path] = []
    if not args.no_per_family:
        for fam_key in fams:
            sub = agg[agg["family"] == fam_key]
            origin = (
                (origin_run, origin_step, o_fam_scores.get)
                if origin_run is not None
                else None
            )
            p = _plot_one_family(
                fam_key,
                sub,
                models,
                colour_map,
                out_dir,
                args.dpi,
                args.pdf,
                origin=origin,
            )
            if p:
                saved.append(p)
            # optional single-language filter view
            if args.language:
                lang_sub = df[(df["family"] == fam_key) & (df["lang"] == args.language)]
                if not lang_sub.empty:
                    lang_agg = (
                        lang_sub.groupby(
                            ["family", "run", "step", "base_metric"], dropna=False
                        )["performance"]
                        .mean()
                        .reset_index()
                        .rename(columns={"performance": "score"})
                    )
                    origin_lang = (
                        (
                            origin_run,
                            origin_step,
                            lambda f: o_lang_scores.get((f, args.language)),
                        )
                        if origin_run is not None
                        else None
                    )
                    p2 = _plot_one_family(
                        fam_key,
                        lang_agg,
                        models,
                        colour_map,
                        out_dir,
                        args.dpi,
                        args.pdf,
                        suffix=f"__{args.language}",
                        origin=origin_lang,
                    )
                    if p2:
                        saved.append(p2)

    # --- per-language small multiples ------------------------------------ #
    if args.by_language:
        agg_lang = aggregate_per_language(df)
        for fam_key in fams:
            sub = agg_lang[agg_lang["family"] == fam_key]
            origin_by_lang = (
                (
                    origin_run,
                    origin_step,
                    lambda l, _f=fam_key: o_lang_scores.get((_f, l)),
                )
                if origin_run is not None
                else None
            )
            p = _plot_by_language(
                fam_key,
                sub,
                models,
                colour_map,
                out_dir,
                args.dpi,
                args.pdf,
                origin=origin_by_lang,
            )
            if p:
                saved.append(p)

    # --- consolidated views ---------------------------------------------- #
    origin_grid = (
        (origin_run, origin_step, o_fam_scores.get) if origin_run is not None else None
    )
    grid_path = _plot_overview_grid(
        agg, models, colour_map, out_dir, args.dpi, origin=origin_grid
    )
    if grid_path:
        saved.append(grid_path)
    agg_summary = _rescale_naive_scores(agg, df) if args.summary == "naive" else agg
    origin_summary = None
    if origin_run is not None:
        summary_frame = _summary_frame(agg_summary, args.summary)
        ssub = summary_frame[summary_frame["run"] == origin_run]
        exact = ssub[ssub["step"] == origin_step]
        if not exact.empty:
            ssub = exact
        origin_summary_score = float(ssub.iloc[0]["score"]) if not ssub.empty else None
        origin_summary = (origin_run, origin_step, lambda _k: origin_summary_score)
    summary_path = _plot_macro_average(
        agg_summary,
        models,
        colour_map,
        out_dir,
        args.dpi,
        origin=origin_summary,
        method=args.summary,
        supergroup=args.supergroup,
    )
    if summary_path:
        saved.append(summary_path)

    print(f"[info] wrote {len(saved)} plot(s) to {out_dir}")
    for p in saved:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
