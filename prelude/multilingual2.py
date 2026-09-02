#!/usr/bin/env python

# -*- coding: utf-8; -*-

#
# compute data mix ratios according to various constraints
#
# ./multilingual2.py --horizon 2e12 --scale 0.2 \
#   --languages languages.txt --repeat 2 \
#   --limit web:0.5 --limit mt:0.3 --limit pdf:0.15 --limit parallel:0.05 \
#   --fill web --fill pdf --step 1e6 flag.csv > multilingual2.txt
#
import argparse;
import io;
from itertools import chain;
import json;
import math;
from operator import itemgetter;
import os;
import re;
import sys;
import time;

def main():

  start = time.time();

  parser = argparse.ArgumentParser(description = "Compute Equitable Multilingual Mix");
  parser.add_argument("--languages", type = str, default = "languages.txt");
  parser.add_argument("--horizon", type = lambda _: int(float(_)), default = 2e12);
  parser.add_argument("--scale", type = float, default = 0.2);
  parser.add_argument("--limit", action = "append", default = list());
  parser.add_argument("--fill", action = "append", default = list());
  parser.add_argument("--step", type = lambda _: int(float(_)), default = int(1e6));
  parser.add_argument("--repeat", type = int, default = 2);
  parser.add_argument("--debug", action = "store_true", default = False);
  parser.add_argument("inputs", nargs = "+");
  arguments = parser.parse_args();

  horizon = round(arguments.horizon * arguments.scale);
  limits = [];
  for _ in arguments.limit:
    fields = _.split(":");
    limits.append((fields[0],
                   min(float(fields[1]), 1.0) if len(fields) > 1 else 1 / len(arguments.limits),
                   max(int(fields[2]), 1) if len(fields) > 2 else arguments.repeat));
  languages, n = dict(), 0;
  with open(arguments.languages) as stream:
    for line in stream:
      if line.startswith("#"): continue;
      fields = line.strip().split(":");
      codes = fields[2].strip().split(" ");
      key = fields[0].strip();
      if key in languages:
        print("multilingual2.py: duplicate key {} in --language map {}; exit."
              "".format(key, arguments.language),
              file = sys.stderr, flush = True);
        sys.exit(1);
      languages[key] = {"name": fields[1].strip(),
                        "codes": {_.strip() for _ in codes},
                        "budget": 0, "n": 0};
      n += len(codes);
  print("# multilingual2.py: {} codes for {} languages; horizon: {:,d} @ {:,.2f}."
        "".format(n, len(languages), horizon, arguments.scale),
        flush = True);
  
  pool, n = dict(), 0;
  total = {"all": 0, "tokens": 0, "wiki": 0, "parallel": 0, "pdf": 0, "web": 0, "mt": 0};
  for file in arguments.inputs:
    with open(file) as stream:
      next(stream);
      for line in stream:
        collection = None;
        for _ in {"baby", "flag"}:
          if _ in file: collection = _;
        fields = line.strip().split(",");
        part = fields[1];
        if len(fields) > 12 and fields[12]: tokens = int(fields[12]);
        else: tokens = int(fields[7] if fields[7] else fields[4]);
        total["all"] += tokens;
        for key, _ in languages.items():
          for code in _["codes"]:
            if code in part:
              set = fields[0];
              _ = {"collection": collection, "set": set, "part": part,
                   "tokens": tokens, "ratio": 0, "n": 0};
              if "nemotron-cc-opus" in set or "nemotron-cc-tower+" in set: _["type"] = "mt";
              if "dochplt" in set or "fineopus" in set: _["type"] = "parallel";
              if "finepdfs" in set: _["type"] = "pdf";
              if "finewiki" in set: _["type"] = "wiki";
              if set in {"hplt-3.0", "hplt-4.0"}: _["type"] = "web";
              if key not in pool: pool[key] = [_];
              else: pool[key].append(_);
        n += 1;

  for key,value in pool.items():
    #
    # prefer finepdfs-edu (among .pdf.), the clean partition of HPLT 4.0 (among .web.),
    # and dochplt (among parallel)
    #
    pool[key].sort(key = lambda _: "-edu" in _["set"] or "clean/" in _["part"] or "dochplt" in _["set"],
                   reverse = True);
    for _ in value:
      total["tokens"] += _["tokens"];
      if "type" in _: total[_["type"]] += _["tokens"];
  print("# multilingual2.py: {} dataset parts for {} languages in {} pool(s);"
        "".format(n, len(pool), len(arguments.inputs)),
        flush = True);
  _ = total["tokens"];
  print("# multilingual2.py: {:,d} tokens (of {:,d});"
        "".format(_, total["all"]),
        flush = True);
  print("# multilingual2.py: {:,.1f}% wiki, {:,.1f}% parallel, "
        "{:,.1f}% pdf, {:,.1f}% web, {:,.1f}% mt."
        "".format(total["wiki"] / _ * 100, total["parallel"] / _ * 100,
                  total["pdf"] / _ * 100, total["web"] / _ * 100, total["mt"] / _ * 100),
        flush = True);
  print("# multilingual2.py: limits:", end = "", flush = True);
  for i, (type, ratio, repeat) in enumerate(limits):
    if i == 0: print(f" {type}:{ratio}:{repeat}", end = "", flush = True);
    else: print(f", {type}:{ratio}:{repeat}", end = "", flush = True);
  print(".", flush = True);
    
  allocation = 0;
  #
  # a first round of equitable allocations, within .limits. and .repeat. constraints
  #
  for language, value in languages.items():
    #
    # per-language budgets, equitable allocated as a share of the overall horizon
    #
    languages[language]["budget"] = budget = round(horizon / len(pool));
    for type, ratio, repeat in limits:
      #
      # per-type limit for the current language, up to a set ratio and repeat count
      #
      limit = min(round(horizon / len(pool) * ratio), horizon - allocation);
      for _ in pool[language]:
        if "type" not in _ or _["type"] != type: continue;
        n = min(_["tokens"] * repeat, budget, limit);
        _["n"] += n;
        if n == _["tokens"] * repeat: _["empty"] = True;
        allocation += n;
        languages[language]["n"] += n;
        if arguments.debug:
          print("{}/{}: {:,d} tokens of {:,d} ({:,.2f}%) {{{:,d} of {:,d}}}; {}: {:,d} [{:,d} of {:,d}]"
                "".format(_["set"], _["part"], n, _["tokens"], n / _["tokens"] * 100,
                          limit, budget, language, languages[language]["n"], allocation, horizon),
                file = sys.stderr, flush = True);
        budget -= n;
        limit -= n;
        if limit == 0: break;
      if budget == 0: break;
  #
  # now, fill up round-robin in 1m increments, where data remains available
  #
  limits.clear();
  for _ in arguments.fill:
    fields = _.split(":");
    limits.append((fields[0],
                   max(int(fields[1]), 1) if len(fields) > 1 else arguments.repeat));
  print("# multilingual2.py: fills:", end = "", flush = True);
  for i, (type, repeat) in enumerate(limits):
    if i == 0: print(f" {type}:{repeat}", end = "", flush = True);
    else: print(f", {type}:{repeat}", end = "", flush = True);
  print(".", flush = True);
    
  if len(limits) == 0: empty = True;
  else: empty = False;
  while not empty and horizon > allocation:
    empty = True;
    for type, repeat in limits:
      for language in languages.keys():
        for _ in pool[language]:
          if "type" not in _ or _["type"] != type or "empty" in _: continue;
          #
          # shun lower-quality sources for types that (in principle) offer a choice
          #
          if type == "web" and "noisy/" in _["part"]: continue;
          if type == "pdf" and "-edu" not in _["set"]: continue;
          n = min(_["tokens"] * repeat - _["n"], arguments.step, horizon - allocation);
          _["n"] += n;
          if n == _["tokens"] * repeat: _["empty"] = True;
          allocation += n;
          languages[language]["n"] += n;
          empty = False;
          if arguments.debug:
            print("{}/{}: {:,d} tokens of {:,d} ({:,.2f}%); {}: {:,d} [{:,d} of {:,d}]"
                  "".format(_["set"], _["part"], n, _["tokens"], n / _["tokens"] * 100,
                            language, languages[language]["n"], allocation, horizon),
                  file = sys.stderr, flush = True);

  if arguments.debug:
    for key, value in pool.items():
      for _ in value:
        ratio = _["n"] / horizon * arguments.scale;
        if ratio:
          print("{}/{}: {:,d} tokens ({:,.2f}%) [{:,d} of {:,d}]; {:,.6f}."
                "".format(_["set"], _["part"],
                          _["n"], _["n"] / _["tokens"] * 100,
                          languages[key]["n"], languages[key]["budget"],
                          _["ratio"]));
  else:
    for _ in sorted(chain(*pool.values()), key = itemgetter("collection", "set", "part")):
      ratio = _["n"] / horizon * arguments.scale;
      if ratio:
        print("{:,.6f} {}/{}/megatron-lm/{}"
              "".format(ratio, _["collection"], _["set"], _["part"]));
  print("# multilingual2.py: allocated {:,d} tokens of {:,d} ({:,.2f}%)."
        "".format(allocation, horizon, allocation / horizon * 100),
        flush = True);

if __name__ == "__main__":
  main();
