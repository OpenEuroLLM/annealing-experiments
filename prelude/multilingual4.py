#!/usr/bin/env python

# -*- coding: utf-8; -*-

#
# compute (equitable, multilingual) data mix ratios according to various constraints
#
# ./multilingual4.py --horizon 3e12 multilingual1.txt > multilingual4.txt
#
import argparse;
from functools import cmp_to_key;
import glob;
import io;
import json;
import math;
from operator import itemgetter;
import os;
import re;
import sys;
import time;

def main():

  start = time.time();

  parser = argparse.ArgumentParser(description = "Optimize BSC-edu Scores for HPLT 4.0");
  parser.add_argument("--root", type=str, default="/scratch/project_465002530/training/collection");
  parser.add_argument("--horizon", type = lambda _: int(float(_)), default = 3e12);
  parser.add_argument("--quiet", action = "store_true", default = False);
  parser.add_argument("inputs", nargs = "+");
  arguments = parser.parse_args();

  pattern = re.compile(r"(0\.[0-9]+)[ \t](.+)$")
  suffix = re.compile(r"/(:?shard)?[0-9_]+_text_document$");
  mix = dict();
  with open(arguments.inputs[0], encoding = "utf-8") as stream:
    for i, line in enumerate(stream):
      line = line.strip();
      if line.startswith("#"): continue;
      _ = pattern.match(line)
      if _ is None:
        if not arguments.quiet:
          print("multilingual4.py(): ignoring invalid line #{}: {}."
                "".format(i, line),
                file = sys.stderr, flush = True);
        continue;
      ratio, path = _.groups();
      ratio = float(ratio);
      if ratio == 0:
        if not arguments.quiet:
          print("multilingual4.py(): ignoring null line #{}: {}."
                "".format(i, line),
                file = sys.stderr, flush = True);
        continue;
      _ = suffix.search(path);
      if _ is not None: path = path[:_.start()];
      if path in mix:
        mix[path]["ratio"] += ratio;
      else:
        dataset = re.sub(r"/megatron-lm.*", "", path);
        part = re.sub(r".+/megatron-lm/?", "", path);
        mix[path] = {"dataset": dataset, "part": part, "ratio": ratio};

  #
  # helper function to sort shard names by BSC-edu score levels
  #
  def cmp(left, right):
    left = left[-len("??.info.json"):];
    right = right[-len("??.info.json"):];
    if left < right: return -1;
    elif left > right: return 1;
    else: return 0;
  #
  # sort lexicographically on dataset (and part) names
  #
  for entry in sorted(mix.values(), key = itemgetter("dataset", "part")):
    dataset = entry["dataset"]; part = entry["part"];
    ratio = entry["ratio"];
    #
    # it turns out, clean/srp_Latn and noisy/srp_Latn are missing
    #
    if "hplt-4.0" in dataset and not "srp_Latn" in part:
      dataset += "-bsc-edu";
      budget = round(ratio * arguments.horizon);
      shards = dict();
      _ = glob.glob(os.path.join(arguments.root,
                                 dataset, "megatron-lm/openeurollm-256k",
                                 part, "*.info.json"));
      for shard in sorted(_, key = cmp_to_key(cmp), reverse = True):
        with open(shard) as _:
          n = json.load(_)["idx"]["total_tokens"];
          tail = os.path.split(shard)[1][:-len(".info.json")];
          _ = min(ratio, entry["ratio"] * n / budget);
          shards["{}/megatron-lm/openeurollm-256k/{}/{}_text_document"
                 "".format(dataset, part, tail)] = _;
          ratio -= _;
#          print(dataset, part, tail, entry["ratio"], budget, n, _, ratio,
#                file = sys.stderr);
          if ratio <= 0: break;
      if ratio > 0: n = entry["ratio"] / (entry["ratio"] - ratio);
      else: n = 1;
      for path, ratio in shards.items():
        print("{:,.10f} {}_text_document".format(ratio * n, path));
    else:
      shards = dict();
      for shard in sorted(glob.glob(os.path.join(arguments.root,
                                                 dataset, "megatron-lm", part,
                                                 "*.info.json"))):
        with open(shard) as _:
          shards[shard] = json.load(_)["idx"]["total_tokens"];
      n = sum(shards.values());
      for shard, i in shards.items():
        tail = os.path.split(shard)[1][:-len(".info.json")];
        print("{:,.10f} {}/megatron-lm/{}/{}_text_document"
              "".format(ratio / n * i, dataset, part, tail));

if __name__ == "__main__":
  main();
