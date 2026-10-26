#!/usr/bin/env python

# -*- coding: utf-8; -*-

#
# compute (equitable, multilingual) data mix ratios according to various constraints
#
# ./multilingual4.py --horizon 2e12 --scale 0.2 multilingual1s.txt > multilingual4.txt
#
import argparse;
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
  parser.add_argument("--horizon", type = lambda _: int(float(_)), default = 2e12);
  parser.add_argument("--scale", type = float, default = 0.2);
  parser.add_argument("--quiet", action = "store_true", default = False);
  parser.add_argument("inputs", nargs = "+");
  arguments = parser.parse_args();

  horizon = round(arguments.horizon * arguments.scale);
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
    
  print(mix)
  #
  # sort lexicographically on dataset (and part) names
  #
  for _ in sorted(mix.values(), key = itemgetter("dataset", "part")):
    print("{:,.6f} {}/megatron-lm/{}"
          "".format(_["ratio"], _["dataset"], _["part"]));

if __name__ == "__main__":
  main();
