#!/usr/bin/env python

# -*- coding: utf-8; -*-

#
#
import argparse;
import glob;
import io;
import json;
import os;
import re;
import sys;
import time;

def main():

  start = time.time();

  parser = argparse.ArgumentParser(description = "Distribute Data Mix Ratio");
  parser.add_argument("--root", type = str, default = "/scratch/project_465002530/training/collection");
  parser.add_argument("inputs", nargs = "+");
  arguments = parser.parse_args();

  pattern = re.compile(r"(0\.[0-9]+)[ \t](.+)$")
  suffix = re.compile(r"/(:?shard)?[0-9_]+_text_document$");
  for file in arguments.inputs:
    with open(file, encoding = "utf-8") as stream:
      for i, line in enumerate(stream):
        line = line.strip();
        if line.startswith("#"): continue;
        _ = pattern.match(line)
        if _ is None:
          if not arguments.debug:
            print("share.py(): ignoring line #{}: {}."
                  "".format(i, line),
                  file = sys.stderr, flush = True);
          continue;
        ratio, path = _.groups();
        ratio = float(ratio);
        if ratio == 0:
          if not arguments.quiet:
            print("share.py(): [#{}] invalid line: {}."
                  "".format(i, line),
                  file = sys.stderr, flush = True);
          continue;
        _ = suffix.search(path);
        if _ is not None: path = path[:_.start()];
        if not os.path.isdir(os.path.join(arguments.root, path)):
          print("share.py(): [#{}] invalid path: {}; exit."
                "".format(i, path),
                file = sys.stderr, flush = True);
          sys.exit(1);
        shards = dict();
        for shard in sorted(glob.glob(os.path.join(arguments.root, path, "*.info.json"))):
          with open(shard) as _:
            _ = json.load(_);
            shards[shard] = _["idx"]["total_tokens"];
        n = sum(shards.values());
        for shard, i in shards.items():
          tail = os.path.split(shard)[1][:-len(".info.json")];
          print("{:,.6f} {}/{}_text_document"
                "".format(ratio / n * i, path, tail));

if __name__ == "__main__":
  main();
