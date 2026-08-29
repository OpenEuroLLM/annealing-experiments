#!/usr/bin/env python

# -*- coding: utf-8; -*-

import argparse;
import io;
import json;
import math;
import os;
import re;
import sys;
import time;

def main():

  start = time.time();

  parser = argparse.ArgumentParser(description = "Compute Equitable Multilingual Mix");
  parser.add_argument("--languages", type = str, default = "languages.txt");
  parser.add_argument("--horizon", type = lambda _: int(float(_)), default = 2e12);
  parser.add_argument("--mt", type = float, default = 0.3);
  parser.add_argument("--parallel", type = float, default = 0.05);
  parser.add_argument("--pdf", type = float, default = 0.20);
  parser.add_argument("--web", type = float, default = 0.5);
  parser.add_argument("--wiki", type = float, default = 0.1);
  parser.add_argument("--repeat", type = float, default = 4);
  parser.add_argument("inputs", nargs = "+");
  arguments = parser.parse_args();

  languages = dict();
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
                        "codes": {_.strip() for _ in codes}};
      print(languages);

    pool = dict();
    for file in arguments.inputs:
      with open(file) as stream:
        next(stream);
        for line in stream:
          fields = line.strip().split(",");
          for key, _ in languages.items():
            for code in _["codes"]:
              part = fields[1];
              if code in part:
                set = fields[0];
                _ = {"set": set, "part": part,
                     "tokens": fields[7] if fields[7] else fields[4]};
                if set in {"hplt-3.0", "hplt-4.0"}: _["type"] = "web";
                if "finepdfs" in set: _["type"] = "pdf";
                if "nemotron-cc-opus" in set or "nemotron-cc-tower+" in set: _["type"] = "mt";
                if "dochplt" in set or "fineopus" in set: _["type"] = "parallel";
                if key not in pool: pool[key] = [_];
                else: pool[key].append(_);
  print(pool);  

if __name__ == "__main__":
  main();
