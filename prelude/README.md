# Notes from Annealing Experiments for the Prelude (`Baby`) Cycle

## “Dominant” vs. “Multilingual” Components

```
egrep -v 'dclm-1.0|finemath-0.0.0|finepdfs-1.0.0/megatron-lm/eng_Latn|finepdfs-edu-1.0.0/megatron-lm/eng_Latn|megamath-0.0.0|nemotron-cc-1.0|olmo-mix-1124|starcoder-0.0.0' ../../training/collection/baby/datamix.txt > multilingual0.txt

cat ../../training/collection/baby/datamix.txt \
| while read line; do if grep "$line" multilingual0.txt > /dev/null; then :; else echo $line; fi; done > dominant0.txt
```

## Higher-Quality Multilingual Data

```
awk -F, '{ if ($1 && $6) printf("%.6f flag/%s/megatron-lm/%s/shard_0000_text_document\n", $6 / 2e12, $2, $3); }' multilingual1.csv > multilingual1.txt
```

## Multilingual Equity

```./multilingual2.py --horizon 2e12 --scale 0.18 --limit web:0.5 --limit mt:0.3 --limit pdf:0.15 --limit parallel:0.05 --fill web --fill pdf flag.csv > multilingual2.txt
```

## Less English: 40% Multilingual

```
awk '/^[0-9.]+ / {printf("%.6f %s\n", $1 * 0.75, $2); }' dominant0.txt > multilingual3.txt
awk '/^[0-9.]+ / {printf("%.6f %s\n", $1 * 2.27, $2); }' multilingual1.txt >> multilingual3.txt
```

## Dominant Mixes

```
for i in jenia*.txt; do egrep 'agenttrove-0.0|common-pile-stackv2-0.1|common-pile-stackv2-edu-0.1|dclm-1.0|dolmino-mix-100b-1125|finemath-0.0.0|finepdfs-1.0.0/megatron-lm/eng_Latn|finepdfs-edu-1.0.0/megatron-lm/eng_Latn|finephrase-0.0.0|hplt-4.0/megatron-lm/clean/eng_Latn|megamath-0.0.0|mixture-vitae-1.0-nuggets|nemotron-cc-1.0|nemotron-pretraining-specialized-1.0|nemotron-pretraining-specialized-1.1|openwebmath-0.0.0|starcoder-0.0.0|swallow-code-2.0|swallow-math-2.0' $i | sed 's,/scratch/project_465002530/training/collection/,,g' | sort -k 2,3 > dominant${i##jenia}; done
```