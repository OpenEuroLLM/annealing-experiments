# Notes from Annealing Experiments for the Prelude (`Baby`) Cycle

##

```
egrep -v 'dclm-1.0|finemath-0.0.0|finepdfs-1.0.0/megatron-lm/eng_Latn|finepdfs-edu-1.0.0/megatron-lm/eng_Latn|megamath-0.0.0|nemotron-cc-1.0|olmo-mix-1124|starcoder-0.0.0' ../../training/collection/baby/datamix.txt > multilingual0.txt

cat ../../training/collection/baby/datamix.txt \
| while read line; do if grep "$line" multilingual0.txt > /dev/null; then :; else echo $line; fi; done > dominant0.txt
```

```
awk -F, '{ if ($1 && $6) printf("%.6f flag/%s/megatron-lm/%s/shard_0000_text_document\n", $6 / 2e12, $2, $3); }' multilingual1.csv > multilingual1.txt
```
