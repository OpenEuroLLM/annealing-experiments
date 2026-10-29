# Annealing mixtures experiments

`plot_eval_progress.py` takes an an input the directories with the output of `oellm-eval` and produces comparative
plots. Run as something like:

```
singularity exec \
--bind /pfs/lustrep4/scratch/project_465002891:/scratch/project_465002891 \
/scratch/project_465002530/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif \
python plot_eval_progress.py \
--input DIR_WITH_SCORES \
--input /scratch/project_465002891/prelude-mid/evals/before-annealing/results.csv \
--output_dir OUTPUT \
--origin_checkpoint /scratch/project_465002891/prelude-mid/hf_models/baby_9b_dense_before-annealing/checkpoints/iter_0953312/ \
--by_languages \
-- supergroup REASONING
```

```bash
for i in 20 50 100 300; do
  for j in english multilingual reasoning; do
    singularity exec --bind /pfs/lustrep4/scratch/project_465002891:/scratch/project_465002891,/scratch/project_465002530 \
      /scratch/project_465002530/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif \
      python plot_eval_progress.py \
        --input prelude/evals/dclm-core-22/0bt \
	--input prelude/evals/oellm-multilingual-eu/0bt \
	--input prelude/evals/reasoning/0bt
        --input prelude/evals/dclm-core-22/${i}bt \
	--input prelude/evals/oellm-multilingual-eu/${i}bt \
	--input prelude/evals/reasoning/${i}bt \
	--origin_checkpoint /scratch/project_465002891/prelude-mid/hf_models/baby_9b_dense_before-annealing/checkpoints/iter_0953312 \
	--supergroups_file prelude/evals/groups1.csv --supergroup ${j} \
	--output_dir prelude/groups1/${i}bt/${j} 2>&1 \
    | tee ${i}bt.${j}.log;
  done;
done
```
