# Annealing mixtures experiments

`plot_eval_progress.py` takes an an input the directories with the output of `oellm-eval` and produces comparative plots. Run as:

```
singularity exec --bind /pfs/lustrep4/scratch/project_465002891:/scratch/project_465002891 /scratch/project_465002530/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif python plot_eval_progress.py --input DIR_WITH_SCORES --input /scratch/project_465002891/prelude-mid/evals/before-annealing/results.csv  --output_dir OUTPUT --origin_checkpoint /scratch/project_465002891/prelude-mid/hf_models/baby_9b_dense_before-annealing/checkpoints/iter_0953312/ --by_languages
```
