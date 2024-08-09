# Anonymous submission

Paper Title: **Bridging Continual Learning of Motion and Self-Supervised Representations**


CODE REPOSITORY CONTENTS
------------------------
The datasets needed for the experimental campaign can be downloaded from the benchmark proposed by Tiezzi et al.,
which can be found at [this link](https://github.com/sailab-code/cl_stochastic_coherence)  in the `data` folder:
The content of the folder should be downloaded and inserted into the `data` folder of this repository. 
Concerning real-world videos (Horse and Rat), preprocessed files are available at [this link](https://drive.google.com/file/d/1sB3idnhbaBdOggWccVqyIDBbr3lJMGuI/view?usp=sharing).

    cmoss :                 folder containing the source code of our CMOSS model
    data :                  folder containing the rendered streams
    lve :                   source folder for handling the video processing and model creation and training  
    runner.py :             experiments runner for 3D-generated streams
    runner_sparse.py :      experiments runner for real-world videos
    best_cmoss_runs.txt :   command lines (and parameters) to reproduce the main results

REPRODUCE PAPER RESULTS
-----------------------

We tested our code with `PyTorch 1.10`. Please install the other required dependencies by running:

```
pip install -r requirements.txt
```

In the `best_cmoss_runs.txt` file there are the command lines (hence, the experiments parameters) required to reproduce the experiments of the main results (Table 1).

RUNNING EXPERIMENTS
-------------------
We provide a `runner.py` script to easily test the proposed model. 
The PyTorch device is chosen through the `--device` argument (`cpu`, `cuda:0`,
`cuda:1`, etc.).

    usage: runner.py [-h] [--laps_unsup LAPS_UNSUP] [--laps_sup LAPS_SUP] [--laps_metrics LAPS_METRICS] 
                 [--step_size_features STEP_SIZE_FEATURES] [--step_size_displacements STEP_SIZE_DISPLACEMENTS] 
                 [--force_gray {yes,no}]  [--dataset DATASET] 
                 [--seed SEED] [--crops CROPS] [--flips FLIPS] [--jitters JITTERS]
                 [--lambda_c_lower LAMBDA_C_LOWER] [--lambda_c_upper LAMBDA_C_UPPER] [--lambda_r LAMBDA_R]  [--lambda_s LAMBDA_S] 
                 [--feature_planes FEATURE_PLANES]  [--device DEVICE] [--features_block FEATURES_BLOCK] [--displacement_block DISPLACEMENT_BLOCK]
                 [--wandb WANDB] [--eval_forgetting EVAL_FORGETTING] 
                 [--eval_forgetting_fractions EVAL_FORGETTING_FRACTIONS] 
                 [--num_pairs NUM_PAIRS] [--lambda_sim LAMBDA_SIM]
                 [--similarity_threshold SIMILARITY_THRESHOLD] [--dissimilarity_threshold DISSIMILARITY_THRESHOLD] [--moving_threshold MOVING_THRESHOLD] 
                 [--sampling_type {plain,motion,features,motion_features}] [--kept_pairs_perc KEPT_PAIRS_PERC] 
                 [--simdis_loss_tau SIMDIS_LOSS_TAU] [--teacher {yes,no}] [--teacher_ema_weight TEACHER_EMA_WEIGHT] 
                 [--simdis_neg_avg SIMDIS_NEG_AVG]

Argument description/mapping with respect to the paper notation:

        --laps_unsup :  number of unsupervised laps where the coherence losses are minimized
        --laps_sup : number of laps on which the supervised templates are provided
        --laps_metrics : number of laps on which the metrics are computed (here the model weight are frozen, no learning is happening)
        --step_size_features : learning rate for the features branch
        --step_size_displacements : learning rate for the features branch
        --crops :  number of random crop augmentations
        --jitters :  number of jitters augmentations
        --flips :  number of horizontal flips  augmentations
        --lambda_c_lower : \lambda_m in the paper
        --lambda_c_upper : \beta_bowtie in the paper
        --lambda_r : \beta_mr in the paper
        --lambda_sim : \beta_f in the paper
        --lambda_s : \beta_m in the paper 
        --simdis_loss_tau : \tau in the paper  
        --num_pairs: maximum number of points for the sampling procedure (corresponds to \ell * \ell from the paper)  
        --max_supervisions : number of supervisions per object
        --force_gray : if "yes", it corresponds to the "BW" of the paper. "no" requires an RGB stream
        --feature_planes : output dimension (number of features) of the features neural branch
        --displacement_block : architecture for the motion prediction branch [default: resunetblocknolastskip]
        --features_block : architecture for the feature branch [default: resunetblock_bias]
        --eval_forgetting : activate the procedure for investigating forgetting behaviour (ablation study on online learning)
        --eval_forgetting_fractions : number of point/additional laps in which forgetting is evaluated 
        --similarity_threshold : \tau_p in the paper
        --dissimilarity_threshold : \tau_n in the paper
        --moving_threshold : \tau_m in the paper
        --dataset  : specify the input stream
        --sampling_type : the nature of the sampling in {plain,motion,features,motion_features} (see ablation studies)
        --kept_pairs_perc : \aleph in the paper
        --teacher : activate the EMA net (default yes)
        --teacher_ema_weight  : \xi in the paper
        --seed : specify the seed
        --simdis_neg_avg : average the term for negative samples in the contrastive term (ablation study)



COLLECTING THE FINAL METRICS 
---------------------------

The final metrics are dumped in the `model_folder/final_metrics.json` file.

This file contains a dictionary with multiple metrics. The metrics that are reported in Table 1 of the paper are under
the key `f1_window_whole_global`. The F1 measured on attention trajectory can be found under the key `f1_window_foa_global`.

COMPETITORS
-----------
For the competitors performances we re-runned [the code by Tiezzi et al](https://github.com/sailab-code/cl_stochastic_coherence).
We slightly updated the code to support the [MOCO](https://github.com/facebookresearch/moco) and 
[PixPro](https://github.com/zdaxie/PixPro) models (pre-trained weights can be found in the respective repositories).  


_NOTICE: PyTorch does not guarantee Reproducibility is not guaranteed by PyTorch across different releases, platforms, hardware. Moreover,
determinism cannot be enforced due to use of PyTorch operations for which deterministic implementations do not exist
(e.g. bilinear upsampling)._



