import argparse
import json
import os
import random
import time
import numpy as np
import torch
import lve
import wandb
from lve.utils import normalize_heatmaps

from packaging import version

start_time = time.time()

def create_progressive_dict(log_dict, progressive_stats):
    progressive_dict = {}
    if 'logged' in log_dict and log_dict['logged'] is not None and len(log_dict['logged']) > 0:
        mapping_dict = {"whole_frame": "whole", "foa_moving": "foam", "foa": "foa"}

        for metric in ["f1", "acc", "u1"]:
            for area in ["whole_frame", "foa", "foa_moving"]:
                for setting in ["window", "running"]:
                    for t in range(len(progressive_stats)):
                        key = f"progressive_{metric}_{setting}_{mapping_dict[area]}_t{t}"
                        if metric in progressive_stats[t][area][setting]:
                            value = progressive_stats[t][area][setting][metric]
                            for i in range(len(value) - 1):
                                progressive_dict[key + "_" + str(i)] = value[i]
                            progressive_dict[key + "_global"] = value[-1]
                            progressive_dict[key + "_good_classes"] = np.mean(value[:-2])
    return progressive_dict

def create_log_dict(args_cmd):
    # logger
    log_dict = {'element': 'stats.metrics', 'log_last_only': True, 'logged': [],
                }

    if args_cmd.save_videos:
        log_dict.update({"pred_img": "prediction_idx", "pred_list": [],
                         "upred_img": "unmasked-prediction_idx", "upred_list": [],
                         "pred_motion": [f"net_motion.{i}" for i in range(args_cmd.n_blocks)],
                         "motion_list": []})
    return log_dict


def run_exp(args_cmd, base_path="./data", pretrained_net=None):
    torch.set_num_threads(3)
    # creating streams
    if args_cmd.dataset == "livingroom":
        supervised_categories = 5
        foa_file = "data/livingroom/foa_log_alpha_c0.1__alpha_of_1.0__alpha_fm_0.0__" + \
                   "max_distance_257__dissipation_0.05__fixation_threshold_speed_25.foa"
    elif args_cmd.dataset == "emptyspace":
        supervised_categories = 5
        foa_file = "data/emptyspace/empty_space_bench_foa_long.foa"
    elif args_cmd.dataset == "solid":
        supervised_categories = 4
        foa_file = "data/solid/foa_new_solid_bench_long.foa"

    fix_motion_v = False
    fix_motion_u = True
    laps = args_cmd.laps_unsup + args_cmd.laps_sup + args_cmd.laps_metrics
    ins_properties = {
        'input_element': os.path.join(base_path, args_cmd.dataset),
        'w': -1, 'h': -1, 'fps': None, 'max_frames': None,
        'laps': laps,
        'force_gray': args_cmd.force_gray == "yes",
        'foa_file': foa_file,
        'motion_disk_type': args_cmd.motion_disk_type,
        'load_sup': not args_cmd.without_supervision,
        'fix_flow_u': fix_motion_u,
        'fix_flow_v': fix_motion_v,
        'shuffle': args_cmd.shuffle == 'yes',
        'seed': args_cmd.seed
    }

    if args_cmd.preload_data:
        print('Preloading data into memory..')
        data = np.load(os.path.join(base_path, args_cmd.dataset, "compressed.npz"))
        ins_properties['input_element'] = {"frames": data["frames"], "motion": data["motion"], "sup": data["sup"],
                                           "fps": data["fps"][0]}
        if args_cmd.motion_disk_type is not None:
            ins_properties['input_element']['motion_additional'] = data['motion_additional']


    ins = lve.InputStream(**ins_properties)


    output_settings = {
        'folder': "output_folder",
        'fps': ins.fps,
        'virtual_save': True,
        'tensorboard': False,
        'save_per_frame_data': True,
        'purge_existing_data': not args_cmd.resume
    }

    #### OPTIONS
    general_options = {
        "device": args_cmd.device,  # "cuda:0",  # cpu, cuda:0, cuda:1, ...
        "seed": args_cmd.seed,  # if smaller than zero, current time is used
        'motion_threshold': -1.0,  # if negative, the whole set of moving pixels are taken
        'sup_batch': 16,
        'sup_persistence': 5,
        'piggyback_frames': supervised_categories * args_cmd.max_supervisions if args_cmd.train == 'yes' else 1,
        "supervision_map": ins.sup_map,
        'backward_optical_flow': False,
        'previous_frame_data_size': args_cmd.previous_frame_data_size,
        # the number of previous frames to process together with the current one
        'previous_frame_offsets': args_cmd.previous_frame_offsets,
        'shuffle': args_cmd.shuffle,
        'batch_size': args_cmd.batch_size,
    }

    sup_policy_options = {
        'type': 'only_moving_objects',
        'min_laps': args_cmd.laps_unsup + 1,
        # first laps which receives supervisions (the one after unsup reps)
        'max_laps': args_cmd.laps_unsup + args_cmd.laps_sup,
        # last repetition which receives supervisions
        'wait_for_frames': 100,  # frames passed before giving a supervision again
        'max_supervisions': args_cmd.max_supervisions  # max supervisions per object
    }


    eq_policy_options = None

    foa_options = {'alpha_c': 0.1,
                   'alpha_of': 1.0,
                   'alpha_fm': 0.0,
                   'alpha_virtual': 0.0,
                   'max_distance': int(0.5 * (ins.w + ins.h)) if int(0.5 * (ins.w + ins.h)) % 2 == 1 else int(
                       0.5 * (ins.w + ins.h)) + 1,
                   'dissipation': 0.1,
                   'fps': ins.fps,
                   'w': ins.w,
                   'h': ins.h,
                   'y': None,
                   'is_online': ins.input_type == lve.InputType.WEB_CAM or ins.input_type == lve.InputType.SAILENV or ins.input_type == lve.InputType.TDW,
                   'fixation_threshold_speed': int(0.1 * 0.5 * (ins.w + ins.h))}

    if args_cmd.interactive_threshold:
        dist_threshold = 0.1
    else:
        if args_cmd.feature_normalization == "yes":
            dist_threshold = [0.000001, 0.0005, 0.0003, 0.0002, 0.0007, 0.001, 0.01, 0.1, 0.25, 0.5, 0.7, 1.0]
        else:
            dist_threshold = [0.1, 2, 10, 18, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600]

    # handling separate step_sizes
    if args_cmd.step_size_displacements is None and args_cmd.step_size_features is None:
        # if both not specified, the learning rate is step size as was before
        args_cmd.step_size_displacements = args_cmd.step_size_features = args_cmd.step_size
    else:
        # both LR must be specified
        assert args_cmd.step_size is None, "Standard step-size must be None!"
        assert args_cmd.step_size_displacements is not None and args_cmd.step_size_features is not None, "Both step sizes for features and displacement must be specified!"

    net_options = {'c': ins.c,
                   'step_size_displacements': args_cmd.step_size_displacements,  # a negative value triggers Adam
                   'step_size_features': args_cmd.step_size_features,  # a negative value triggers Adam
                   'step_size_decay': args_cmd.step_size_decay,
                   'supervised_categories': supervised_categories,
                   "classifier": "NN",  # 'NN', 'neural'
                   'dist_threshold': dist_threshold,
                   'freeze': args_cmd.train == "no",
                   'training_max_laps': laps - args_cmd.laps_metrics,
                   'normalize': args_cmd.input_normalization == "yes",
                   # new stuff
                   'n_blocks': args_cmd.n_blocks,
                   'gradient_type': args_cmd.gradient_type,
                   "arch_mode": args_cmd.arch_mode,
                   "regularization_type": args_cmd.regularization_type,
                   "consistency_type": args_cmd.consistency_type,
                   'vision_block': {
                       'weights_reinit': args_cmd.weights_reinit,
                       'features': {
                           'block_name': args_cmd.features_block,
                           'planes': args_cmd.feature_planes,
                           'stride': 1,
                           'use_initial_for_features': False,
                           'lambda_e': args_cmd.lambda_ef,
                           'normalize': args_cmd.feature_normalization == "yes",
                           'softmax_temp': args_cmd.softmax_temp,
                           'first_block_identity': args_cmd.first_block_identity
                       },
                       'displacements': {
                           'motion_disk_type': args_cmd.motion_disk_type,
                           'block_name': args_cmd.displacement_block,
                           'planes': 2,
                           'stride': 1,
                           'use_coarser_for_displacements': False,
                           'lambda_s': args_cmd.lambda_s,  # 0.1, 1.0, 10.0,
                           'lambda_e': args_cmd.lambda_em,  # 0.1, 1.0, 10.0,
                           'feature_detach': args_cmd.feature_detach,
                       }
                   },

                   'charb_eps': args_cmd.charb_eps,
                   'charb_alpha': args_cmd.charb_alpha,
                   'lambda_c_upper': args_cmd.lambda_c_upper,
                   'lambda_c_lower': args_cmd.lambda_c_lower,
                   'lambda_r': args_cmd.lambda_r,
                   'lambda_dis': args_cmd.lambda_dis,
                   'lambda_sim': args_cmd.lambda_sim,
                   'tau': args_cmd.tau,
                   'unsupervised_log': args_cmd.unsupervised_log,
                   'loss_type': args_cmd.loss_type,
                   'crops': args_cmd.crops,
                   'removed_percentage': args_cmd.removed_percentage,
                   'flips': args_cmd.flips,
                   'jitters': args_cmd.jitters,
                   'num_pairs': args_cmd.num_pairs,
                   'similarity_threshold': args_cmd.similarity_threshold,
                   'dissimilarity_threshold': args_cmd.dissimilarity_threshold,
                   'moving_threshold': args_cmd.moving_threshold,
                   'moving_vs_static_only': args_cmd.moving_vs_static_only,
                   'simdis_type': args_cmd.simdis_type,
                   'sampling_type': args_cmd.sampling_type,
                   'kept_pairs_perc': args_cmd.kept_pairs_perc,
                   'simdis_loss': args_cmd.simdis_loss,
                   'simdis_loss_tau': args_cmd.simdis_loss_tau,
                   'batch_norm': args_cmd.batch_norm,
                   'gradient_clip': args_cmd.gradient_clip,
                   'detach_cur': args_cmd.detach_cur == "yes",
                   'teacher': args_cmd.teacher == "yes",
                   'eval_forgetting': args_cmd.eval_forgetting,
                   'eval_forgetting_fractions': args_cmd.eval_forgetting_fractions,
                   'save_time': args_cmd.save_time,
                   'preload_data': args_cmd.preload_data,
                   'teacher_ema_weight': args_cmd.teacher_ema_weight,
                   'features_scheduling': args_cmd.features_scheduling,
                   'sampling_features': args_cmd.sampling_features,
                   'simdis_neg_avg': args_cmd.simdis_neg_avg,
                   }


    net_options["total_features"] = net_options["vision_block"]["features"]["planes"][-1]


    # if architecture is "pretrained", then initilize net_option as follows
    if net_options["arch_mode"] == "pretrained":
        # assign the entire conjugate-based network (only possible when called by the offline code)
        # NB key starts with "_" in order not to save it
        net_options["_whole_net"] = pretrained_net

    # avoid computing metrics and giving supervisions --  only unsupervised training
    if args_cmd.without_supervision:

        worker = lve.WorkerConj(ins.w, ins.h, ins.c, ins.fps, ins, options={
            **general_options,
            "sup_policy": None,
            "foa": None,
            "net": net_options,
        })

        log_opts = {'': general_options,
                    'net': net_options,
                    '': {
                        'force_gray': args_cmd.force_gray,
                        'dataset': args_cmd.dataset,
                        'notes': args_cmd.notes
                    }
                    }
    else:
        # avoid computing metrics
        if args_cmd.interactive_threshold:
            worker = lve.WorkerConj(ins.w, ins.h, ins.c, ins.fps, options={
                **general_options,
                "sup_policy": sup_policy_options,
                "foa": foa_options,
                "net": net_options,
            })

            log_opts = {'': general_options,
                        'sup_policy': sup_policy_options,
                        'net': net_options,
                        '': {
                            'laps_unsup': args_cmd.laps_unsup,
                            'laps_sup': args_cmd.laps_sup,
                            'force_gray': args_cmd.force_gray,
                            'dataset': args_cmd.dataset,
                            'notes': args_cmd.notes
                        }
                        }
        else:
            metrics_options = {'window': ins.effective_video_frames,
                               'min_laps': args_cmd.laps_unsup + 1,  # we save a bit of time
                               'trash_class': ins.sup_map['background'] if 'background' in ins.sup_map else None,
                               "conj_evaluation": True
                               }

            # creating worker
            worker = lve.WorkerConj(ins.w, ins.h, ins.c, ins.fps, ins, options={
                **general_options,
                "sup_policy": sup_policy_options,
                "eq_policy": eq_policy_options,
                "foa": foa_options,
                "net": net_options,
                "metrics": metrics_options
            })

            log_opts = {'': general_options,
                        'sup_policy': sup_policy_options,
                        'eq_policy': eq_policy_options,
                        'net': net_options,
                        'metrics': metrics_options,
                        '': {
                            'laps_unsup': args_cmd.laps_unsup,
                            'laps_sup': args_cmd.laps_sup,
                            'force_gray': args_cmd.force_gray,
                            'dataset': args_cmd.dataset,
                            'notes': args_cmd.notes
                        }
                        }

    log_dict = create_log_dict(args_cmd)

    total_options = {}
    for prefix, dic in log_opts.items():
        if dic is not None:
            for key, val in dic.items():
                total_options[prefix + "_" + key] = val

    # processing stream
    if args_cmd.wandb_tag is not None:
        wandb_tags = [args_cmd.wandb_tag]
    else:
        wandb_tags = []
    if args_cmd.wandb:
        WANDB_PROJ = "cmoss"
        WANDB_ENTITY = "test"
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, config=total_options, tags=wandb_tags, notes=args_cmd.wandb_note)
        if args_cmd.model_watch:
            wandb.watch(worker.net, log="all", log_freq=10)
        RUN_ID = wandb.run.id
    else:
        from datetime import datetime
        RUN_ID = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')

    if args_cmd.dataset == "solid" and args_cmd.force_gray == "no":
        print('solid_benchmark should be run with force_gray for consistency.')
        if args_cmd.wandb:
            wandb.run.finish(exit_code=1)
        os._exit(1)

    if args_cmd.model_folder is None:
        model_folder = "models" + os.sep + RUN_ID
    else:
        model_folder = args_cmd.model_folder

    outs = lve.OutputStream(**output_settings)
    v_proc = None
    try:
        vprocessor_kwargs = {'input_stream': ins, 'output_stream': outs, 'worker': worker,
                             'model_folder': model_folder,
                             'resume': args_cmd.resume, 'wandb': args_cmd.wandb,
                             'print_every': args_cmd.print_every, 'save_every': args_cmd.save_every,
                             'visualization_port': args_cmd.visualization_port,
                             'only_metrics_save': args_cmd.only_metrics_save
                             }
        v_proc = lve.VProcessor(**vprocessor_kwargs)
        v_proc.process_video(log_dict=log_dict)
    except Exception as e:
        v_proc.visual_server.close()
        raise e

    elapsed_time = time.time() - start_time

    # closing streams
    ins.close()
    outs.close()

    print("")
    print("Elapsed: " + str(elapsed_time) + " seconds")

    # extract list of predicted motion frames
    if "pred_motion" in log_dict:
        pred_motion_videos = log_dict["motion_list"]
        list_vids = []
        for vid in range(args_cmd.n_blocks):
            extract_list = [el[vid] for el in pred_motion_videos]
            if len(extract_list) > 0:
                list_vids.append(np.stack(extract_list))

    # final evaluation
    metric_dict = {}
    if not args_cmd.without_supervision:
        if 'logged' in log_dict and log_dict['logged'] is not None and len(log_dict['logged']) > 0:
            final_stats = log_dict['logged'][-1]
            print("F1 Global (whole-frame, window, all classes + global):")
            print(final_stats['whole_frame']['window']['f1'][:])

            mapping_dict = {"whole_frame": "whole", "foa_moving": "foam", "foa": "foa"}

            for metric in ["f1", "acc"]:
                for area in ["whole_frame", "foa", "foa_moving"]:
                    for setting in ["window", "running"]:
                        key = f"{metric}_{setting}_{mapping_dict[area]}"
                        value = final_stats[area][setting][metric]
                        for i in range(len(value) - 1):
                            metric_dict[key + "_" + str(i)] = value[i]
                        metric_dict[key + "_global"] = value[-1]
                        metric_dict[key + "_good_classes"] = np.mean(value[:-2])

            for metric in ["u1", "ucc"]:
                for area in ["whole_frame"]:
                    for setting in ["window", "running"]:
                        key = f"{metric}_{setting}_{mapping_dict[area]}"
                        value = final_stats[area][setting][metric]
                        for i in range(len(value) - 1):
                            metric_dict[key + "_" + str(i)] = value[i]
                        metric_dict[key + "_global"] = value[-1]

        # dump metrics dict to file
        with open(os.path.join("model_folder", 'final_metrics.json'), 'w') as fp:
            json.dump(metric_dict, fp)

        if args_cmd.wandb and "pred_img" in log_dict and args_cmd.save_videos:
            pred_video = np.stack(log_dict["pred_list"])
            metric_dict["pred_video"] = wandb.Video(pred_video, fps=25, format="mp4")

    if args_cmd.wandb:
        if "pred_motion" in log_dict and args_cmd.save_videos:
            for j in range(args_cmd.n_blocks):
                metric_dict[f"motion_video_b{j}"] = wandb.Video(list_vids[j], fps=25, format="mp4")

        wandb.log(metric_dict)

    if args_cmd.eval_forgetting:
        best_threshold_idx = v_proc.metrics_container.pick_best_sup_stats()
        # set internal eval forgetting flag
        worker.set_eval_forgetting()

        progressive_stats = []
        for t in range(args_cmd.eval_forgetting_fractions):
            # Learning step
            learning_frames = ins.effective_video_frames // args_cmd.eval_forgetting_fractions
            ins_properties_ = dict(ins_properties)
            ins_properties_['max_frames'] = (t + 1) * learning_frames
            ins_properties_['skip_frames'] = t * learning_frames
            ins_properties_['laps'] = 1

            vprocessor_kwargs_ = dict(vprocessor_kwargs)
            ins = lve.InputStream(**ins_properties_)
            worker.ins = ins
            worker.options['net']['freeze'] = False
            worker.options['sup_policy']['type'] = "nothing"
            worker.options['metrics']['min_laps'] = 100000

            outs = lve.OutputStream(**output_settings)
            vprocessor_kwargs_.update({'input_stream': ins, 'output_stream': outs, 'worker': worker, 'resume': False, 'wandb': False})
            log_dict = create_log_dict(args_cmd)
            vproc = lve.VProcessor(**vprocessor_kwargs_)
            vproc.metrics_container.set_saved_sup_threshold_idx(best_threshold_idx)
            vproc.process_video(log_dict=log_dict)
            ins.close()
            outs.close()


            laps_unsup = 0
            laps_sup = 0
            laps_metrics = 1
            laps = laps_unsup + laps_sup + laps_metrics

            ins_properties_ = dict(ins_properties)
            ins_properties_['max_frames'] = -1
            ins_properties_['laps'] = laps

            vprocessor_kwargs_ = dict(vprocessor_kwargs)
            ins = lve.InputStream(**ins_properties_)
            worker.ins = ins
            worker.options['net']['freeze'] = True
            worker.options['sup_policy']['type'] = "nothing"
            worker.options['metrics']['min_laps'] = 1

            outs = lve.OutputStream(**output_settings)
            vprocessor_kwargs_.update({'input_stream': ins, 'output_stream': outs, 'worker': worker, 'resume': False, 'wandb': False})
            log_dict = create_log_dict(args_cmd)
            vproc = lve.VProcessor(**vprocessor_kwargs_)
            vproc.metrics_container.set_saved_sup_threshold_idx(best_threshold_idx)
            vproc.process_video(log_dict=log_dict)
            ins.close()
            outs.close()

            final_stats = log_dict['logged'][-1]
            print("F1 Global (whole-frame, window, all classes + global):")
            print(final_stats['whole_frame']['window']['f1'][:])
            progressive_stats.append(dict(final_stats))

        wandb.log(create_progressive_dict(log_dict, progressive_stats=progressive_stats))


def get_runner_parser():
    parser = argparse.ArgumentParser(description='CMOSS experiments')
    parser.add_argument('--laps_unsup', type=int, default=0)
    parser.add_argument('--laps_sup', type=int, default=1)
    parser.add_argument('--laps_metrics', type=int, default=1)
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--step_size_decay', type=float, default=None)
    parser.add_argument('--step_size_features', type=float, default=None)
    parser.add_argument('--step_size_displacements', type=float, default=None)
    parser.add_argument('--max_supervisions', type=int, default=3)
    parser.add_argument('--notes', type=str, default=None)
    parser.add_argument('--force_gray', type=str, default="yes", choices=["yes", "no"])
    parser.add_argument('--train', type=str, default="yes", choices=["yes", "no"])
    parser.add_argument('--dataset', type=str, default="emptyspace")
    parser.add_argument('--wandb_tag', type=str, default=None)
    parser.add_argument('--wandb_note', type=str, default=None)
    parser.add_argument('--model_watch', type=str, default="false",
                        help='activate wandb model_watch')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--crops', type=int, default=0, help='Activate crops augmentation')
    parser.add_argument('--removed_percentage', type=float, default=0.1, help='Removed percentage of pixels in crops augmentation')
    parser.add_argument('--flips', type=int, default=0, help='Number of flips for data augmentation, in [0, 1, 2, 3]')
    parser.add_argument('--jitters', type=int, default=0, help='Number of color jitterings for data augmentation >= 0')
    parser.add_argument('--lambda_c_lower', type=str, default="0,")
    parser.add_argument('--lambda_c_upper', type=str, default="0,")
    parser.add_argument('--lambda_r', type=float, default=0.0)
    parser.add_argument('--lambda_dis', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lambda_s', type=str, default='1,10,100')
    parser.add_argument('--lambda_ef', type=str, default='0,0,0')
    parser.add_argument('--lambda_em', type=str, default='1,1,1')
    parser.add_argument('--feature_planes', type=str, default='24,36,48')
    parser.add_argument('--charb_eps', type=float, default=0.001)
    parser.add_argument('--charb_alpha', type=float, default=0.5)
    parser.add_argument('--feature_normalization', type=str, default="yes", choices=["yes", "no"])
    parser.add_argument('--input_normalization', type=str, default="yes", choices=["yes", "no"])
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--eps_error', type=float, default=0.5)
    parser.add_argument('--reduce_data', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--features_block', type=str, default="resunetblock_bias")
    parser.add_argument('--displacement_block', type=str, default="resunetblocknolastskip")
    # if the first block must be from disk, use "motion_farneback,convblock"  or "motion_pwc,convblock"
    parser.add_argument('--arch_mode', type=str, default="scratch")
    parser.add_argument('--softmax_temp', type=float, default=1.0)
    parser.add_argument('--feature_detach', type=str, default="false")
    parser.add_argument('--preload_data', type=str, default="false")
    parser.add_argument('--first_block_identity', type=str, default="false")  # notice, if true,
    # than the feature_planes first entry must be the image channels (1 if gray, 3 if RGB)
    parser.add_argument('--plot_embeddings', type=str, default=None)
    parser.add_argument('--warping_type', type=str, default='explicit', choices=['implicit', 'explicit'])
    parser.add_argument('--gradient_type', type=str, default='sobel', choices=['hs', 'shift', 'sobel'])
    parser.add_argument('--loss_type', type=str, default='charb', choices=['charb', 'xent', 'kl_div'])
    parser.add_argument('--wandb', type=str, default="true",
                        help='activate wandb')
    parser.add_argument('--resume', type=str, default="false")
    parser.add_argument('--eval_forgetting', type=str, default="false")
    parser.add_argument('--eval_forgetting_fractions', type=int, default=None)
    parser.add_argument('--interactive_threshold', type=str, default="false",
                        help='activate interactive threshold')
    parser.add_argument('--unsupervised_log', type=str, default="false",
                        help='Log into wandb also during unsupervised learning')
    parser.add_argument('--without_supervision', type=str, default="false",
                        help='Do not use any foa information, do not compute metrics from supervised pixels')
    parser.add_argument('--model_folder', type=str, default="model_folder")
    parser.add_argument('--save_videos', type=str, default="true",
                        help='Save videos of last lap in wandb - put it to false for long streams!')
    parser.add_argument('--only_metrics_save', type=str, default="false",
                        help='Skip model save!')
    parser.add_argument('--regularization_type', type=str, default="standard",
                        help='Regularization type')
    parser.add_argument('--previous_frame_data_size', type=int, default=1,
                        help='Number of previous frames provided by the input stream at each time instant')
    parser.add_argument('--previous_frame_offsets', type=str, default=-1,
                        help='Comma-separated list of negative offsets.')
    parser.add_argument('--shuffle', type=str, default="no", choices=["yes", "no"],
                        help='Randomize data from the input stream')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Mini-batch size')
    parser.add_argument('--visualization_port', type=int, default=0,
                        help='Visualization port')
    parser.add_argument('--print_every', type=int, default=100,
                        help='Print stats every n frames')
    parser.add_argument('--weights_reinit', type=float, default=None,
                        help='Gain for weights reinitialization')

    parser.add_argument('--num_pairs', type=int, default=10000)
    parser.add_argument('--save_time', type=str, default="false")

    parser.add_argument('--lambda_sim', type=float, default="0.0")
    parser.add_argument('--similarity_threshold', type=float, default=0.0)
    parser.add_argument('--dissimilarity_threshold', type=float, default=0.0)
    parser.add_argument('--moving_threshold', type=float, default=0.5)
    parser.add_argument('--moving_vs_static_only', type=str, default="false")
    parser.add_argument('--simdis_type', type=str, default="both_mixed", choices=['single', 'both', 'both_mixed', 'mixed'])
    parser.add_argument('--sampling_type', type=str, default="motion_features",
                        choices=['plain', 'motion', 'features', 'motion_features'])
    parser.add_argument('--sampling_features', type=str, default='first',
                        choices=['first', 'second'])
    parser.add_argument('--kept_pairs_perc', type=float, default=1.0)
    parser.add_argument('--simdis_loss', type=str, default="plain", choices=['plain', 'logexp'])
    parser.add_argument('--simdis_loss_tau', type=float, default=1.0)

    parser.add_argument('--detach_cur', type=str, default="no", choices=["yes", "no"])
    parser.add_argument('--teacher', type=str, default="no", choices=["yes", "no"])
    parser.add_argument('--teacher_ema_weight', type=float, default=0.999)

    parser.add_argument('--batch_norm', type=str, default="false")
    parser.add_argument('--features_scheduling', type=int, default=None)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--consistency_type', type=str, default="plain", choices=["plain", "masked"])

    parser.add_argument('--simdis_neg_avg', type=str, default="false")

    return parser


if __name__ == "__main__":
    parser = get_runner_parser()
    args_cmd = parser.parse_args()
    args_cmd.input_channels = 3 if args_cmd.force_gray == "no" else 1
    args_cmd.feature_planes = [args_cmd.input_channels] + [int(x) for x in args_cmd.feature_planes.split(",")]
    args_cmd.lambda_c_lower = [float(x) for x in args_cmd.lambda_c_lower.split(",")]
    args_cmd.lambda_c_upper = [float(x) for x in args_cmd.lambda_c_upper.split(",")]
    args_cmd.lambda_s = [float(x) for x in args_cmd.lambda_s.split(",")]
    args_cmd.lambda_ef = [float(x) for x in args_cmd.lambda_ef.split(",")]
    args_cmd.lambda_em = [float(x) for x in args_cmd.lambda_em.split(",")]

    args_cmd.model_watch = args_cmd.model_watch in {'True', 'true'}
    args_cmd.wandb = args_cmd.wandb in {'True', 'true'}
    args_cmd.preload_data = args_cmd.preload_data in {'True', 'true'}
    args_cmd.resume = args_cmd.resume in {'True', 'true'}
    args_cmd.save_time = args_cmd.save_time in {'True', 'true'}
    args_cmd.only_metrics_save = args_cmd.only_metrics_save in {'True', 'true'}
    args_cmd.interactive_threshold = args_cmd.interactive_threshold in {'True', 'true'}

    args_cmd.unsupervised_log = args_cmd.unsupervised_log in {'True', 'true'}
    args_cmd.without_supervision = args_cmd.without_supervision in {'True', 'true'}
    args_cmd.save_videos = args_cmd.save_videos in {'True', 'true'}
    args_cmd.feature_detach = args_cmd.feature_detach in {'True', 'true'}
    args_cmd.first_block_identity = args_cmd.first_block_identity in {'True', 'true'}
    args_cmd.moving_vs_static_only = args_cmd.moving_vs_static_only in {'True', 'true'}
    args_cmd.batch_norm = args_cmd.batch_norm in {'True', 'true'}
    args_cmd.simdis_neg_avg = args_cmd.simdis_neg_avg in {'True', 'true'}
    args_cmd.eval_forgetting = args_cmd.eval_forgetting in {'True', 'true'}
    if args_cmd.previous_frame_offsets is not None:
        if len(args_cmd.previous_frame_offsets.strip()) == 0:
            args_cmd.previous_frame_offsets = None
        else:
            args_cmd.previous_frame_offsets = [int(x) for x in args_cmd.previous_frame_offsets.split(",")]
    if "dpt" in args_cmd.features_block or "dpt" in args_cmd.displacement_block:
        assert args_cmd.n_blocks == 1, "DPT architecture with n_block > 1 not supported!"

    args_cmd.features_block = feature_block_list = args_cmd.features_block.split(",")
    args_cmd.displacement_block = displacement_block_list = args_cmd.displacement_block.split(",")
    if "motion_" in displacement_block_list[0]:  # only works when first is from disk!
        args_cmd.motion_disk_type = displacement_block_list[0]
    else:
        args_cmd.motion_disk_type = None

    # to specify every block: if n_blocks ==3, and specify only two blocks, repeat the last one until number of n_blocks
    # ex: n_blocks =3, feature_block="convblock, resunetblock" => args.feature_block = [convblock, resunetblock, resunetblock]
    if args_cmd.n_blocks > 1 and (len(feature_block_list) != args_cmd.n_blocks):
        args_cmd.features_block = feature_block_list + [feature_block_list[-1]] * (
                args_cmd.n_blocks - len(feature_block_list))

    if args_cmd.n_blocks > 1 and (len(displacement_block_list) != args_cmd.n_blocks):
        args_cmd.displacement_block = displacement_block_list + [displacement_block_list[-1]] * (
                args_cmd.n_blocks - len(displacement_block_list))

    if "motion_" in args_cmd.displacement_block[0] and args_cmd.crops > 0:
        raise Exception("Cannot generate crops with motion loaded from disk!")



    # setting the seeds

    # setting up seeds for random number generators
    seed = int(time.time()) if args_cmd.seed < 0 else int(args_cmd.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # enforcing a deterministic behaviour, when possible
    py_version = torch.__version__
    if version.parse(py_version) > version.parse("1.11"):
        print(f"Pytorch version: {py_version};  compatible with detertministic algohrithms")
        torch.use_deterministic_algorithms(True, warn_only=True)

    run_exp(args_cmd)
