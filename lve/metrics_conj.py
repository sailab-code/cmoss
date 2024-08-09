import numpy
import numpy as np
import collections
from collections import OrderedDict
import lve
import os
import torch
import copy


class ConjMetricsContainer:
    def __init__(self, num_classes, w, features, output_stream, options, threshold_list):
        self.metrics_list = [
            SupervisedMetrics(num_classes, w, features, output_stream, options, thresh_idx, thresh)
            for thresh_idx, thresh in enumerate(threshold_list)]

        self.output_stream = output_stream
        self.conj_metrics = Metrics(num_classes,  w, features, output_stream, options)
        self.num_classes = num_classes  # background class is already counted
        self.w = w
        self.output_stream = output_stream
        self.window_size = options['window']  # these are the metrics-related options only
        self.trash_class = options['trash_class']  # these are the metrics-related options only
        self.features = features
        self.threshold_list = threshold_list
        self.__best_threshold_idx = None

        # only for structure

        self.__stats = OrderedDict({
            'whole_frame':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            "consistency_lower": None,
                            "consistency_upper": None,
                            "consistency_absolute": None,
                            "consistency_relative": None,
                            "motion_smoothness": None,

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            "consistency_lower": None,
                            "consistency_upper": None,
                            "consistency_absolute": None,
                            "consistency_relative": None,
                            "motion_smoothness": None,
                            "best_threshold": None
                        },
                },
            'foa':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),

                        },
                },
            'foa_moving':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),

                        }
                }
        })

        self.output_stream.register_output_elements(self.get_output_types())

    def update(self, full_sup):
        # update metrics for the various thresholds
        for i in self.metrics_list:
            i.update(full_sup)
        self.conj_metrics.update(full_sup)

    def get_saved_sup_threshold(self):
        return self.__best_threshold_idx, self.threshold_list[self.__best_threshold_idx]

    def set_saved_sup_threshold_idx(self, threshold_idx):
        self.__best_threshold_idx = threshold_idx

    def compute(self):
        for obj in self.metrics_list:
            obj.compute()
        self.conj_metrics.compute()
        # update output stream

        best_threshold_idx = self.pick_best_sup_stats() if self.__best_threshold_idx is None else self.__best_threshold_idx

        best_supervised_stats = self.metrics_list[best_threshold_idx].get_stats()
        conj_stats = self.conj_metrics.get_stats()

        dict_stats = OrderedDict()
        for area, value_area in best_supervised_stats.items():
            dict_stats[area] = {}
            for setting, value_setting in value_area.items():
                d1 = copy.deepcopy(value_setting)
                d1.update(conj_stats[area][setting])
                dict_stats[area][setting] = d1

        self.__stats = dict_stats
        self.__stats["whole_frame"]["window"]["best_threshold"] = self.threshold_list[best_threshold_idx]

        # "prediction_idx": prediction_idx_tensor_detached[0],
        best_preds = self.output_stream.get_output_elements()["prediction_idx-list"]["data"][best_threshold_idx]
        best_sup_probs = self.output_stream.get_output_elements()["sup-probs-list"]["data"][best_threshold_idx]
        self.output_stream.get_output_elements()["prediction_idx"]["data"] = best_preds
        self.output_stream.get_output_elements()["sup-probs"]["data"] = best_sup_probs

        self.output_stream.save_elements({"stats.metrics": self.__stats,  # dictionary
                                          "logs.metrics": self.__convert_stats_values_to_list(),  # CSV log
                                          "tb.metrics": self.__stats}, prev_frame=True)

    def pick_best_sup_stats(self):
        best_metric = -1.
        best_threshold_idx = None

        for obj in self.metrics_list:
            metric = obj.get_stats()["whole_frame"]["window"]["f1"][-1]
            if metric > best_metric:
                best_threshold_idx = obj.thresh_idx
                best_metric = metric

        return best_threshold_idx

    def save(self, model_folder):
        for obj in self.metrics_list:
            obj.save(model_folder)
        self.conj_metrics.save(model_folder)

    def load(self, model_folder):
        for obj in self.metrics_list:
            obj.load(model_folder)
        self.conj_metrics.load(model_folder)

    def get_output_types(self):
        output_types = {
            "stats.metrics": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.metrics": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.metrics__header": ['frame'] + self.__convert_stats_keys_to_list()
        }
        return output_types

    def __convert_stats_values_to_list(self):
        stats_list = []
        for area, area_d in self.__stats.items():
            for setting, setting_d in area_d.items():
                for metric, metric_v in setting_d.items():
                    if isinstance(metric_v, list):
                        for m_v in metric_v:
                            stats_list.append(m_v)
                    else:
                        stats_list.append(metric_v)
        return stats_list

    def __convert_stats_keys_to_list(self):
        stats_list = []
        for area, area_d in self.__stats.items():
            for setting, setting_d in area_d.items():
                for metric, metric_v in setting_d.items():
                    if isinstance(metric_v, list):
                        ml = len(metric_v)
                        for k in range(0, ml - 1):
                            stats_list.append(metric + '_c' + str(k))
                        stats_list.append(metric + '_glob')
                    else:
                        stats_list.append(metric)
        return stats_list


class Confusion:
    def __init__(self, labels, predictions):
        self.cm = numpy.zeros((labels, predictions))

    def get_cm(self):
        return self.cm


class SupervisedMetrics:
    def __init__(self, num_classes, w, features, output_stream, options, thresh_idx, thresh):
        # options
        self.num_classes = num_classes  # background class is already counted
        self.w = w
        self.output_stream = output_stream
        self.window_size = options['window']  # these are the metrics-related options only
        self.trash_class = options['trash_class']  # these are the metrics-related options only
        self.features = features
        self.thresh_idx = thresh_idx
        self.thresh = thresh

        # references to the confusion and contingency matrices
        self.running_unmasked_confusion_whole_frame = Confusion(num_classes-1, num_classes-1)
        self.running_confusion_whole_frame = Confusion(num_classes, num_classes)
        self.running_confusion_foa = Confusion(num_classes, num_classes)
        self.running_confusion_foa_moving = Confusion(num_classes, num_classes)

        self.window_unmasked_confusion_whole_frame = collections.deque(maxlen=self.window_size)
        self.window_confusion_whole_frame = collections.deque(maxlen=self.window_size)
        self.window_confusion_foa = collections.deque(maxlen=self.window_size)
        self.window_confusion_foa_moving = collections.deque(maxlen=self.window_size)
        # add initial dummy one
        self.window_confusion_foa_moving.appendleft(Confusion(num_classes, num_classes).get_cm())

        self.__stats = OrderedDict({
            'whole_frame':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'uacc': [None] * (self.num_classes),
                            'uf1': [None] * (self.num_classes),

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'uacc': [None] * (self.num_classes),
                            'uf1': [None] * (self.num_classes),
                        },
                },
            'foa':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                        },
                },
            'foa_moving':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),

                        }
                }
        })

    def get_stats(self):
        return self.__stats

    def save(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "metrics_supervised" + os.sep + str(self.thresh_idx) + os.sep
        if not os.path.exists(metrics_model_folder):
            os.makedirs(metrics_model_folder)

        # saving metrics-status related tensors
        torch.save({"running_confusion_whole_frame": self.running_confusion_whole_frame,
                    "running_confusion_foa": self.running_confusion_foa,
                    "running_confusion_foa_moving": self.running_confusion_foa_moving,
                    "window_confusion_whole_frame": self.window_confusion_whole_frame,
                    "window_confusion_foa": self.window_confusion_foa,
                    "window_confusion_foa_moving": self.window_confusion_foa_moving,
                    },
                   metrics_model_folder + "metrics.pth")

    def load(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "metrics_supervised" + os.sep + str(self.thresh_idx) + os.sep

        # loading metrics-status related tensors
        if os.path.exists(metrics_model_folder + "metrics.pth"):
            metrics_status = torch.load(metrics_model_folder + "metrics.pth")

            self.running_confusion_whole_frame = metrics_status["running_confusion_whole_frame"]
            self.running_confusion_foa = metrics_status["running_confusion_foa"]
            self.running_confusion_foa_moving = metrics_status["running_confusion_foa_moving"]
            self.window_confusion_whole_frame = metrics_status["window_confusion_whole_frame"]
            self.window_confusion_foa = metrics_status["window_confusion_foa"]
            self.window_confusion_foa_moving = metrics_status["window_confusion_foa_moving"]

    def compute_confusion(self, y, y_pred, n=None):
        if n is None: n = self.num_classes
        indices = n * y.to(torch.int64) + y_pred.to(torch.int64)
        m = torch.bincount(indices,
                           minlength=n ** 2).reshape(n, n)
        return m

    @staticmethod
    def __compute_matrices_and_update_running(pred, target, compute, running_cm, running_cm_window):
        current_cm = compute(y_pred=torch.as_tensor(pred), y=torch.as_tensor(target)).numpy()
        running_cm.cm = running_cm.cm + current_cm

        # windowed confusion matrix update
        running_cm_window.appendleft(current_cm)

    def update(self, full_sup):
        # reset the movement flag

        # get the current frame targets
        targets, indices = full_sup

        # getting model predictions, gather predictions from output stream, they are already in numpy!
        unmasked_pred_idx = self.output_stream.get_output_elements()["unmasked-prediction_idx"]["data"]
        pred_idx = self.output_stream.get_output_elements()["prediction_idx-list"]["data"][
            self.thresh_idx]  # get the current frame
        motion = self.output_stream.get_output_elements()["motion"]["data"] # get the current optical flow

        # computing confusion and contingency matrices
        SupervisedMetrics.__compute_matrices_and_update_running(pred=pred_idx, target=targets,
                                                                compute=self.compute_confusion,
                                                                running_cm=self.running_confusion_whole_frame,
                                                                running_cm_window=self.window_confusion_whole_frame)
        notbg = targets != self.num_classes-1
        unmasked_pred_idx_ = unmasked_pred_idx[notbg]
        targets_ = targets[notbg]
        def compute_confusion_(y, y_pred):
            return self.compute_confusion(y, y_pred, self.num_classes-1)
        SupervisedMetrics.__compute_matrices_and_update_running(pred=unmasked_pred_idx_, target=targets_,
                                                                compute=compute_confusion_,
                                                                running_cm=self.running_unmasked_confusion_whole_frame,
                                                                running_cm_window=self.window_unmasked_confusion_whole_frame)

        # restricting to the FOA
        # TODO check foa coords
        foax = int(self.output_stream.get_output_elements()["stats.worker"]["data"]["foax"])
        foay = int(self.output_stream.get_output_elements()["stats.worker"]["data"]["foay"])

        pred_foa = torch.tensor([pred_idx[foax * self.w + foay]])
        target_foa = torch.tensor([targets[foax * self.w + foay]])

        # computing confusion and contingency matrices (FOA only)
        SupervisedMetrics.__compute_matrices_and_update_running(pred=pred_foa, target=target_foa,
                                                                compute=self.compute_confusion,
                                                                running_cm=self.running_confusion_foa,
                                                                running_cm_window=self.window_confusion_foa)

        # computing confusion and contingency matrices (FOA moving only)
        if np.linalg.norm(motion[foax, foay, :]) > 0.:
            # set the movement flag
            # self.there_is_movement_flag = True
            SupervisedMetrics.__compute_matrices_and_update_running(pred=pred_foa, target=target_foa,
                                                                    compute=self.compute_confusion,
                                                                    running_cm=self.running_confusion_foa_moving,
                                                                    running_cm_window=self.window_confusion_foa_moving)

    @staticmethod
    def __compute__all__supervised_metrics(confusion_mat):
        per_class_accuracy, global_accuracy = Metrics.accuracy(confusion_mat)
        per_class_f1, global_f1 = Metrics.f1(confusion_mat)

        return {'acc': np.append(per_class_accuracy, global_accuracy),
                'f1': np.append(per_class_f1, global_f1),
                }

    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        """

        # computing all the metrics, given the pre-computed matrices
        metrics_whole_frame_unmasked = \
            SupervisedMetrics.__compute__all__supervised_metrics(self.running_unmasked_confusion_whole_frame.cm)
        metrics_whole_frame_window_unmasked = \
            SupervisedMetrics.__compute__all__supervised_metrics(np.sum(self.window_unmasked_confusion_whole_frame, axis=0))

        metrics_whole_frame = \
            SupervisedMetrics.__compute__all__supervised_metrics(self.running_confusion_whole_frame.cm)
        metrics_foa = \
            SupervisedMetrics.__compute__all__supervised_metrics(self.running_confusion_foa.cm)

        metrics_foa_moving = \
            SupervisedMetrics.__compute__all__supervised_metrics(self.running_confusion_foa_moving.cm)

        metrics_whole_frame_window = \
            SupervisedMetrics.__compute__all__supervised_metrics(np.sum(self.window_confusion_whole_frame, axis=0))
        metrics_foa_window = \
            SupervisedMetrics.__compute__all__supervised_metrics(np.sum(self.window_confusion_foa, axis=0))
        metrics_foa_moving_window = \
            SupervisedMetrics.__compute__all__supervised_metrics(np.sum(self.window_confusion_foa_moving, axis=0))

        self.__stats.update({
            'whole_frame':
                {
                    'running':
                        {
                            'acc': metrics_whole_frame['acc'].tolist(),
                            'f1': metrics_whole_frame['f1'].tolist(),
                            'ucc': metrics_whole_frame_unmasked['acc'].tolist(),
                            'u1': metrics_whole_frame_unmasked['f1'].tolist(),

                        },
                    'window':
                        {
                            'acc': metrics_whole_frame_window['acc'].tolist(),
                            'f1': metrics_whole_frame_window['f1'].tolist(),
                            'ucc': metrics_whole_frame_window_unmasked['acc'].tolist(),
                            'u1': metrics_whole_frame_window_unmasked['f1'].tolist(),
                        },
                },
            'foa':
                {
                    'running':
                        {
                            'acc': metrics_foa['acc'].tolist(),
                            'f1': metrics_foa['f1'].tolist(),

                        },
                    'window':
                        {
                            'acc': metrics_foa_window['acc'].tolist(),
                            'f1': metrics_foa_window['f1'].tolist(),

                        },
                },
            'foa_moving':
                {
                    'running':
                        {
                            'acc': metrics_foa_moving['acc'].tolist(),
                            'f1': metrics_foa_moving['f1'].tolist(),

                        },
                    'window':
                        {
                            'acc': metrics_foa_moving_window['acc'].tolist(),
                            'f1': metrics_foa_moving_window['f1'].tolist(),

                        }
                }
        })


class Metrics:

    def __init__(self, num_classes, w, features, output_stream, options):

        # options
        self.num_classes = num_classes  # background class is already counted
        self.w = w
        self.output_stream = output_stream
        self.window_size = options['window']  # these are the metrics-related options only
        self.trash_class = options['trash_class']  # these are the metrics-related options only
        self.features = features

        self.t_counter = 0
        self.running_consistency_lower = 0.0
        self.running_consistency_upper = 0.0
        self.consistency_window_upper = collections.deque(maxlen=self.window_size)
        self.consistency_window_lower = collections.deque(maxlen=self.window_size)

        self.running_consistency_absolute = 0.0
        self.consistency_absolute_window = collections.deque(maxlen=self.window_size)
        self.running_consistency_relative = 0.0
        self.consistency_relative_window = collections.deque(maxlen=self.window_size)
        

        self.running_smoothness = 0.0
        self.smoothness_window = collections.deque(maxlen=self.window_size)

        # references to the confusion and contingency matrices

        self.running_confusion_self_whole_frame = Confusion(num_classes, num_classes)  #
        self.running_confusion_self_foa = Confusion(num_classes, num_classes)
        self.running_confusion_self_foa_moving = Confusion(num_classes, num_classes)

        self.window_confusion_self_whole_frame = collections.deque(maxlen=self.window_size)
        self.window_confusion_self_foa = collections.deque(maxlen=self.window_size)
        self.window_confusion_self_foa_moving = collections.deque(maxlen=self.window_size)
        # add initial dummy one
        self.window_confusion_self_foa_moving.appendleft(Confusion(num_classes, num_classes).get_cm())

        self.window_contingency_whole_frame = collections.deque(maxlen=self.window_size)
        self.window_contingency_foa = collections.deque(maxlen=self.window_size)
        self.window_contingency_foa_moving = collections.deque(maxlen=self.window_size)

        self.__stats = OrderedDict({
            'whole_frame':
                {
                    'running':
                        {
                            "consistency_lower": None,
                            "consistency_upper": None,
                            "consistency_absolute": None,
                            "consistency_relative": None,
                            "motion_smoothness": None,

                        },
                    'window':
                        {

                            "consistency_lower": None,
                            "consistency_upper": None,
                            "consistency_absolute": None,
                            "consistency_relative": None,
                            "motion_smoothness": None,

                        },
                },
            'foa':
                {
                    'running':
                        {

                        },
                    'window':
                        {

                        },
                },
            'foa_moving':
                {
                    'running':
                        {

                        },
                    'window':
                        {

                        }
                }
        })

    def get_stats(self):
        return self.__stats

    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        """

        # running metrics whole life
        whole_running_consistency_lower = self.running_consistency_lower / (
                self.t_counter + 1e-20)
        whole_running_consistency_upper = self.running_consistency_upper / (
                self.t_counter + 1e-20)
        # window coherence_t
        window_consistency_upper = np.sum(self.consistency_window_upper) / len(self.consistency_window_upper)
        window_consistency_lower = np.sum(self.consistency_window_lower) / len(self.consistency_window_lower)

        # running metrics whole life
        whole_running_consistency_absolute = self.running_consistency_absolute / (
                self.t_counter + 1e-20)
        # window coherence_t
        window_consistency_absolute = np.sum(self.consistency_absolute_window) / len(self.consistency_absolute_window)

        # running metrics whole life
        whole_running_consistency_relative = self.running_consistency_relative / (
                self.t_counter + 1e-20)
        # window coherence_t
        window_consistency_relative = np.sum(self.consistency_relative_window) / len(self.consistency_relative_window)

        whole_running_smoothness = self.running_smoothness / (
                self.t_counter + 1e-20)
        # window coherence_t
        window_smoothness = np.sum(self.smoothness_window) / len(self.smoothness_window)



        self.__stats.update({
            'whole_frame':
                {
                    'running':
                        {
                            'consistency_lower': whole_running_consistency_lower,
                            'consistency_upper': whole_running_consistency_upper,
                            'consistency_absolute': whole_running_consistency_absolute,
                            'consistency_relative': whole_running_consistency_relative,
                            'motion_smoothness': whole_running_smoothness,

                        },
                    'window':
                        {
                            'consistency_lower': window_consistency_lower,
                            'consistency_upper': window_consistency_upper,
                            'consistency_absolute': window_consistency_absolute,
                            'consistency_relative': window_consistency_relative,
                            'motion_smoothness': window_smoothness,

                        },
                },
        })

    def update(self, full_sup):

        # 
        consistency_lower = self.output_stream.get_output_elements()["stats.worker"]["data"]["consistency_lower"]
        consistency_upper = self.output_stream.get_output_elements()["stats.worker"]["data"]["consistency_upper"]
        self.running_consistency_upper += consistency_upper
        self.running_consistency_lower += consistency_lower
        self.consistency_window_upper.appendleft(consistency_upper)
        self.consistency_window_lower.appendleft(consistency_lower)

        #
        consistency_absolute = self.output_stream.get_output_elements()["stats.worker"]["data"]["consistency_absolute"]
        self.running_consistency_absolute += consistency_absolute
        self.consistency_absolute_window.appendleft(consistency_absolute)

        #
        consistency_relative = self.output_stream.get_output_elements()["stats.worker"]["data"]["consistency_relative"]
        self.running_consistency_relative += consistency_relative
        self.consistency_relative_window.appendleft(consistency_relative)
        

        # 
        smoothness = self.output_stream.get_output_elements()["stats.worker"]["data"]["motion_smoothness"]
        self.running_smoothness += smoothness
        self.smoothness_window.appendleft(smoothness)

        #
        self.t_counter += 1

    def load(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "metrics_unsupervised" + os.sep

        # loading metrics-status related tensors
        if os.path.exists(metrics_model_folder + "metrics.pth"):
            metrics_status = torch.load(metrics_model_folder + "metrics.pth")
            self.t_counter = metrics_status["t_counter"]
            self.running_consistency_upper = metrics_status["running_consistency_upper"]
            self.running_consistency_lower = metrics_status["running_consistency_lower"]
            self.consistency_window_upper = metrics_status["consistency_window_upper"]
            self.consistency_window_lower = metrics_status["consistency_window_lower"]
            self.running_smoothness = metrics_status["running_smoothness"]
            self.smoothness_window = metrics_status["smoothness_window"]



    def save(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "metrics_unsupervised" + os.sep
        if not os.path.exists(metrics_model_folder):
            os.makedirs(metrics_model_folder)

        # saving metrics-status related tensors
        torch.save({
            "running_consistency_upper": self.running_consistency_upper,
            "running_consistency_lower": self.running_consistency_lower,
            "consistency_window_upper": self.consistency_window_upper,
            "consistency_window_lower": self.consistency_window_lower,
            "running_smoothness": self.running_smoothness,
            "smoothness_window": self.smoothness_window,
            "t_counter": self.t_counter
        },
            metrics_model_folder + "metrics.pth")

    def print_info(self):
        s = "   metrics {"
        i = 0
        for k, v in self.__stats.items():
            s += (k + (": {0:.3e}".format(v) if abs(v) >= 1000 else ": {0:.3f}".format(v)))
            if (i + 1) % 7 == 0:
                if i < len(self.__stats) - 1:
                    s += ",\n           "
                else:
                    s += "}"
            else:
                if i < len(self.__stats) - 1:
                    s += ", "
                else:
                    s += "}"
            i += 1

        print(s)

    def compute_confusion(self, y, y_pred):
        indices = self.num_classes * y.to(torch.int64) + y_pred.to(torch.int64)
        m = torch.bincount(indices,
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return m

    @staticmethod
    def __compute_matrices_and_update_running(pred, target, compute, running_cm, running_cm_window):
        current_cm = compute(y_pred=torch.as_tensor(pred), y=torch.as_tensor(target)).numpy()
        running_cm.cm = running_cm.cm + current_cm

        # windowed confusion matrix update
        running_cm_window.appendleft(current_cm)

    @staticmethod
    def accuracy(cm):
        acc_det = cm.sum(axis=1)
        acc_det[acc_det == 0] = 1
        per_class_accuracy = cm.diagonal() / acc_det
        global_accuracy = np.mean(per_class_accuracy)  # macro
        return per_class_accuracy, global_accuracy

    @staticmethod
    def f1(cm):
        num_classes = cm.shape[0]
        per_class_f1 = np.zeros(num_classes)

        for c in range(0, num_classes):
            tp = cm[c, c]
            fn = np.sum(cm[c, :]) - tp
            fp = np.sum(cm[:, c]) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.
            per_class_f1[c] = (2. * p * r) / (p + r) if (p + r) > 0 else 0.

        global_f1 = np.mean(per_class_f1)  # macro
        return per_class_f1, global_f1
