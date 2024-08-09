import os
import sys
import signal
import numpy as np
from functools import partial
import socket
import torch

import lve
import time
from threading import Event
import subprocess
import wandb

from lve.sup_policies.sup_policy import SupervisionPolicy
from lve.metrics_conj import ConjMetricsContainer

from lve.utils import visualize_flows, visualize_flows_no_save, flow_to_color_static, get_normalization_factor, \
    plot_standard_heatmap, plot_sampled_points
import matplotlib.pyplot as plt


class VProcessor:

    def __init__(self, input_stream, output_stream, worker, model_folder, visualization_port=0,
                 resume=False, stop_condition=None, wandb=False, save_every=1000, print_every=25, save_callback=None,
                 only_metrics_save=False):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.worker = worker
        self.model_folder = os.path.abspath(model_folder)
        self.saved_at = None
        self.save_callback = save_callback
        self.print_every = print_every

        # creating model folder
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        # locks and other remote-control-related fields
        self.__wandb = wandb
        self.__event_visualization_is_happening = Event()
        self.__event_processing_is_running = Event()
        self.__worker_options_to_change = None
        self.__event_visualization_is_happening.set()
        self.__event_processing_is_running.set()

        # registering output elements that will be saved by the worker
        self.output_stream.register_output_elements(self.worker.get_output_types())

        # metrics (registering also metric-related output elements)
        if 'metrics' in self.worker.options and self.worker.options['metrics'] is not None:

            if 'conj_evaluation' in self.worker.options['metrics']:
                self.metrics_container = ConjMetricsContainer(
                    num_classes=self.worker.options['net']['supervised_categories'],
                    w=input_stream.w,
                    options=self.worker.options['metrics'],
                    features=self.worker.options['net']["total_features"],
                    output_stream=self.output_stream,
                    threshold_list=self.worker.options['net']["dist_threshold"])
            elif 'sup_evaluation' in self.worker.options['metrics']:
                self.metrics_container = SupMetricsContainer(
                    num_classes=self.worker.options['net']['supervised_categories'],
                    w=input_stream.w,
                    options=self.worker.options['metrics'],
                    features=self.worker.options['net']["total_features"],
                    output_stream=self.output_stream,
                    threshold_list=self.worker.options['net']["dist_threshold"])
            else:
                if 'supervised_categories' in self.worker.options['net']:
                    self.metrics_container = WwMetricsContainer(
                        num_classes=self.worker.options['net']['supervised_categories'],
                        num_clusters=self.worker.options['net'][
                            'unsupervised_categories'],
                        w=input_stream.w,
                        options=self.worker.options['metrics'],
                        num_what=self.worker.options['net']["num_what"],
                        output_stream=self.output_stream,
                        threshold_list=self.worker.options['net']["dist_threshold"])
                elif 'hs_evaluation' in self.worker.options['metrics']:
                    self.metrics_container = HsMetricsContainer(
                        options=self.worker.options['metrics'],
                        output_stream=self.output_stream)
        else:
            self.metrics_container = None

        # stream options
        self.options = {'input': input_stream.readable_input,
                        'input_type': lve.InputType.readable_type(input_stream.input_type),
                        'w': input_stream.w,
                        'h': input_stream.h,
                        'c': input_stream.c,
                        'fps': input_stream.fps,
                        'frames': input_stream.frames,
                        'laps': input_stream.laps,
                        'max_frames': input_stream.max_frames,
                        'output_folder': output_stream.folder,
                        'output_folder_gzipped_bin': self.output_stream.is_gzipping_binaries(),
                        'output_folder_files_per_subfolder':
                            self.output_stream.get_max_number_of_files_per_subfolder(),
                        'output_folder_data_types': self.output_stream.get_data_types(),
                        'sup_policy': self.worker.options.get('sup_policy', None)
                        }

        # opening a visualization service
        self.visual_server = lve.VisualizationServer(visualization_port,
                                                     output_folder=output_stream.folder,
                                                     model_folder=self.model_folder,
                                                     v_processor=self)

        # running tensorboard (on the port right next the one of the visualization server)
        if self.output_stream.tensorboard:
            tensorboad_port = visualization_port + 1
            subprocess.Popen(["tensorboard --logdir=" + self.output_stream.folder + os.sep + "tensorboard" +
                              " --host 0.0.0.0 --port=" + str(tensorboad_port)], shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            tensorboad_port = -1

        # updating model options
        self.options.update({'tensorboard_port': str(tensorboad_port),
                             'visualization_server_ip': str(self.visual_server.ip),
                             'visualization_server_port': str(self.visual_server.port),
                             'visualization_server_url': "http://" + str(self.visual_server.ip) +
                                                         ":" + str(self.visual_server.port)})

        self.options["worker"] = self.worker.options

        # turning on output generation "on request"
        if self.output_stream.virtual_save:
            self.worker.set_heavy_output_data_needed(False)

        # supervision policy
        if self.options['sup_policy'] is not None:
            self.sup_policy = SupervisionPolicy.create_from_options(
                self.input_stream.effective_video_frames,
                self.options['sup_policy']
            )
        else:
            self.sup_policy = None


        self.save_every = save_every
        self.batch_size = self.options["worker"]["batch_size"] if "batch_size" in self.options["worker"] else 1
        self.save_every = max((self.save_every // self.batch_size),1) * self.batch_size

        # printing info
        self.input_stream.print_info()
        print("")
        self.output_stream.print_info()
        print("")
        self.visual_server.print_info()
        print("")

        # resuming (if needed)
        if resume:
            print("Resuming...")
            self.__resumed = True
            self.load()
        else:
            print("No resume...")
            self.__resumed = False

        self.only_metrics_save = only_metrics_save
        # saving references
        self.__stop_condition = stop_condition

        # finally, saving options to disk (keep this at the end of the method)
        self.__save_options()

    def __del__(self):
        if hasattr(self, 'visual_server'): self.visual_server.close()

    def process_video(self, log_dict=None):
        check_ctrlc = True
        elapsed_time = 0.0

        # handling CTRL-C
        def interruption(status_array_vprocessor, signal, frame):
            status_array = status_array_vprocessor[0]
            vprocessor = status_array_vprocessor[1]
            vprocessor.remote_allow_processing()
            if status_array[0] < 4:
                status_array[0] = status_array[0] + 1
                print("*** CTRL-C detected, waiting for current processing to finish...")
            else:
                print("*** Several CTRL-C detected, forcing exit.")
                os._exit(1)

        status = [1]
        if check_ctrlc:
            signal.signal(signal.SIGINT, partial(interruption, [status, self]))

        # counters and info
        steps = 0
        tot_frames_str = "/" + str(self.input_stream.frames) if self.input_stream.frames != sys.maxsize else ""
        cur_frame_number = self.input_stream.get_last_frame_number() + 1
        step_time = np.zeros(self.batch_size)
        step_io_time = np.zeros(self.batch_size)
        step_io_time_2 = np.zeros(self.batch_size)
        fps = 0.0
        batch_img = [None] * self.batch_size
        batch_index = 0

        displacement_from_disk_flag = "displacements" in self.worker.options["net"]["vision_block"] and \
                                      self.worker.options["net"]["vision_block"]["displacements"][
                                          "motion_disk_type"] is not None

        # hack for saving pred_video
        start_step = self.input_stream.frames - self.input_stream.effective_video_frames  # record only the last lap

        self.frame_flip = False
        printed = False

        if not self.__resumed:
            self.save(steps)

        save_time = self.options["worker"]["net"]["save_time"]

        # main loop over frames
        while status[0] == 1:
            saving = not save_time or (steps % self.save_every == 0)
            self.worker.set_heavy_output_data_needed(saving or steps >= start_step) # stop optimization after unsupervised runs

            if steps % self.print_every == 0:
                wandb_line = ""
                servername_line = " ({:})".format(socket.gethostname())
                print("Processing frame " + str(cur_frame_number) + tot_frames_str +
                      " {prev_time: " + "{0:.3f}".format(step_time[batch_index]) + ", prev_proctime: " +
                      "{0:.3f}".format(step_time[batch_index] -
                                       step_io_time[batch_index] -
                                       step_io_time_2[batch_index]) + ", " +
                      "avg_fps: " + "{0:.3f}".format(fps) + "}" + wandb_line + servername_line)
                printed = True

            start_step_time = time.time()
            if 'min_laps' in self.worker.options[
                "metrics"] and self.input_stream.current_repetition == \
                    self.worker.options["metrics"]['min_laps'] - 1:
                # MAJOR WARNING (TODO) in stochastic learning, the stream is NOT shuffled when providing supervisions
                # AND THE MODEL IS FROZEN!!!! (it does not happen in the non-stochastic case!)
                if self.input_stream.shuffle is True:
                    self.input_stream.set_shuffle(False, from_next_repetition=True)

            if 'min_laps' in self.worker.options[
                "metrics"] and self.input_stream.current_repetition >= \
                    self.worker.options["metrics"]['min_laps']:

                # MAJOR WARNING (TODO) in stochastic learning, the stream is NOT shuffled when providing supervisions
                # AND THE MODEL IS FROZEN!!!! (it does not happen in the non-stochastic case!)
                if not self.worker.options['net']['freeze']:
                    self.worker.options['net']['freeze'] = True

            # getting next frame(s), eventually packing them into a batched-tensor
            batch_index = steps % self.batch_size
            if batch_index == 0:
                batch_img = [None] * self.batch_size
                batch_of = [None] * self.batch_size
                batch_of_unity = [None] * self.batch_size
                batch_supervisions = [None] * self.batch_size
                batch_foa = [None] * self.batch_size

                prev_batch_img = [None] * self.batch_size
                prev_batch_of = [None] * self.batch_size
                prev_batch_supervisions = [None] * self.batch_size
                prev_batch_foa = [None] * self.batch_size

                got_something = False
                got_sup = False

                for i in range(0, self.batch_size):
                    img, of, supervisions, foa = self.input_stream.get_next()
                    prev_img_s, prev_of_s, prev_supervisions_s, prev_foa_s = \
                        self.input_stream.get_prev(self.options['worker']['previous_frame_data_size'],
                                                   offsets=self.options['worker']['previous_frame_offsets'])

                    ################ FLIPPING THE IMAGE IN TEST ONLY
                    if 'training_max_laps' in self.worker.options['net'] \
                            and self.input_stream.current_repetition > self.worker.options['net'][
                        'training_max_laps'] and 'test_flip' in self.worker.options['net'] and \
                            self.worker.options['net']['test_flip']:  # here we are only in the metrics laps
                        img = np.flip(img, axis=(0, 1)).copy() if img is not None else None
                        supervisions_values_flip = np.flip(supervisions[0]).copy() if supervisions is not None else None
                        supervisions = (supervisions_values_flip, supervisions[1]) if supervisions is not None else None

                        # of = np.flip(of).copy() # if we simply flip it, the flow direction is wrong!

                    if displacement_from_disk_flag:
                        if img is not None:
                            unity_of, of = of  # now, of is the motion from disk - the unity disk is used only for
                        else:
                            # reached end of stream
                            unity_of, of = None, None

                    # if reached the end of the stream...
                    if img is None or self.__stop_condition is not None and self.__stop_condition():
                        if i > 0:
                            batch_img = batch_img[0:i]
                            batch_of = batch_of[0:i]
                            if displacement_from_disk_flag:
                                batch_of_unity = batch_of_unity[0:i]
                            batch_supervisions = batch_supervisions[0:i]
                            batch_foa = batch_foa[0:i]

                            prev_batch_img = prev_batch_img[0:i]
                            prev_batch_of = prev_batch_of[0:i]
                            prev_batch_supervisions = prev_batch_supervisions[0:i]
                            prev_batch_foa = prev_batch_foa[0:i]
                        break
                    else:
                        got_something = True
                        batch_img[i] = img
                        batch_of[i] = of  # it can be None
                        if displacement_from_disk_flag:
                            batch_of_unity[i] = unity_of
                        batch_supervisions[i] = supervisions  # it can be None
                        batch_foa[i] = foa  # it can be None

                        prev_batch_img[i] = prev_img_s  # this is a list (that's why it ends with the plural, '_s')
                        prev_batch_of[i] = prev_of_s  # this is a list (that's why it ends with the plural, '_s')
                        prev_batch_supervisions[
                            i] = prev_supervisions_s  # this is a list (that's why it ends with '_s')
                        prev_batch_foa[i] = prev_foa_s  # this is a list (that's why it ends with the plural, '_s')

                        if supervisions is not None:
                            got_sup = True

                # purging
                if not got_sup:
                    batch_supervisions = None
                    unfiltered_batch_supervisions = None
                else:
                    if self.sup_policy is not None:
                        unfiltered_batch_supervisions = batch_supervisions.copy()
                        frame_idx = self.input_stream.get_last_frame_number()
                        if displacement_from_disk_flag:
                            batch_supervisions = self.sup_policy.apply(
                                frame_idx, batch_img,
                                batch_of_unity, batch_supervisions,
                                batch_foa
                            )
                        else:
                            batch_supervisions = self.sup_policy.apply(
                                frame_idx, batch_img,
                                batch_of, batch_supervisions,
                                batch_foa
                            )
                    else:
                        unfiltered_batch_supervisions = batch_supervisions

                step_io_time[batch_index] = time.time() - start_step_time

                # stop condition
                if not got_something:
                    print("End of stream!")
                    break

                # preparing the output stream
                self.output_stream.clear_data_of_output_elements()

                # if supervised learning and still in the learning phase, pass the whole supervisions
                if "supervised_learning" in self.worker.options["net"]:
                    if 'min_laps' in self.worker.options["metrics"] and self.input_stream.current_repetition < \
                            self.worker.options["metrics"]['min_laps']:
                        # give all the supervisions only when laps_unsup
                        self.worker.process_frame(batch_img, of=batch_of, supervisions=batch_supervisions,
                                                  unfiltered_supervisions=unfiltered_batch_supervisions,
                                                  foa=batch_foa)
                    else:
                        # processing frame (forward)
                        self.worker.process_frame(batch_img, of=batch_of, supervisions=batch_supervisions,
                                                  foa=batch_foa)
                else:
                    self.worker.set_previous_frames_data(prev_batch_img, of=prev_batch_of,
                                                         supervisions=prev_batch_supervisions, foa=prev_batch_foa)
                    self.worker.process_frame(batch_img, of=batch_of,
                                              supervisions=batch_supervisions, foa=batch_foa,
                                              unfiltered_supervisions=unfiltered_batch_supervisions)

                # printing
                if printed:
                    self.worker.print_info()
                    printed = False

                # saving output
                start_io_time_2 = time.time()

                self.output_stream.save_element("frames", batch_img[0])
                self.output_stream.save_elements(self.worker.get_output(batch_index=0))
                self.output_stream.save_done()
                step_io_time_2[batch_index] = time.time() - start_io_time_2
            else:
                step_io_time[batch_index] = time.time() - start_step_time

                # stop condition
                if batch_index >= len(batch_img):
                    print("End of stream!")
                    break

                # saving output
                start_io_time_2 = time.time()

                self.output_stream.save_element("frames", batch_img[batch_index])
                self.output_stream.save_elements(self.worker.get_output(batch_index=batch_index))
                self.output_stream.save_done()
                step_io_time_2[batch_index] = time.time() - start_io_time_2

            output_elements = self.output_stream.get_output_elements()

            if 'training_max_laps' in self.worker.options['net'] \
                    and self.input_stream.current_repetition > self.worker.options['net']['training_max_laps']:
                self.worker.options['net']['freeze'] = True

                if self.input_stream.get_last_frame_number() > self.worker.options['net']['training_max_laps'] \
                        * self.input_stream.effective_video_frames + 1:
                    self.worker.options['piggyback_frames'] = 1
            if 'training_max_frames' in self.worker.options['net'] \
                    and self.worker.options['net'][
                'training_max_frames'] is not None and self.input_stream.get_last_frame_number() > \
                    self.worker.options['net']['training_max_frames']:
                self.worker.options['net']['freeze'] = True

            # updating worker parameters (backward)
            if batch_index == self.batch_size - 1:
                internal_data = self.worker.export_internal_data()
                updated = self.worker.update_model_parameters()

            # metrics computation
            if 'sup-probs' in output_elements and output_elements["sup-probs"]["data"] is not None and \
                    "metrics" in self.worker.options and \
                    ('min_laps' not in self.worker.options["metrics"] or
                     self.input_stream.current_repetition >= self.worker.options["metrics"]['min_laps']):
                self.metrics_container.update(full_sup=unfiltered_batch_supervisions[batch_index])
                self.metrics_container.compute()
                # self.metrics.print_info() #

            if "metrics" in self.worker.options and 'hs_evaluation' in self.worker.options['metrics']:
                if 'predicted_motion' in output_elements and output_elements["predicted_motion"][
                    "data"] is not None and \
                        self.worker.options["metrics"] is not None:
                    self.metrics_container.update(
                        updated=updated,
                        hs_invariance_term=output_elements['stats.worker']['data']['loss_invariance'],
                        hs_smoothness_term=output_elements['stats.worker']['data']['loss_smoothness'],
                        hs_loss=output_elements['stats.worker']['data']['hs_loss'],
                        photo_and_smooth_loss=output_elements['stats.worker']['data'][
                            'photo_and_smooth_loss'],
                        photo_term=output_elements['stats.worker']['data']['loss_photometric'],
                        flow_std=output_elements['stats.worker']['data']['flow_std'],
                        recon_acc=output_elements['stats.worker']['data']['recon_acc'],
                        motion_mask=output_elements['motion_mask']['data'],
                        predicted_motion_mask=output_elements['predicted_motion_mask']['data']
                    )
                    self.metrics_container.compute()
                    # self.metrics.print_info() #

            # saving (every 1000 steps)
            if steps % self.save_every == 0 and (steps > 0 or (not self.__resumed)):
                if not self.only_metrics_save:
                    print("Saving model...")
                self.save(steps)

            # eventually logging on a python list some output elements that could be needed
            if log_dict is not None:
                if log_dict['element'] in output_elements:
                    elem = output_elements[log_dict['element']]['data']
                    if isinstance(elem, list):
                        log_dict['logged'].append(elem.copy())
                    elif isinstance(elem, dict):
                        log_dict['logged'].append(elem.copy())
                    elif isinstance(elem, str):
                        log_dict['logged'].append(str(elem))
                    if log_dict['log_last_only'] and len(log_dict['logged']) > 1:
                        del log_dict['logged'][0]
                if steps >= start_step:
                    if "template_0_video" in log_dict:
                        distance = 'cosine' if self.options['worker']['net']['vision_block']['features'][
                            'normalize'] else 'euclidean'
                        features = self.output_stream.get_output_elements()['features']['data']
                        template_list = self.worker.sup_buffer.get_embeddings()  # list of embedding vector (each of them dim x 1)
                        for i in range(len(template_list)):
                            template = template_list[i]  # [1, num_what ]
                            template_str = 'template_{:}_video'.format(i)

                            what_ref = template.reshape(features.shape[1], 1, 1).detach().cpu().numpy()
                            if distance == 'euclidean':
                                diff = features[0, :, :, :] - what_ref
                                dist = np.linalg.norm(diff, axis=0)
                            else:
                                dist = 1.0 - np.sum((features[0, :, :, :] * what_ref), axis=0)
                            log_dict[template_str].append(dist)

                    if "foa_video" in log_dict:
                        stats_worker = self.output_stream.get_output_elements()['stats.worker']
                        foax = stats_worker['data']['foax']
                        foay = stats_worker['data']['foay']
                        features = self.output_stream.get_output_elements()['features']['data']
                        what_ref = features[0, :, int(foax), int(foay)]
                        what_ref = what_ref.reshape(features.shape[1], 1, 1)
                        if distance == 'euclidean':
                            diff = features[0, :, :, :] - what_ref
                            dist = np.linalg.norm(diff, axis=0)
                        else:
                            dist = 1.0 - np.sum((features[0, :, :, :] * what_ref), axis=0)
                        log_dict["foa_video"].append(dist)
                        log_dict["foa_coords"].append((foax, foay))
                    # append frames predictions only when required
                    if "pred_img" in log_dict:
                        if log_dict['pred_img'] in self.output_stream.get_output_elements():
                            pred = self.output_stream.get_output_elements()[log_dict['pred_img']]['data']
                            rgb = lve.utils.indices_to_rgb(pred)
                            rgb = rgb.reshape((self.input_stream.h, self.input_stream.w, 3))
                            rgb = np.transpose(rgb, (2, 0, 1))  # channels as first dim
                            log_dict['pred_list'].append(rgb)
                    if "upred_img" in log_dict:
                        if log_dict['upred_img'] in self.output_stream.get_output_elements():
                            pred = self.output_stream.get_output_elements()[log_dict['upred_img']]['data']
                            rgb = lve.utils.indices_to_rgb(pred)
                            rgb = rgb.reshape((self.input_stream.h, self.input_stream.w, 3))
                            rgb = np.transpose(rgb, (2, 0, 1))  # channels as first dim
                            log_dict['upred_list'].append(rgb)
                    # append frames predicted motions only when required
                    if "pred_motion" in log_dict:
                        if 'net_motion.0' in self.output_stream.get_output_elements():
                            list_motions = log_dict['pred_motion']
                            pred_list_motion = []
                            for mot in list_motions:  # for every motion produced by different blocks
                                flows = self.output_stream.get_output_elements()[mot][
                                    'data']  # get the motion of the block
                                flow_u = flows[0, 0, :, :]
                                flow_v = flows[0, 1, :, :]
                                flow_uv = np.stack((flow_u, flow_v), axis=2)
                                factor = get_normalization_factor(self)
                                flow_color = flow_to_color_static(flow_uv, static_normalization_factor=factor,
                                                                  convert_to_bgr=False)
                                pred_list_motion.append(np.transpose(flow_color, (2, 0, 1)))
                            log_dict['motion_list'].append(pred_list_motion)

            # locking and other remote-control-related procedures
            self.__event_processing_is_running.set()
            self.__event_visualization_is_happening.wait()
            self.__event_processing_is_running.clear()

            # handling received commands (and updating the output data, if some commands had effects on such data)
            if self.worker.handle_commands(batch_index=batch_index):
                self.output_stream.save_elements(self.worker.get_output(batch_index=batch_index), prev_frame=True)
            self.__handle_hot_option_changes()

            # updating system status
            cur_frame_number = cur_frame_number + 1
            steps = steps + 1
            end_of_step_time = time.time()
            step_time[batch_index] = end_of_step_time - start_step_time

            if batch_index == self.batch_size - 1:
                if steps == self.batch_size:
                    elapsed_time = np.sum(step_time)
                elapsed_time = elapsed_time * 0.95 + np.sum(step_time) * 0.05
                fps = self.batch_size / elapsed_time
            elif steps == 1:
                fps = 1.0 / step_time[0]

        # last save
        self.save(steps)
        print("Done! (model saved)")

        # quit the visualization service
        self.__event_visualization_is_happening.set()
        self.__event_processing_is_running.set()
        self.visual_server.close()

    def load(self):

        # loading options
        loaded_options = lve.utils.load_json(self.model_folder + os.sep + "options.json")

        # checking if inout stream options have changed
        input_steam_opts = ['input', 'w', 'h', 'c', 'fps', 'frames', 'laps', 'max_frames']

        input_stream_changed = False
        for io_opt in input_steam_opts:
            if self.options[io_opt] != loaded_options[io_opt]:
                input_stream_changed = True
                print("WARNING: input stream configuration has changed! current " + io_opt + ": " +
                      str(self.options[io_opt]) + ", loaded " + io_opt + ": " + str(loaded_options[io_opt]))

        try:
            # loading status info
            status_info = lve.utils.load_json(self.model_folder + os.sep + "status_info.json")

            # setting up the input stream
            if not input_stream_changed:
                self.input_stream.set_last_frame_number(status_info['input_stream.last_frame_number'])
                assert abs(self.input_stream.get_last_frame_time() - status_info['input_stream.last_frame_time']) < 0.01

            # setting up the output stream
            if not self.output_stream.is_newly_created():
                self.output_stream.set_last_frame_number(status_info['output_stream.last_frame_number'])
        except:
            print('Cannot load status')

        try:
            if self.sup_policy is not None:
                self.sup_policy.load(self.model_folder)
        except:
            print('Cannot load sup policy')

        try:
            # setting up the main worker
            self.worker.load(self.model_folder)
            print('Loaded worker')
        except Exception as e:
            print(e)

        try:
            # loading metrics
            self.metrics_container.load(self.model_folder, self.worker.options['metrics'])
        except:
            print('Cannot load metrics')

    def save(self, steps):
        if not self.only_metrics_save:
            if self.saved_at is not None and steps - self.saved_at <= 1:
                print('*** Skipping saving, saved', steps - self.saved_at, 'steps ago')
                return False

            status_info = {'input_stream.last_frame_number': self.input_stream.get_last_frame_number(),
                           'input_stream.last_frame_time': self.input_stream.get_last_frame_time(),
                           'output_stream.last_frame_number': self.output_stream.get_last_frame_number(),
                           'number_of_frames_processed_during_last_run': steps}

            # saving stream/processing status
            lve.utils.save_json(self.model_folder + os.sep + "status_info.json", status_info)

            if self.sup_policy is not None:
                # save supervision policy
                self.sup_policy.save(self.model_folder)

            # saving worker
            self.worker.save(self.model_folder)

        stats_worker = self.output_stream.get_output_elements()['stats.worker']

        if self.metrics_container is not None:
            stats_metrics = self.output_stream.get_output_elements()['stats.metrics']
        else:
            stats_metrics = {'data': None}

        if stats_worker['data'] is not None and len(stats_worker['data']) > 0 and \
                'loss' in stats_worker['data'] and np.isnan(stats_worker['data']['loss']):
            print('Loss is nan')  # CHECK
            if self.__wandb:
                wandb.run.finish(exit_code=1)
            os._exit(1)

        if self.__wandb and stats_metrics['data'] is not None and len(stats_metrics['data']) > 0 \
                and stats_worker['data'] is not None and len(stats_worker['data']) > 0:
            if 'what' in self.output_stream.get_output_elements():
                foax = stats_worker['data']['foax']
                foay = stats_worker['data']['foay']
                what = self.output_stream.get_output_elements()['what']['data']
                what_ref = what[0, :, int(foax), int(foay)]
                lve.utils.plot_what_heatmap(what, what_ref=what_ref, anchor=(foax, foay),
                                            distance='cosine' if self.options['worker']['net'][
                                                'normalize'] else 'euclidean',
                                            out=self.model_folder + os.sep + 'spatial_coherence_heatmap.png')
                heatmaps = {'spatial_coherence_heatmap': wandb.Image(
                    self.model_folder + os.sep + 'spatial_coherence_heatmap.png')}
                _, template_classes = self.worker.sup_buffer.get_embeddings_labels()

                if True:

                    template_list = self.worker.sup_buffer.get_embeddings()  # list of embedding vector (each of them dim x 1)
                    for i in range(len(template_list)):
                        template = template_list[i]  # [1, num_what ]
                        template_str = 'template_{:}_heatmap'.format(i)
                        lve.utils.plot_what_heatmap(what, what_ref=template.cpu().numpy(), anchor=None,
                                                    distance='cosine' if self.options['worker']['net'][
                                                        'normalize'] else 'euclidean',
                                                    out=self.model_folder + os.sep + template_str + '.png')
                        heatmaps[template_str] = wandb.Image(self.model_folder + os.sep + template_str + '.png')

                wandb_dict = {
                    'loss': stats_worker['data']['loss'],
                    'loss_t': stats_worker['data']['loss_t'],
                    'loss_s_in': stats_worker['data']['loss_s_in'],
                    'loss_s_out': stats_worker['data']['loss_s_out'],
                    'coherence_loss_t_window': stats_metrics['data']["whole_frame"]["window"]["coherence_loss_t"],
                    'coherence_loss_s_in_window': stats_metrics['data']["whole_frame"]["window"]["coherence_loss_s_in"],
                    'coherence_loss_s_out_window': stats_metrics['data']["whole_frame"]["window"][
                        "coherence_loss_s_out"],
                    'coherence_loss_t_running': stats_metrics['data']["whole_frame"]["running"]["coherence_loss_t"],
                    'coherence_loss_s_in_running': stats_metrics['data']["whole_frame"]["running"][
                        "coherence_loss_s_in"],
                    'coherence_loss_s_out_running': stats_metrics['data']["whole_frame"]["running"][
                        "coherence_loss_s_out"],
                    'best_threshold': stats_metrics['data']["whole_frame"]["window"]["best_threshold"],
                    'frame': wandb.Image(self.output_stream.get_output_elements()['frames']['data']),
                    'template_classes': template_classes.cpu().numpy(),
                    'pred_img': wandb.Image(
                        lve.utils.indices_to_rgb(self.output_stream.get_output_elements()['prediction_idx']['data']) \
                            .reshape((self.input_stream.h, self.input_stream.w, 3)))
                }

                wandb_dict.update(heatmaps)

            if 'sup_evaluation' in self.worker.options['metrics']:

                foax = stats_worker['data']['foax']
                foay = stats_worker['data']['foay']
                features = self.output_stream.get_output_elements()['features']['data']
                old_features = self.output_stream.get_output_elements()['old_features']['data']
                warped_features = self.output_stream.get_output_elements()['warped_features']['data']
                what_ref = features[0, :, int(foax), int(foay)]
                plt_heat = lve.utils.plot_what_heatmap_no_save(features, what_ref=what_ref, anchor=(foax, foay),
                                                               distance='cosine' if
                                                               self.options['worker']['net']['vision_block'][
                                                                   'features'][
                                                                   'normalize'] else 'euclidean')
                heatmaps = {'spatial_coherence_heatmap': wandb.Image(plt_heat)}
                _, template_classes = self.worker.sup_buffer.get_embeddings_labels()

                template_list = self.worker.sup_buffer.get_embeddings()  # list of embedding vector (each of them dim x 1)
                for i in range(len(template_list)):
                    template = template_list[i]  # [1, num_what ]
                    template_str = 'template_{:}_heatmap'.format(i)
                    pltheat = lve.utils.plot_what_heatmap_no_save(features, what_ref=template.cpu().numpy(),
                                                                  anchor=None,
                                                                  distance='cosine' if
                                                                  self.options['worker']['net']['vision_block'][
                                                                      'features']['normalize'] else 'euclidean',
                                                                  )
                    heatmaps[template_str] = wandb.Image(pltheat)

                wandb_dict = {
                    'loss': stats_worker['data']['loss'],
                    'acc': stats_worker['data']['acc'],
                    'f1': stats_worker['data']['f1'],
                    'sup_loss': stats_worker['data']['sup_loss'],
                    'spatial_coherence': stats_worker['data']['spatial_coherence'],
                    'closeness': stats_worker['data']['closeness'],
                    'similarity_loss': stats_worker['data']['similarity_loss'],
                    'dissimilarity_loss': stats_worker['data']['dissimilarity_loss'],
                    'simdissim_loss': stats_worker['data']['dissimilarity_loss'] + stats_worker['data']['similarity_loss'],
                    'best_threshold': stats_metrics['data']["whole_frame"]["window"]["best_threshold"],
                    'frame': wandb.Image(self.output_stream.get_output_elements()['frames']['data'][..., ::-1]),
                    'old_frame': wandb.Image(self.output_stream.get_output_elements()['old_frames']['data'][..., ::-1]),
                    'template_classes': template_classes.cpu().numpy(),
                    'pred_img': wandb.Image(
                        lve.utils.indices_to_rgb(self.output_stream.get_output_elements()['prediction_idx']['data']) \
                            .reshape((self.input_stream.h, self.input_stream.w, 3))),
                    'upred_img': wandb.Image(
                        lve.utils.indices_to_rgb(
                            self.output_stream.get_output_elements()['unmasked-prediction_idx']['data']) \
                            .reshape((self.input_stream.h, self.input_stream.w, 3)))
                }

                output_elem = self.output_stream.get_output_elements()
                fr = output_elem['frames']['data'][..., ::-1]
                for i in range(0, self.options['worker']['net']['n_blocks']):
                    motion = output_elem[f"net_motion.{i}"]['data']
                    motion_dic, rm_fnames_ = visualize_flows_no_save(motion, prefix=f'predicted_motion.{i}')
                    wandb_dict.update(motion_dic)

                    if sum(self.options['worker']['net']['vision_block']['features']['lambda_mf']) > 0.:
                        if f"closeness_2d.{i}" in output_elem and output_elem[f"closeness_2d.{i}"] is not None:
                            closeness_2d = output_elem[f"closeness_2d.{i}"]['data']
                            wandb_dict.update({f'closeness_2d{i}': plot_standard_heatmap(torch.tensor(closeness_2d))})

                    if self.options['worker']['net']['lambda_sim'] + self.options['worker']['net']['lambda_dis'] != 0:
                        if f"simdissim_points.{i}" in output_elem and output_elem[f"simdissim_points.{i}"]['data'] is not None:
                            simdissim_points = output_elem[f"simdissim_points.{i}"]['data']
                            wandb_dict.update({f'simdissim_points{i}': plot_sampled_points(simdissim_points, h=self.input_stream.h, w=self.input_stream.w, underlay=fr)})

                    if sum(self.options['worker']['net']['vision_block']['features']['lambda_sp']) != 0.:
                        if f"fsmoothness_2d.{i}" in output_elem and output_elem[f"fsmoothness_2d.{i}"]['data'] is not None:
                            fsmoothness_2d = output_elem[f"fsmoothness_2d.{i}"]['data']
                            wandb_dict.update(
                                {
                                    f'fsmoothness_2d{i}': plot_standard_heatmap(torch.tensor(fsmoothness_2d)),
                                })
                            super_features_2d = output_elem[f"super_features_2d.{i}"]['data']
                            super_features_norm = lve.utils.plot_grid(super_features_2d[0], normalize=False)
                            wandb_dict.update({"super_features_norm_color": wandb.Image(super_features_norm)})
                            out_features_2d = output_elem[f"out_features_2d.{i}"]['data']
                            out_features_norm = lve.utils.plot_grid(out_features_2d[0], normalize=False)
                            wandb_dict.update({"out_features_norm_color": wandb.Image(out_features_norm)})
                            plt.close()
                wandb_dict.update(heatmaps)

                # get predicted flow and predicted frames

                for i in range(0, self.options['worker']['net']['n_blocks']):
                    motion = self.output_stream.get_output_elements()[f"net_motion.{i}"]['data']
                    motion_dic, rm_fnames_ = visualize_flows_no_save(motion, prefix=f'predicted_motion.{i}')
                    wandb_dict.update(motion_dic)

                wandb_dict.update(heatmaps)

                # get predicted flow and predicted frames

                n_blockss = self.options['worker']['net']['n_blocks']

                # convert again to torch because wandb uses the torchvision make_grid utility in case of torch tensors
                wandb_dict.update({"features": wandb.Image(torch.from_numpy(features[0]).unsqueeze(1))})
                wandb_dict.update({"old_features": wandb.Image(torch.from_numpy(old_features[0]).unsqueeze(1))})
                wandb_dict.update({"warped_features": wandb.Image(torch.from_numpy(warped_features[0]).unsqueeze(1))})
                wandb.log(wandb_dict)
                plt.close('all')

            if 'conj_evaluation' in self.worker.options['metrics']:

                foax = stats_worker['data']['foax']
                foay = stats_worker['data']['foay']
                features = self.output_stream.get_output_elements()['features']['data']
                old_features = self.output_stream.get_output_elements()['old_features']['data']
                warped_features = self.output_stream.get_output_elements()['warped_features']['data']
                what_ref = features[0, :, int(foax), int(foay)]
                plt_heat = lve.utils.plot_what_heatmap_no_save(features, what_ref=what_ref, anchor=(foax, foay),
                                                               distance='cosine' if
                                                               self.options['worker']['net']['vision_block'][
                                                                   'features'][
                                                                   'normalize'] else 'euclidean')
                heatmaps = {'spatial_coherence_heatmap': wandb.Image(plt_heat)}
                _, template_classes = self.worker.sup_buffer.get_embeddings_labels()

                template_list = self.worker.sup_buffer.get_embeddings()  # list of embedding vector (each of them dim x 1)
                for i in range(len(template_list)):
                    template = template_list[i]  # [1, num_what ]
                    template_str = 'template_{:}_heatmap'.format(i)
                    pltheat = lve.utils.plot_what_heatmap_no_save(features, what_ref=template.cpu().detach().numpy(),
                                                                  anchor=None,
                                                                  distance='cosine' if
                                                                  self.options['worker']['net']['vision_block'][
                                                                      'features']['normalize'] else 'euclidean',
                                                                  )
                    heatmaps[template_str] = wandb.Image(pltheat)

                wandb_dict = {
                    'loss': stats_worker['data']['loss'],
                    'consistency_lower': stats_worker['data']['consistency_lower'],
                    'consistency_upper': stats_worker['data']['consistency_upper'],
                    'consistency_absolute': stats_worker['data']['consistency_absolute'],
                    'consistency_relative': stats_worker['data']['consistency_relative'],
                    'similarity_loss': stats_worker['data']['similarity_loss'],
                    'dissimilarity_loss': stats_worker['data']['dissimilarity_loss'],
                    'simdissim_loss': stats_worker['data']['dissimilarity_loss'] + stats_worker['data']['similarity_loss'],
                    'motion_smoothness': stats_worker['data']['motion_smoothness'],
                    'consistency_window_lower': stats_metrics['data']["whole_frame"]["window"]["consistency_lower"],
                    'consistency_window_upper': stats_metrics['data']["whole_frame"]["window"]["consistency_upper"],
                    'consistency_absolute_window': stats_metrics['data']["whole_frame"]["window"][
                        "consistency_absolute"],
                    'consistency_relative_window': stats_metrics['data']["whole_frame"]["window"][
                        "consistency_relative"],
                    'motion_smoothness_window': stats_metrics['data']["whole_frame"]["window"]["motion_smoothness"],
                    'running_consistency_lower': stats_metrics['data']["whole_frame"]["running"]["consistency_lower"],
                    'running_consistency_upper': stats_metrics['data']["whole_frame"]["running"]["consistency_upper"],
                    'running_consistency_absolute': stats_metrics['data']["whole_frame"]["running"][
                        "consistency_absolute"],
                    'running_consistency_relative': stats_metrics['data']["whole_frame"]["running"][
                        "consistency_relative"],
                    'running_motion_smoothness': stats_metrics['data']["whole_frame"]["running"][
                        "motion_smoothness"],
                    'best_threshold': stats_metrics['data']["whole_frame"]["window"]["best_threshold"],
                    'frame': wandb.Image(self.output_stream.get_output_elements()['frames']['data'][..., ::-1]),
                    'old_frame': wandb.Image(self.output_stream.get_output_elements()['old_frames']['data'][..., ::-1]),
                    'template_classes': template_classes.cpu().numpy(),
                    'pred_img': wandb.Image(
                        lve.utils.indices_to_rgb(self.output_stream.get_output_elements()['prediction_idx']['data']) \
                            .reshape((self.input_stream.h, self.input_stream.w, 3))),
                    'upred_img': wandb.Image(
                        lve.utils.indices_to_rgb(
                            self.output_stream.get_output_elements()['unmasked-prediction_idx']['data']) \
                            .reshape((self.input_stream.h, self.input_stream.w, 3)))
                }

                output_elem = self.output_stream.get_output_elements()
                fr = output_elem['frames']['data'][..., ::-1]
                for i in range(0, self.options['worker']['net']['n_blocks']):
                    if output_elem[f"dfeat_x_2d.{i}"]['data'] is not None:
                        d_feat_x = output_elem[f"dfeat_x_2d.{i}"]['data']
                        d_feat_y = output_elem[f"dfeat_y_2d.{i}"]['data']

                        wandb_dict.update({f'dfeat_x_b{i}': plot_standard_heatmap(torch.tensor(d_feat_x)),
                                           f'dfeat_y_b{i}': plot_standard_heatmap(torch.tensor(d_feat_y))
                                           })
                    d_flow_x = output_elem[f"dflow_x_2d.{i}"]['data']
                    d_flow_y = output_elem[f"dflow_y_2d.{i}"]['data']
                    wandb_dict.update({f'dflow_x_b{i}': plot_standard_heatmap(torch.tensor(d_flow_x)),
                                       f'dflow_y_b{i}': plot_standard_heatmap(torch.tensor(d_flow_y)),
                                       })

                    if self.options['worker']['net']['lambda_sim'] + self.options['worker']['net']['lambda_dis'] != 0:

                        if f"simdissim_points.{i}" in output_elem and output_elem[f"simdissim_points.{i}"]['data'] is not None:
                            simdissim_points = output_elem[f"simdissim_points.{i}"]['data']
                            wandb_dict.update({f'simdissim_points{i}': plot_sampled_points(simdissim_points, h=self.input_stream.h, w=self.input_stream.w, underlay=fr)})

                    if f"occl_2d_lower.{i}" in output_elem and output_elem[f"occl_2d_lower.{i}"]['data'] is not None:
                        occl_2d_lower = output_elem[f"occl_2d_lower.{i}"]['data']
                        wandb_dict.update({f'occl_2d_lower_b{i}': plot_standard_heatmap(torch.tensor(occl_2d_lower))})

                    if f"occl_2d_upper.{i}" in output_elem and output_elem[f"occl_2d_upper.{i}"]['data'] is not None:
                        occl_2d_upper = output_elem[f"occl_2d_upper.{i}"]['data']
                        wandb_dict.update({f'occl_2d_upper_b{i}': plot_standard_heatmap(torch.tensor(occl_2d_upper))})

                    motion = output_elem[f"net_motion.{i}"]['data']
                    motion_dic, rm_fnames_ = visualize_flows_no_save(motion, prefix=f'predicted_motion.{i}')
                    wandb_dict.update(motion_dic)
                wandb_dict.update(heatmaps)

                # get predicted flow and predicted frames

                for i in range(0, self.options['worker']['net']['n_blocks']):
                    motion = self.output_stream.get_output_elements()[f"net_motion.{i}"]['data']
                    motion_dic, rm_fnames_ = visualize_flows_no_save(motion, prefix=f'predicted_motion.{i}')
                    wandb_dict.update(motion_dic)

                wandb_dict.update(heatmaps)

                # get predicted flow and predicted frames

                n_blockss = self.options['worker']['net']['n_blocks']


                features_unnorm = lve.utils.make_grid_numpy(np.expand_dims(features[0], axis=1))
                wandb_dict.update({"features_unnorm": wandb.Image(features_unnorm)})
                plt.close()
                features_norm = lve.utils.plot_grid(features[0], normalize=False)
                wandb_dict.update({"features_norm_color": wandb.Image(features_norm)})
                plt.close()
                # convert again to torch because wandb uses the torchvision make_grid utility in case of torch tensors
                # wandb_dict.update({"features": wandb.Image(torch.from_numpy(features[0]).unsqueeze(1))})
                old_features_unnorm = lve.utils.make_grid_numpy(np.expand_dims(old_features[0], axis=1))
                wandb_dict.update({"old_features_unnorm": wandb.Image(old_features_unnorm)})
                plt.close()
                warped_features_unnorm = lve.utils.make_grid_numpy(np.expand_dims(warped_features[0], axis=1))
                wandb_dict.update({"warped_features_unnorm": wandb.Image(warped_features_unnorm)})
                plt.close()

                wandb_dict.update(
                    {f"consistency_lower_b{i}": stats_worker['data'][f"consistency_lower_b{i}"] for i in
                     range(n_blockss)})
                wandb_dict.update(
                    {f"consistency_upper_b{i}": stats_worker['data'][f"consistency_upper_b{i}"] for i in
                     range(n_blockss)})
                wandb_dict.update(
                    {f"consistency_absolute_b{i}": stats_worker['data'][f"consistency_absolute_b{i}"] for i in
                     range(n_blockss)})
                wandb_dict.update(
                    {f"consistency_relative_b{i}": stats_worker['data'][f"consistency_relative_b{i}"] for i in
                     range(n_blockss)})

                wandb.log(wandb_dict)
                plt.close('all')
            else:
                if self.save_callback is None:
                    if 'predicted_motion' in self.output_stream.get_output_elements():
                        warped, prev_frame, frame, flow, motion, motion_mask, predicted_motion_mask = self.worker.get_warped_frame()
                        wandb_dict = {
                            'photo_and_smooth_w': stats_metrics['data']["whole_frame"]["window"]["photo_and_smooth"],
                            'hs_w': stats_metrics['data']["whole_frame"]["window"]["hs"],
                            'hs_invariance_w': stats_metrics['data']["whole_frame"]["window"]["hs_invariance"],
                            'hs_smoothness_w': stats_metrics['data']["whole_frame"]["window"]["hs_smoothness"],
                            'warped': wandb.Image(warped[0, 0]),
                            'prev_frame': wandb.Image(prev_frame[0, 0]),
                            'frame': wandb.Image(frame[0, 0]),
                            'largest_predicted_flow_x': stats_worker['data']["largest_predicted_flow_x"],
                            'largest_predicted_flow_y': stats_worker['data']["largest_predicted_flow_y"],
                            'error_rate_w': stats_metrics['data']["whole_frame"]["window"]["error_rate"],
                            'l2_dist_w': stats_metrics['data']["whole_frame"]["window"]["l2_dist"],
                            'l2_dist_moving_w': stats_metrics['data']["whole_frame"]["window"]["l2_dist_moving"],
                            'l2_dist_still_w': stats_metrics['data']["whole_frame"]["window"]["l2_dist_still"],
                            'eq_iterations_w': stats_metrics['data']["whole_frame"]["window"]["eq_iterations"],
                            'recon_acc_w': stats_metrics['data']["whole_frame"]["window"]["recon_acc"],
                            'moving_f1_w': stats_metrics['data']["whole_frame"]["window"]["moving_f1"],
                            'moving_acc_w': stats_metrics['data']["whole_frame"]["window"]["moving_acc"],
                            'moving_cm_w': stats_metrics['data']["whole_frame"]["window"]["moving_cm"],
                            'moving_precision_w': stats_metrics['data']["whole_frame"]["window"]["moving_precision"],
                            'moving_recall_w': stats_metrics['data']["whole_frame"]["window"]["moving_recall"]
                        }
                        flow_dic, rm_fnames = visualize_flows(flow, prefix='flow')
                        wandb_dict.update(flow_dic)
                        motion_dic, rm_fnames_ = visualize_flows(motion, prefix='ground_truth')
                        wandb_dict.update(motion_dic)

                        wandb.log(wandb_dict)
                        for x in rm_fnames + rm_fnames_:
                            os.remove(x)
                else:
                    self.save_callback(stats_metrics, stats_worker)

        if self.__wandb and 'sup_evaluation' in self.worker.options['metrics'] and \
                'supervised_log' in self.options['worker']['net'] and stats_worker[
            'data'] is not None and len(stats_worker['data']) > 0:
            if stats_metrics['data'] is not None and len(stats_metrics['data']) > 0:
                pass
            else:
                wandb_dict = {
                    'loss': stats_worker['data']['loss'],
                    'acc': stats_worker['data']['acc'],
                    'f1': stats_worker['data']['f1'],
                    'sup_loss': stats_worker['data']['sup_loss'],
                    'spatial_coherence': stats_worker['data']['spatial_coherence'],
                    'closeness': stats_worker['data']['closeness'],
                    'frame': wandb.Image(self.output_stream.get_output_elements()['frames']['data'][..., ::-1]),
                    'old_frame': wandb.Image(self.output_stream.get_output_elements()['old_frames']['data'][..., ::-1]),
                }

                foax = stats_worker['data']['foax']
                foay = stats_worker['data']['foay']
                features = self.output_stream.get_output_elements()['features']['data']
                what_ref = features[0, :, int(foax), int(foay)]
                plt_heat = lve.utils.plot_what_heatmap_no_save(features, what_ref=what_ref, anchor=(foax, foay),
                                                               distance='cosine' if
                                                               self.options['worker']['net']['vision_block'][
                                                                   'features'][
                                                                   'normalize'] else 'euclidean')
                heatmaps = {'spatial_coherence_heatmap': wandb.Image(plt_heat)}
                # wandb_dict.update({"features": wandb.Image(torch.from_numpy(features[0]).unsqueeze(1))})
                # print(features[0].shape)
                features_unnorm = lve.utils.make_grid_numpy(np.expand_dims(features[0], axis=1))
                wandb_dict.update({"features_unnorm": wandb.Image(features_unnorm)})
                features_norm = lve.utils.plot_grid(features[0], normalize=False)
                wandb_dict.update({"features_norm_color": wandb.Image(features_norm)})

                output_elem = self.output_stream.get_output_elements()
                fr = output_elem['frames']['data'][..., ::-1]
                for i in range(0, self.options['worker']['net']['n_blocks']):
                    motion = output_elem[f"net_motion.{i}"]['data']
                    motion_dic, rm_fnames_ = visualize_flows_no_save(motion, prefix=f'predicted_motion.{i}')
                    wandb_dict.update(motion_dic)


                    if sum(self.options['worker']['net']['lambda_c_upper']) > 0.:
                        consistency_upper_2d = output_elem[f"consistency_upper_2d.{i}"]['data']
                        consistency_relative_2d = output_elem[f"consistency_relative_2d.{i}"]['data']
                        consistency_absolute_2d = output_elem[f"consistency_absolute_2d.{i}"]['data']
                        wandb_dict.update(
                            {f'consistency_upper_2d{i}': plot_standard_heatmap(consistency_upper_2d, underlay=fr)})
                        wandb_dict.update({f'consistency_relative_2d{i}': plot_standard_heatmap(consistency_relative_2d,
                                                                                                underlay=fr)})
                        wandb_dict.update({f'consistency_absolute_2d{i}': plot_standard_heatmap(consistency_absolute_2d,
                                                                                                underlay=fr)})
                    if self.options['worker']['net']['lambda_sim'] + self.options['worker']['net']['lambda_dis'] != 0:
                        if f"simdissim_points.{i}" in output_elem and output_elem[f"simdissim_points.{i}"]['data'] is not None:
                            simdissim_points = output_elem[f"simdissim_points.{i}"]['data']
                            wandb_dict.update({f'simdissim_points{i}': plot_sampled_points(simdissim_points, h=self.input_stream.h, w=self.input_stream.w, underlay=fr)})

                    if sum(self.options['worker']['net']['vision_block']['features']['lambda_mf']) > 0.:
                        if f"closeness_2d.{i}" in output_elem and output_elem[f"closeness_2d.{i}"]['data'] is not None:
                            closeness_2d = output_elem[f"closeness_2d.{i}"]['data']
                            wandb_dict.update({f'closeness_2d{i}': plot_standard_heatmap(torch.tensor(closeness_2d))})

                    if self.options['worker']['net']['lambda_sim'] + self.options['worker']['net']['lambda_dis'] != 0:
                        if f"simdissim_points.{i}" in output_elem and output_elem[f"simdissim_points.{i}"]['data'] is not None:
                            simdissim_points = output_elem[f"simdissim_points.{i}"]['data']
                            wandb_dict.update({f'simdissim_points{i}': plot_sampled_points(simdissim_points, h=self.input_stream.h, w=self.input_stream.w, underlay=fr)})

                    if sum(self.options['worker']['net']['vision_block']['features']['lambda_sp']) != 0.:
                        if f"fsmoothness_2d.{i}" in output_elem and output_elem[f"fsmoothness_2d.{i}"]['data'] is not None:
                            fsmoothness_2d = output_elem[f"fsmoothness_2d.{i}"]['data']
                            wandb_dict.update(
                                {
                                    f'fsmoothness_2d{i}': plot_standard_heatmap(torch.tensor(fsmoothness_2d))
                                })
                            super_features_2d = output_elem[f"super_features_2d.{i}"]['data']
                            out_features_2d = output_elem[f"out_features_2d.{i}"]['data']
                            super_features_norm = lve.utils.plot_grid(super_features_2d[0], normalize=False)
                            wandb_dict.update({"super_features_norm_color": wandb.Image(super_features_norm)})
                            out_features_norm = lve.utils.plot_grid(out_features_2d[0], normalize=False)
                            wandb_dict.update({"out_features_norm_color": wandb.Image(out_features_norm)})
                            plt.close()

                wandb_dict.update(heatmaps)
                if self.output_stream.get_output_elements()['sup_prediction_idx'] is not None:
                    wandb_dict.update({'sup_pred_img': wandb.Image(
                        lve.utils.indices_to_rgb(self.output_stream.get_output_elements()['sup_prediction_idx']['data']) \
                            .reshape((self.input_stream.h, self.input_stream.w, 3)))})

                wandb.log(wandb_dict)
                plt.close('all')

        if self.__wandb and 'unsupervised_log' in self.options['worker']['net'] and stats_worker[
            'data'] is not None and len(stats_worker['data']) > 0:
            if stats_metrics['data'] is not None and len(stats_metrics['data']) > 0:
                pass
            else:
                wandb_dict = {
                    'loss': stats_worker['data']['loss'],
                    'consistency_lower': stats_worker['data']['consistency_lower'],
                    'consistency_upper': stats_worker['data']['consistency_upper'],

                    'similarity_loss': stats_worker['data']['similarity_loss'],
                    'dissimilarity_loss': stats_worker['data']['dissimilarity_loss'],
                    'simdissim_loss': stats_worker['data']['dissimilarity_loss'] + stats_worker['data'][
                        'similarity_loss'],

                    'consistency_absolute': stats_worker['data']['consistency_absolute'],
                    'consistency_relative': stats_worker['data']['consistency_relative'],

                    'motion_smoothness': stats_worker['data']['motion_smoothness'],
                    'frame': wandb.Image(self.output_stream.get_output_elements()['frames']['data'][..., ::-1]),
                }

                foax = stats_worker['data']['foax']
                foay = stats_worker['data']['foay']
                features = self.output_stream.get_output_elements()['features']['data']
                old_features = self.output_stream.get_output_elements()['old_features']['data']
                warped_features = self.output_stream.get_output_elements()['warped_features']['data']
                what_ref = features[0, :, int(foax), int(foay)]
                plt_heat = lve.utils.plot_what_heatmap_no_save(features, what_ref=what_ref, anchor=(foax, foay),
                                                               distance='cosine' if
                                                               self.options['worker']['net']['vision_block'][
                                                                   'features'][
                                                                   'normalize'] else 'euclidean')
                heatmaps = {'spatial_coherence_heatmap': wandb.Image(plt_heat)}
                # wandb_dict.update({"features": wandb.Image(
                #     torch.from_numpy(features[0]).unsqueeze(1))})  # selects only the first element from the batch!
                # print(features[0].shape)
                features_unnorm = lve.utils.make_grid_numpy(np.expand_dims(features[0], axis=1))
                wandb_dict.update({"features_unnorm": wandb.Image(features_unnorm)})
                plt.close()
                features_norm = lve.utils.plot_grid(features[0])
                wandb_dict.update({"features_norm_color": wandb.Image(features_norm)})
                plt.close()
                old_features_unnorm = lve.utils.make_grid_numpy(np.expand_dims(old_features[0], axis=1))
                wandb_dict.update({"old_features_unnorm": wandb.Image(old_features_unnorm)})
                plt.close()
                warped_features_unnorm = lve.utils.make_grid_numpy(np.expand_dims(warped_features[0], axis=1))
                wandb_dict.update({"warped_features_unnorm": wandb.Image(warped_features_unnorm)})
                plt.close()
                output_elem = self.output_stream.get_output_elements()
                fr = output_elem['frames']['data'][..., ::-1]
                wandb_dict.update({
                    'loss_window': output_elem["window_loss"]['data'],
                    'similarity_loss_window': output_elem["window_similarity_loss"]['data'],
                    'dissimilarity_loss_window': output_elem["window_dissimilarity_loss"]['data'],
                    'simdissim_loss_window': output_elem["window_dissimilarity_loss"]['data'] + output_elem["window_similarity_loss"]['data']
                })
                if self.worker.scheduler_features is not None:
                    wandb_dict.update({'scheduled_lr_features': self.worker.scheduler_features.get_last_lr()[0]})
                for i in range(0, self.options['worker']['net']['n_blocks']):
                    if output_elem[f"dfeat_x_2d.{i}"]['data'] is not None:
                        d_feat_x = output_elem[f"dfeat_x_2d.{i}"]['data']
                        d_feat_y = output_elem[f"dfeat_y_2d.{i}"]['data']

                        wandb_dict.update({f'dfeat_x_b{i}': plot_standard_heatmap(torch.tensor(d_feat_x)),
                                           f'dfeat_y_b{i}': plot_standard_heatmap(torch.tensor(d_feat_y))
                                           })
                    d_flow_x = output_elem[f"dflow_x_2d.{i}"]['data']
                    d_flow_y = output_elem[f"dflow_y_2d.{i}"]['data']
                    wandb_dict.update({f'dflow_x_b{i}': plot_standard_heatmap(torch.tensor(d_flow_x)),
                                       f'dflow_y_b{i}': plot_standard_heatmap(torch.tensor(d_flow_y)),
                                       })

                    if sum(self.options['worker']['net']['lambda_c_upper']) > 0.:
                        consistency_upper_2d = output_elem[f"consistency_upper_2d.{i}"]['data']
                        consistency_relative_2d = output_elem[f"consistency_relative_2d.{i}"]['data']
                        consistency_absolute_2d = output_elem[f"consistency_absolute_2d.{i}"]['data']
                        wandb_dict.update({f'consistency_upper_2d{i}': plot_standard_heatmap(consistency_upper_2d, underlay=fr)})
                        wandb_dict.update({f'consistency_relative_2d{i}': plot_standard_heatmap(consistency_relative_2d, underlay=fr)})
                        wandb_dict.update({f'consistency_absolute_2d{i}': plot_standard_heatmap(consistency_absolute_2d, underlay=fr)})

                    if self.options['worker']['net']['lambda_sim'] + self.options['worker']['net']['lambda_dis'] != 0:
                        if f"simdissim_points.{i}" in output_elem and output_elem[f"simdissim_points.{i}"]['data'] is not None:
                            simdissim_points = output_elem[f"simdissim_points.{i}"]['data']
                            wandb_dict.update({f'simdissim_points{i}': plot_sampled_points(simdissim_points, h=self.input_stream.h, w=self.input_stream.w, underlay=fr)})


                    if f"occl_2d_lower.{i}" in output_elem and output_elem[f"occl_2d_lower.{i}"]['data'] is not None:
                        occl_2d_lower = output_elem[f"occl_2d_lower.{i}"]['data']
                        wandb_dict.update({f'occl_2d_lower_b{i}': plot_standard_heatmap(torch.tensor(occl_2d_lower))})

                    if f"occl_2d_upper.{i}" in output_elem and output_elem[f"occl_2d_upper.{i}"]['data'] is not None:
                        occl_2d_upper = output_elem[f"occl_2d_upper.{i}"]['data']
                        wandb_dict.update({f'occl_2d_upper_b{i}': plot_standard_heatmap(torch.tensor(occl_2d_upper))})

                    motion = output_elem[f"net_motion.{i}"]['data']
                    motion_dic, rm_fnames_ = visualize_flows_no_save(motion, prefix=f'predicted_motion.{i}')
                    wandb_dict.update(motion_dic)
                wandb_dict.update(heatmaps)

                # get predicted flow and predicted frames

                for i in range(0, self.options['worker']['net']['n_blocks']):
                    motion = self.output_stream.get_output_elements()[f"net_motion.{i}"]['data']
                    motion_dic, rm_fnames_ = visualize_flows_no_save(motion, prefix=f'predicted_motion.{i}')
                    wandb_dict.update(motion_dic)


                n_blockss = self.options['worker']['net']['n_blocks']

                wandb_dict.update(
                    {f"consistency_lower_b{i}": stats_worker['data'][f"consistency_lower_b{i}"] for i in
                     range(n_blockss)})
                wandb_dict.update(
                    {f"consistency_upper_b{i}": stats_worker['data'][f"consistency_upper_b{i}"] for i in
                     range(n_blockss)})
                wandb_dict.update(
                    {f"consistency_upper_initial_b{i}": stats_worker['data'][f"consistency_upper_initial_b{i}"] for i in
                     range(n_blockss)})
                wandb_dict.update(
                    {f"consistency_absolute_b{i}": stats_worker['data'][f"consistency_absolute_b{i}"] for i in
                     range(n_blockss)})
                wandb_dict.update(
                    {f"consistency_relative_b{i}": stats_worker['data'][f"consistency_relative_b{i}"] for i in
                     range(n_blockss)})

                wandb.log(wandb_dict)
                plt.close('all')

        # saving metrics
        if 'metrics' in self.worker.options and self.worker.options['metrics'] is not None:
            self.metrics_container.save(self.model_folder)

        self.saved_at = steps

    # called by the web-server to handle requests from the visualization client
    def remote_change_option(self, name, value_str):
        print("Received worker option change request: " + str(name) + " -> " + str(value_str))
        names = name.split(".")
        opt = self.options["worker"]
        for n in names:
            if n not in opt:
                print("ERROR: Unknown option: " + name)
                return
            else:
                if n != names[-1]:
                    opt = opt[n]
                else:
                    current_value = opt[n]
        try:
            if isinstance(current_value, int):
                casted_value = int(value_str)
            elif isinstance(current_value, float):
                casted_value = float(value_str)
            elif isinstance(current_value, bool):
                casted_value = bool(value_str)
            elif isinstance(current_value, str):
                casted_value = str(value_str)
            elif isinstance(current_value, list):
                values_str_array = current_value.strip()[1:-1].split(",")
                if isinstance(current_value[0], int):
                    casted_value = [int(i) for i in values_str_array]
                elif isinstance(current_value[0], float):
                    casted_value = [float(i) for i in values_str_array]
                elif isinstance(current_value[0], str):
                    casted_value = [str(i) for i in values_str_array]
            else:
                print("ERROR: Skipping option change request due to unhandled type: " + name + " -> " + value_str)
                return False
        except ValueError:
            print("ERROR: Skipping option change request due not-matching type: " + name + " -> " + value_str)
            return False

        if self.__worker_options_to_change is None:
            self.__worker_options_to_change = {}
        self.__worker_options_to_change[name] = {"fields": names, "value": casted_value}
        return True

    # called by the web-server to handle requests from the visualization client
    def remote_command(self, command_name, command_value):
        print("Received command: " + command_name + " -> " + str(command_value))
        self.worker.send_command(command_name, command_value)

    # called by the web-server to handle requests from the visualization client
    def remote_allow_processing_next_frame_only(self):
        if self.output_stream.virtual_save:
            self.worker.set_heavy_output_data_needed(True)

        self.__event_visualization_is_happening.set()  # allow the system to process next frames...
        self.__event_visualization_is_happening.clear()  # ...and immediately block it again
        return self.output_stream.get_last_frame_number() + 1

    # called by the web-server to handle requests from the visualization client
    def remote_allow_processing(self):
        if self.output_stream.virtual_save:
            self.worker.set_heavy_output_data_needed(False)

        self.__event_visualization_is_happening.set()  # always allow to process next frames

    # called by the web-server to handle requests from the visualization client
    def remote_is_processing_allowed(self):
        return self.__event_visualization_is_happening.is_set()

    # called by the web-server to handle requests from the visualization client
    def remote_disable_processing_asap(self):
        if self.output_stream.virtual_save:
            self.worker.set_heavy_output_data_needed(True)

        self.__event_visualization_is_happening.clear()  # block attempts to process next frame

    # called by the web-server to handle requests from the visualization client
    def remote_get_data_to_visualize(self, data_identifier):
        self.__event_visualization_is_happening.clear()  # block attempts to process next frame
        self.__event_processing_is_running.wait()  # wait until processing of the current frame has ended
        try:
            return self.output_stream.get_output_elements()[data_identifier]["data"]
        except KeyError:
            return None

    def __handle_hot_option_changes(self):
        something_changed = False
        if self.__worker_options_to_change is not None:
            for name in self.__worker_options_to_change:
                opt_w = self.worker.options
                names = self.__worker_options_to_change[name]["fields"]
                for n in names:
                    if n != names[-1]:
                        opt_w = opt_w[n]
                    else:

                        # patching the case of boolean
                        if isinstance(opt_w[n], bool):
                            if self.__worker_options_to_change[name]["value"] != 0:
                                self.__worker_options_to_change[name]["value"] = True
                            else:
                                self.__worker_options_to_change[name]["value"] = False

                        if opt_w[n] != self.__worker_options_to_change[name]["value"]:
                            opt_w[n] = self.__worker_options_to_change[name]["value"]
                            something_changed = True

            if something_changed:
                self.__save_options()
        self.__worker_options_to_change = None

    def __save_options(self):

        # filtering options: keeping only key that do not start with "_"
        options_filtered = {}
        queue = [self.options]
        queue_f = [options_filtered]

        while len(queue) > 0:
            opt = queue.pop()
            opt_f = queue_f.pop()
            if isinstance(opt, dict):
                for k in opt:
                    if k[0] != '_':
                        if isinstance(opt[k], dict):
                            opt_f[k] = {}
                            queue.append(opt[k])
                            queue_f.append(opt_f[k])
                        else:
                            opt_f[k] = opt[k]  # copying

        lve.utils.save_json(self.model_folder + os.sep + "options.json", options_filtered)
