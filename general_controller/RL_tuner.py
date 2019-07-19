from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import nni
from src.utils import Logger
from nni.tuner import Tuner
from src.general_controller import GeneralController
from src.tf_flags import *
from collections import OrderedDict


def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_name+'.log')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


logger = build_logger("RL_controller")


def build_controller(ControllerClass):
    controller_model = ControllerClass(
        search_for=FLAGS.search_for,
        search_whole_channels=FLAGS.controller_search_whole_channels,
        skip_target=FLAGS.controller_skip_target,
        skip_weight=FLAGS.controller_skip_weight,
        num_cells=FLAGS.child_num_cells,
        num_layers=FLAGS.child_num_layers,
        num_branches=FLAGS.child_num_branches,
        out_filters=FLAGS.child_out_filters,
        lstm_size=64,
        lstm_num_layers=1,
        lstm_keep_prob=1.0,
        tanh_constant=FLAGS.controller_tanh_constant,
        op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
        temperature=FLAGS.controller_temperature,
        lr_init=FLAGS.controller_lr,
        lr_dec_start=0,
        lr_dec_every=1000000,  # never decrease learning rate
        l2_reg=FLAGS.controller_l2_reg,
        entropy_weight=FLAGS.controller_entropy_weight,
        bl_dec=FLAGS.controller_bl_dec,
        use_critic=FLAGS.controller_use_critic,
        optim_algo="adam",
        sync_replicas=FLAGS.controller_sync_replicas,
        num_aggregate=FLAGS.controller_num_aggregate,
        num_replicas=FLAGS.controller_num_replicas)

    return controller_model


def get_controller_ops(controller_model):
    """
    Args:
      images: dict with keys {"train", "valid", "test"}.
      labels: dict with keys {"train", "valid", "test"}.
    """

    controller_ops = {
        "train_step": controller_model.train_step,
        "loss": controller_model.loss,
        "train_op": controller_model.train_op,
        "lr": controller_model.lr,
        "grad_norm": controller_model.grad_norm,
        "valid_acc": controller_model.valid_acc,
        "optimizer": controller_model.optimizer,
        "baseline": controller_model.baseline,
        "entropy": controller_model.sample_entropy,
        "sample_arc": controller_model.sample_arc,
        "skip_rate": controller_model.skip_rate,
    }

    return controller_ops


class RLTuner(Tuner):

    def __init__(self, child_steps, controller_steps):

        self.child_steps = child_steps
        self.controller_steps = controller_steps
        self.controller_model = build_controller(GeneralController)

        self.graph = tf.Graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(
            allow_soft_placement=True, gpu_options=gpu_options)

        self.controller_model.build_trainer()
        self.controller_ops = get_controller_ops(self.controller_model)

        hooks = []
        if FLAGS.controller_training and FLAGS.controller_sync_replicas:
            sync_replicas_hook = self.controller_ops["optimizer"].make_session_run_hook(
                True)
            hooks.append(sync_replicas_hook)

        self.sess = tf.train.SingularMonitoredSession(
            config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir)
        logger.debug('initlize controller_model done.')

        self.epoch = 0
        self.pos = 0

    def generate_parameters(self, parameter_id, trial_job_id=None):
        current_arc_code = self.sess.run(self.controller_model.sample_arc)
        start_idx = 0
        current_config = {
            self.choice_key: 'train' if self.pos < self.child_steps else 'validate'
        }
        for layer_id, (layer_name, info) in enumerate(self.search_space):
            mutable_block = info['mutable_block']
            if mutable_block not in current_config:
                current_config[mutable_block] = dict()
            layer_choice_idx = current_arc_code[start_idx]
            if layer_id != 0:
                input_start = start_idx + 1
            else:
                input_start = start_idx
            inputs_idxs = current_arc_code[input_start: input_start + layer_id]
            inputs_idxs = [idx for idx, val in enumerate(inputs_idxs) if val == 1]
            current_config[mutable_block][layer_name] = dict()
            current_config[mutable_block][layer_name]['chosen_layer'] = info['layer_choice'][layer_choice_idx]
            current_config[mutable_block][layer_name]['chosen_inputs'] = [
                info['optional_inputs'][ipi] for ipi in inputs_idxs]
            start_idx += 1 + layer_id
        # update pos and epoch
        self.pos = (self.pos + 1) % (self.child_steps + self.controller_steps)
        self.epoch += int(self.pos == 0)

        return current_config

    def controller_one_step(self, epoch, valid_acc_arr):
        logger.debug("Epoch %s: Training controller", epoch)
        run_ops = [
            self.controller_ops["loss"],
            self.controller_ops["entropy"],
            self.controller_ops["lr"],
            self.controller_ops["grad_norm"],
            self.controller_ops["valid_acc"],
            self.controller_ops["baseline"],
            self.controller_ops["skip_rate"],
            self.controller_ops["train_op"],
        ]

        loss, entropy, lr, gn, val_acc, bl, _, _ = self.sess.run(run_ops, feed_dict={
            self.controller_model.valid_acc: valid_acc_arr})

        controller_step = self.sess.run(self.controller_ops["train_step"])

        log_string = ""
        log_string += "ctrl_step={:<6d}".format(controller_step)
        log_string += " loss={:<7.3f}".format(loss)
        log_string += " ent={:<5.2f}".format(entropy)
        log_string += " lr={:<6.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(gn)
        log_string += " acc={:<6.4f}".format(val_acc)
        log_string += " bl={:<5.2f}".format(bl)
        log_string += " child acc={:<5.2f}".format(valid_acc_arr)
        logger.debug(log_string)
        return

    def receive_trial_result(self, parameter_id, parameters, reward, trial_job_id):
        logger.debug("epoch:\t"+str(self.epoch))
        logger.debug("pos:\t"+str(self.pos))
        logger.debug(parameter_id)
        logger.debug(reward)
        if self.pos > self.child_steps:
            self.controller_one_step(self.epoch, reward)

    def update_search_space(self, data):
        # Extract choice
        choice_key = list(
            filter(lambda k: k.strip().endswith('choice'), list(data)))
        if len(choice_key) > 0:
            self.choice_key = choice_key[0]
            data.pop(choice_key[0])
        # Sort layers and generate search space
        self.search_space = []
        data = OrderedDict(
            sorted(data.items(), key=lambda tp: int(tp[0].split('_')[-1])))
        for block_id, layers in data.items():
            data[block_id] = OrderedDict(
                sorted(layers.items(), key=lambda tp: int(tp[0].split('_')[-1])))
            for layer_id, info in data[block_id].items():
                info['mutable_block'] = block_id
                self.search_space.append((layer_id, info))
        logger.debug(self.search_space)


if __name__ == "__main__":
    tf.app.run()
