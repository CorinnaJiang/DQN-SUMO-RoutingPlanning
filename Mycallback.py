from typing import Dict
import argparse
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from MainEnv import SumoRouteEnv

class MyCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: SumoRouteEnv ,
                         policies: Dict[str, Policy], episode: Episode,
                         **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!"
        print("episode {} started.".format(
            episode.episode_id))
        episode.user_data["traveling_time"] = []
        episode.hist_data["traveling_time"] = []

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        policies: Dict[str, Policy] = None,
                        episode: Episode,
                        **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        traveling_time = abs(episode.last_observation_for()[2])
        # raw_angle = abs(episode.last_raw_obs_for()[2])
        # assert pole_angle == raw_angle
        episode.user_data["traveling_time"].append(traveling_time)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: SumoRouteEnv ,
                         policies: Dict[str, Policy], episode: Episode,
                         **kwargs):
        # Make sure this episode is really done.
        assert episode.batch_builder.policy_collectors[
            "default_policy"].batches[-1]["dones"][-1], \
            "ERROR: `on_episode_end()` should only be called " \
            "after episode is done!"
        traveling_time = np.mean(episode.user_data["traveling_time"])
        print("episode {}  ended with length {} and pole "
              "angles {}".format(episode.episode_id,episode.length,
                                 traveling_time))
        episode.custom_metrics["traveling_time"] = traveling_time
        episode.hist_data["traveling_time"] = episode.user_data["traveling_time"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(
            trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
        print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
            policy, result["sum_actions_in_train_batch"]))

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: Episode, agent_id: str,
            policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
