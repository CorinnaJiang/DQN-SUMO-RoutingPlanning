from copy import deepcopy
import logging
import pathlib
from pprint import pformat
import random
import sys
import traceback

import gym
import traci
from ray.tune.registry import register_env
import ray
import os

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.tune.logger import pretty_print
from MainEnv import SumoRouteEnv
import tensorflow as tf
import shutil
from ray import tune
from Mycallback import MyCallback




class Rltrain:

    def __init__(self, mode):
        self.mode = mode
        # self.run = 0
        # self.veh_id = "vehicle_0"
    def _main(self):
        dir = '/Users/corinnajiang/Desktop/Desktop – Corinna’s MacBook Pro/Master-disseration/Simulation/project code/data/net2/'
        print('------------main-----------------')
        logging.basicConfig(level = logging.WARN)
        logger = logging.getLogger('DQN')
        """ Training example """
        # init directory in which to save checkpoints
        chkpt_root = "/log/"
        shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
        print('------------logger setting fin-----------------')
        # init directory in which to log results
        ray_results = "/log/ray_result/".format(os.getenv("HOME"))
        shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
        # start Ray -- add `local_mode=True` here for debugging
        print('------------tune-----------------')
        # traci.close()
        config = {
            'env' : "test_env",
            # 'callbacks':{
            #     'on_train_result' : tune.function(on_train_result),
            # },
            'callbacks': MyCallback,
            'env_config' : {
            'dir' : dir,
            'net_file': dir + 'osm.net.xml',
            'route_file': dir + 'test3.rou.xml',
            'sim_file': dir + "osm.sumocfg",
            'veh_id': 'vehicle_0',
            'Start_edge': "632695808", #'gneE13',
            'Destination_edge': "633554833#0", #'gneE19',
            'use_gui': False,
            'out_csv_name': None,
            'delta_time': 1,
            'run': 0,
        },
            'num_gpus': 0,
            'framework': 'tf',
            'double_q': True,
            'dueling': True,
            'num_atoms': 1,
            'noisy': False,
            'prioritized_replay': False,
            'n_step': 1,
            'target_network_update_freq': 8000,
            'lr': .0000625,
            'adam_epsilon': .00015,
            'hiddens': [512],
            'learning_starts': 20000,
            # 'buffer_size': 1000000,
            'rollout_fragment_length': 4,
            'train_batch_size': 32,
            'exploration_config':{
            'epsilon_timesteps': 200000,
            'final_epsilon': 0.01,
            },
            'replay_buffer_config':{
                'capacity' :1000000,
            },
            'prioritized_replay_alpha': 0.5,
            'final_prioritized_replay_beta': 1.0,
            'prioritized_replay_beta_annealing_timesteps': 2000000,
            'num_gpus': 0,
            "num_workers":0,
            'timesteps_per_iteration': 10000,
            'eager_tracing' : True,
        }
        # ray.init(local_mode=True)
        # ray.init()
        ray.init(ignore_reinit_error=True)
        ray.tune.registry.register_env("test_env", lambda config: SumoRouteEnv(config))
        # env = gym.make("test_env")

        print('------------initial starting-----------------')

        if self.mode == 'test':
            path = '/Users/corinnajiang/ray_results/DQN_test_env_2022-01-21_15-25-38zxkhoyk_/checkpoint_000002/checkpoint-2'
            env = SumoRouteEnv(config['env_config'])
            agent = self.load(config,path)
            self.test(env)
        else:
            stop = {
                # "training_iteration": 100,
                "timesteps_total": 100,
                # "episode_reward_mean": 1000,
                # 'DONE' : True
            }
            print('------------tune running-----------------')
        # # traci.close()
        # #

        # resources = PGTrainer.default_resource_request(config).to_json()
            results = ray.tune.run(
                # trainer,
                # 'DQN',
                # 'PG',
                self.my_train,
                config=config,
                stop=stop,
                # verbose=2,
                # checkpoint_freq=1,
                # resources_per_trial=resources,
                # num_samples=10,

                # restore=args.from_checkpoint
            )
            print(results)

        ray.shutdown()
        # And check the results.
        # if args.as_test:
        #     check_learning_achieved(results, args.stop_reward)

        # try:
        #     print('start the train')
        #     result = trainer.train()
        #     print('Results: \n {}'.format(pretty_print(result)))
        # except Exception:
        #     EXC_TYPE, EXC_VALUE, EXC_TRACEBACK = sys.exc_info()
        #     traceback.print_exception(EXC_TYPE, EXC_VALUE, EXC_TRACEBACK, file=sys.stdout)
        # finally:
        #     ray.shutdown()


        # for i in range(1000):
        #
        #     result = trainer.train()
        #     print('check the epoach')
        #     print('result = {}'.format(result))
        #
        #     if i % 10 == 0:
        #         checkpoint = trainer.save()
        #         print('checkpoint saved at', checkpoint)
    #
    # if __name__ == '__main__':
    #     num = tf.config.experimental.list_physical_devices()
    #     print('gpu devive :',num)
    #     _main()
    #     status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    #     n_iter = 3
    #     for n in range(n_iter):
    #         print('start the train')
    #         result = agent.train()
    #         # chkpt_file = agent.save(chkpt_root)
    #         print(result)
            # print(status.format(
            #         n + 1,
            #         result["episode_reward_min"],
            #         result["episode_reward_mean"],
            #         result["episode_reward_max"],
            #         result["episode_len_mean"],
            #         # chkpt_file
            #         ))


            # examine the trained policy
        # policy = agent.get_policy()
        # model = policy.model
        # print(model.base_model.summary())


            # apply the trained policy in a rollout
            # agent.restore(chkpt_file)
            # env = gym.make(select_env)
            #
            # state = env.reset()
            # sum_reward = 0
            # n_step = 20
            #
            # for step in range(n_step):
            #     action = agent.compute_action(state)
            #     state, reward, done, info = env.step(action)
            #     sum_reward += reward
            #
            #     env.render()
            #
            #     if done == 1:
            #         # report at the end of each episode
            #         print("cumulative reward", sum_reward)
            #         state = env.reset()
            #         sum_reward = 0

    def my_train(self, config, reporter):
        agent = DQNTrainer(config=config,env='test_env')
        # agent = PGTrainer(config=config, env='test_env')
        for _ in range(10):
            result = agent.train()
            result['phase'] = 1
            reporter(**result)
            phase_time = result['timesteps_total']
        state = agent.save()
        agent.stop()

    # def on_train_result(info):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         info["trainer"].__name__, info["result"]["episodes_this_iter"]))
    def load(self,config, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = DQNTrainer(config=config,env='test_env')
        return self.agent.restore(path)

    def test(self,  env):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        # run until episode ends
        traci.close()
        episode_reward = 0
        done = False
        obs = env.reset()
        print('obs', obs)
        while not done:
            action = self.agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        return episode_reward


if __name__ == "__main__":
    num = tf.config.experimental.list_physical_devices()
    print('gpu devive :',num)
    path = '/Users/corinnajiang/ray_results/DQN_test_env_2022-01-21_15-25-38zxkhoyk_/checkpoint_000002/checkpoint-2'
    DQNTrain=Rltrain(mode='te')
    # DQNTrain.load(path)
    # DQNTrain.test()
    DQNTrain._main()