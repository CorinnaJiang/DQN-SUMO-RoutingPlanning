from copy import deepcopy
import logging
import pathlib
from pprint import pformat
import random
import sys
import traceback

import ray

from ray.rllib.agents.dqn import dqn
from ray.tune.logger import pretty_print
from base import MyEnv

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger('a3ctrain')

def _main():
    """ Training example """

    # Initialize RAY.
    ray.tune.registry.register_env('test_env', MyEnv)
    ray.init()

    # Algorithm.
    policy_class = dqn.DQNTFPolicy

    # https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/trainer.py#L44
    # https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/a3c/a3c.py#L14
    policy_conf = dqn.DEFAULT_CONFIG
    policy_conf['double_q'] = 'complete_episodes'
    policy_conf['double_q']= True
    policy_conf['dueling'] = True
    policy_conf['num_atoms'] = 1
    policy_conf['noisy'] = False
    policy_conf['prioritized_replay'] =False
    policy_conf['n_step'] = 1
    policy_conf['target_network_update_freq']=8000
    policy_conf['lr'] = .0000625
    policy_conf['adam_epsilon'] =.00015
    policy_conf['hiddens'] = [512]
    policy_conf['learning_starts'] = 20000
    policy_conf['buffer_size'] = 1000000
    policy_conf['rollout_fragment_length']= 4
    policy_conf['train_batch_size'] =  32
    # policy_conf['exploration_config']:
    policy_conf['epsilon_timesteps'] = 200000
    policy_conf['final_epsilon'] =  0.01
    policy_conf['prioritized_replay_alpha'] = 0.5
    policy_conf['final_prioritized_replay_beta'] =1.0
    policy_conf['prioritized_replay_beta_annealing_timesteps'] =2000000
    policy_conf['num_gpus'] = 0.2
    policy_conf['timesteps_per_iteration'] = 10000

    scenario_config = deepcopy(MyEnv)
    # scenario_config['seed'] = 42
    scenario_config['log_level'] = 'INFO'
    scenario_config['sumo_config']['sumo_connector'] = 'traci'
    scenario_config['sumo_config']['trace_file'] = True
    scenario_config['sumo_config']['sumo_gui'] = True
    scenario_config['sumo_config']['sumo_cfg'] = '{}/data/test.sumocfg'.format(
        pathlib.Path(__file__).parent.absolute())
    scenario_config['sumo_config']['sumo_params'] = ['--collision.action', 'warn']
    scenario_config['sumo_config']['end_of_sim'] = 3600 # [s]
    scenario_config['sumo_config']['update_freq'] = 10 # number of traci.simulationStep()
                                                       # for each environment step.
    scenario_config['sumo_config']['log_level'] = 'INFO'
    logger.info('Scenario Configuration: \n %s', pformat(scenario_config))

    logger.info('a3cA3C Configuration: \n %s', pformat(policy_conf))
    trainer = dqn.Trainer(env='test_env',
                             config=policy_conf)

    # Single training iteration, just for testing.
    try:
        result = trainer.train()
        print('Results: \n {}'.format(pretty_print(result)))
    except Exception:
        EXC_TYPE, EXC_VALUE, EXC_TRACEBACK = sys.exc_info()
        traceback.print_exception(EXC_TYPE, EXC_VALUE, EXC_TRACEBACK, file=sys.stdout)
    finally:
        ray.shutdown()

if __name__ == '__main__':
    _main()
