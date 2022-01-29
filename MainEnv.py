import os
import sys
import pandas as pd
from ray.rllib.env.env_context import EnvContext
from typing import Optional, Union, Tuple
from pathlib import Path

import vehicle

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
    import traci.constants as tc
else:
    raise Exception("Please declare environment variable 'SUMO_HOME'")
import traci
import gym
from gym.envs.registration import EnvSpec
from gym.spaces import space
import time
import subprocess

from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from vehicle import VehicleEnv
import numpy as np
# import traci
# import sumolib as traci

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from Network import NetworkEnv
from gym.spaces import Tuple as gymTuple
LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ
# def env(**kwargs):
#     env = SumoRouteEnv(**kwargs)
#     env = wrappers.AssertOutOfBoundsWrapper(env)
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env
#
#
# parallel_env = parallel_wrapper_fn(env)


class SumoRouteEnv(gym.Env, VehicleEnv, NetworkEnv):
    CONNECTION_LABEL = 0  # For traci multi-client support
    # net_file = 'data/test.net.xml',
    # route_file = 'data/test.rou.xml',
    # sim_file = "data/test.sumocfg",
    # veh_id = 'vehicle_0',
    # Start_edge = 'gneE13',
    # Destination_edge = 'gneE19',
    # use_gui = True
    timesrecord = 0
    def __init__(
            self,
            config: EnvContext
            # net_file: str ='data/test.net.xml',
            # route_file: str ='data/test.rou.xml',
            # sim_file: str = "data/test.sumocfg",
            # veh_id= 'vehicle_0',
            # Start_edge = 'gneE13',
            # Destination_edge = 'gneE19',
            # use_gui: bool = False,
            # out_csv_name: Optional[str] = None,
            # virtual_display: Optional[Tuple[int, int]] = None,
            # begin_time: int = 0,
            # num_seconds: int = 2000000,
            # max_depart_delay: int = 100000,
            # time_to_teleport: int = -1,
            # delta_time: int = 5,
            # single_agent: bool = False,
            # sumo_seed: Union[str, int] = 'random',
            # fixed_ts: bool = False,
            # sumo_warnings: bool = True,
    ):
        # super(SumoRouteEnv, self).__init__()
        self._net =config ['net_file']
        self._route = config['route_file']
        self.sim_file = config['sim_file']
        self.use_gui = config['use_gui']
        self.veh_id = config['veh_id']
        self.dest_edge =config['Destination_edge']
        self.Start_edge = config['Start_edge']
        self.delta_time = config['delta_time']
        self.edge_choice = config['Start_edge']
        self.run = config['run']
        self.dir = config['dir']
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        # self.virtual_display = virtual_display



        # # self.begin_time = begin_time
        self.sim_max_time = 500
         # seconds on sumo at each step
        # self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        # self.time_to_teleport = time_to_teleport
        # self.single_agent = single_agent
        # self.sumo_seed = sumo_seed
        # self.fixed_ts = fixed_ts
        # self.sumo_warnings = sumo_warnings
        # self.label = str(SumoRouteEnv.CONNECTION_LABEL)
        # SumoRouteEnv.CONNECTION_LABEL += 1
        self.sumo = None
        self.edge_choice_keeper = None
        self.curr_routelist_keeper = None
        self.next_obers_keeper = None
        self.reward_keeper =None
        self.update = None
        self.reward = None
        self.done = None
        self.sumo = None
        self.actionList_keeper = None
        self.label = str(SumoRouteEnv.CONNECTION_LABEL)
        # self.routlist
        print('------initia traci -------')
        # traci.init(port=8813, host="localhost")
        VehicleEnv.__init__(self, self.veh_id, self.Start_edge, self.dest_edge, list(),self._net)
        # if LIBSUMO:
        #     print('---libsumo-----')
        # if SumoRouteEnv.CONNECTION_LABEL != 0:
        #     print('connection label',SumoRouteEnv.CONNECTION_LABEL)
        #     traci.close()
        #     sys.stdout.flush()
        #     self._sumo_binary = sumolib.checkBinary('sumo')

            # self.initSimulator(False, 8813)
            # time.sleep(10)
            # traci.init(port=5666, host="127.0.0.1")
        print('getcwd:      ', os.getcwd())
        print('__file__:    ', __file__)
        traci.start([sumolib.checkBinary('sumo'),  "-c", self.sim_file],numRetries=1)  # Start only to retrieve traffic light information
        print('---initi traci finished ----')
        #     conn = traci
        # else:
        #     print('---initi connec-----')
        #     # traci.close()
        #     traci.start([sumolib.checkBinary('sumo'), "-c", self.sim_file, "--tripinfo-output", "tripinfo.xml"], label='init_connection'+self.label)
        #     conn = traci.getConnection('init_connection'+self.label)
        # time.sleep(10)
        # self._start_simulation()
        # traci.init(port=8813,host='localhost',doSwitch=True,label='default')
        # traci.init()
        # conn.start([sumolib.checkBinary('sumo'), "-c", self.sim_file,
        #              "--tripinfo-output", "tripinfo.xml"],port=8813)
        step = 0
        print('-------initial vehiv par-----')
        self._sumo_step()
        step +=1
        self.actionList = VehicleEnv.generate_routeList(self, veh_id=self.veh_id)
        traci.close()
        # self.actionList = VehicleEnv.generate_routeList(self, veh_id=self.veh_id)
        print('action',self.actionList)
        # traci.close()

        self.vehicles = dict()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = EnvSpec('SUMORL-v0')
        self.metrics = []
        self.out_csv_name = config['out_csv_name']
        self.state = None
        # self.observations = {ts: None for ts in self.ts_ids}
        # self.rewards = {ts: None for ts in self.ts_ids}
        print('------------reset veh-----------------')
        VehicleEnv.__init__(self, self.veh_id, self.Start_edge, self.dest_edge,list(),self._net)

        print('initialVehcileEv successfully')

    def _start_simulation(self):
        print('------------_start_simulation-----------------')
        # traci.start([self._sumo_binary, "-c", self.sim_file,
        #              "--tripinfo-output", "tripinfo.xml"])
        # step = 0
        # traci.simulationStep()
        # step = 0
        # while step < 1000:
        #     traci.simulationStep()
        #     step += 1
        # traci.close()
        SumoRouteEnv.CONNECTION_LABEL += 1
        sumo_cmd = [self._sumo_binary,
                    '-n', self._net,
                    '-r', self._route,
                    '-c', self.sim_file,]
                    # '--max-depart-delay', str(self.max_depart_delay),
                    # '--waiting-time-memory', '10000',
                    # '--time-to-teleport', str(self.time_to_teleport)]
        # if self.begin_time > 0:
        #     sumo_cmd.append('-b {}'.format(self.begin_time))
        # if self.sumo_seed == 'random':
        #     sumo_cmd.append('--random')
        # else:
        #     sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        # if not self.sumo_warnings:
        #     sumo_cmd.append('--no-warnings')
        # if self.use_gui:
        #     sumo_cmd.extend(['--start', '--quit-on-end'])
        #     if self.virtual_display is not None:
        #         sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
        #         from pyvirtualdisplay.smartdisplay import SmartDisplay
        #         print("Creating a virtual display.")
        #         self.disp = SmartDisplay(size=self.virtual_display)
        #         self.disp.start()
        #         print("Virtual display started.")

        # if LIBSUMO:
        #     traci.start(sumo_cmd)
        #     self.sumo = traci
        # else:
        #     traci.start(sumo_cmd, label=self.label)
        #     self.sumo = traci.getConnection(self.label)
        traci.start(sumo_cmd)
        self.sumo = traci
        print("start the simulation")

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self):
        print('------------reset-----------------')
        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        VehicleEnv.__init__(self, self.veh_id, self.Start_edge, self.dest_edge, list(), self._net)
        self._start_simulation()
        self._sumo_step()

        states = self.computer_observation(self.edge_choice)
        self.edge_choice_keeper = None
        self.curr_routelist_keeper = None
        self.next_obers_keeper = None
        self.reward_keeper =None
        self.actionList_keeper = None


        # collect information of the state of the network based on the
        # environment class used
        # self.state = np.asarray(states).T

        # observation associated with the reset (no warm-up steps)
        # observation = np.(states)
        # print('observationcopy', type(observation))

        # print('observationcopy', np.dtype(observation[0]))
        # print('observationcopy', observation.shape)
        # print('observationcopy', observation)


        return states

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    def step(self, rl_actions):
        '''
        whether to take action or not take action according to the
        :param rl_actions:
        :return:
        '''
        print('step begining:', rl_actions)
        # if self.actionList is None:
        #     VehicleEnv.__init__(self, self.veh_id, self.Start_edge, self.dest_edge, list())
        if rl_actions is None or rl_actions == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._sumo_step()
            print(' self.edge_choice_keeper', self.edge_choice_keeper)
            # if vehicle.VehicleEnv.arrive_decision_zone(self, self.veh_id):
            edge_choice = self._apply_rl_actions(rl_actions)
            # check rl action is vaild
            # if edge_choice == False:
            #     print('invaild action')
            #     return
        # states = self.get_state()
            cur_speed = VehicleEnv.get_veh_speed(self,self.veh_id)
            print('cur_speed',cur_speed)

            if self.update and cur_speed !=0:
                next_observation = self.computer_observation(edge_choice)
                self.next_obers_keeper = next_observation
                # self.states = np.asarray(states).T
                # print('state', states)
                # print('type -- state', type(states))
                #
                # next_observation = np.copy(states)
                # next_observation = tuple(states)
                print('next_observation', next_observation)
                print('next_observation type ', type(next_observation))
                reward = self.compute_reward(edge_choice)
                self.reward_keeper = reward
            else:
                next_observation = self.next_obers_keeper
                reward = self.reward_keeper
                if edge_choice =='NoneConnection':
                    reward = -10000000
            wait_time = VehicleEnv.get_wait_time(self,self.veh_id)
            done = (edge_choice == self.dest_edge) or self.sim_step > self.sim_max_time or wait_time >=100 or (edge_choice == 'NoneConnection')#or (edge_choice==False)

            infos = {}
            print('done or not ,',done)
            # else:
            #     next_observation = self.computer_observation(self.edge_choice_keeper)
            #     done = (self.edge_choice_keeper == self.dest_edge)
            #     reward = self.compute_reward(self.edge_choice_keeper)
            #     self._sumo_step()
                # all = np.array(out)

            # reward = self.compute_reward(edge_choice, fail=crash)
        self.reward = reward
        self.done = done
        return next_observation, reward, done, infos
        # elif edge_choice == self.Arrived:
        #     return False

    @property
    def action_space(self):
        print('check action space done', self.actionList)
        # if self.actionList is None:
        #     return Discrete(1)
        # print('action_space',action_space)
        return Discrete(6)

    @property
    def observation_space(self):
        '''
        After applying action, the return valued
        :return:
        '''
        observe_space = Box(
                        low=-10000,
                        high=10000,
                        shape=(3,),
                        dtype=np.float64)
        print('observe_space',observe_space)
        return observe_space
        #
        # vehicle_number= Box(
        #             low=-np.inf,
        #             high=np.inf,
        #             shape=(1,),
        #             dtype=np.float32)
        # vehicle_position= Box(
        #             low=-np.inf,
        #             high=np.inf,
        #             shape=(1,),
        #             dtype=np.float32)
        # destination_pos= Box(
        #             low=-np.inf,
        #             high=np.inf,
        #             shape=(2,),
        #             dtype=np.float32)
        #
        # vehicle_2D_positions= Box(
        #             low=-np.inf,
        #             high=np.inf,
        #             shape=(2,),
        #             dtype=np.float32)
        #
        # print('check observation space done')
        # observ = gymTuple([vehicle_number,
        #           vehicle_position,
        #           destination_pos,
        #           vehicle_2D_positions])
        # print('type of observ ',type(observ))
        # print('tuple', observ)
        # return observ

    def get_state(self):
        '''
        state1 : the numbers of vehicle in the edges
        state2 : the average driving speeds int the edges
        state3 : vehicle position
        state4 : traveling time
        :return:
        '''
        # choose_edge = rl_actions
        vehicle_number = VehicleEnv.vehicle_number_inedge(self, self.veh_id)
        # mean_speed = traci.edge.getLastStepMeanSpeed(choose_edge)
        vehicle_position = VehicleEnv.get_vehicle_position(self, self.veh_id)
        # average_travel_time = VehicleEnv.get_travel_time(self, choose_edge)
        destination_pos = NetworkEnv.get_edge_2D_position(self, self.dest_edge, self._net)[1]
        vehicle_2D_position = VehicleEnv.get_veh_2D_position(self, self.veh_id)
        total_traveling_time = self.sumo.simulation.getTime()
        state = [vehicle_number,
                 vehicle_position,
                 destination_pos,
                 vehicle_2D_position,
                 total_traveling_time]
        states = np.asarray(state, dtype=object)
        print('state:----', states)
        print('state:----', type(states))
        return states

    def _apply_rl_actions(self, rl_actions):
        '''
        After appling an action, how the state will change
        :param rl_actions: the actinon (choose_edgeID) of RL vehicle
        :return: the change of the state-- traveling time which can be used
        in reward calcaulation
        '''
        choose_route_index = rl_actions
        print('time to take action:',choose_route_index)

        # VehicleEnv.unsubscribe_vehicle(self, self.veh_id
        self.actionList = VehicleEnv.generate_routeList(self, veh_id=self.veh_id)
        # self.action_space.n = len(self.actionList)-1
        if self.actionList is not False and self.actionList !='Intersection':
            if choose_route_index > len(self.actionList)-1:
                # self.update = True
                choose_route_index = len(self.actionList)-1
        # check the validaition of the action
            print('self action list', self.actionList)

        edge_choice = VehicleEnv.choose_rl_routes(self, self.veh_id, choose_route_index)
        # if edge_choice in self.actionList and self.edge_choice_keeper in self.actionList:
        #     edge_check = False
        # else:
        #     edge_check = True
        # print('1 st edge_choice', edge_choice)
        # if len(self.actionList) !=0 or self.actionList is not None or self.actionList is not False or self.actionList !='Intersection' or self.edge_choice_keeper is not None or edge_choice is not None:
        #     edge_check = (edge_choice not in self.actionList) and (self.edge_choice_keeper in self.actionList)
        # else: edge_check = False

        if edge_choice is not None and edge_choice !='NoneConnection':
            self.update = True
            self.edge_choice_keeper = edge_choice
            self.actionList_keeper = self.actionList
            print('2rd edge choice',edge_choice)

            cur_routelist = VehicleEnv.assign_rl_route(self, self.veh_id, edge_choice)
            # if cur_routelist is False:
            #     print('something went wrong')
            #     self.reset()

            self.curr_routelist_keeper = cur_routelist
            print('2rd  cur_routelist', cur_routelist)
            # if VehicleEnv.arrive_excution_zone(self,self.veh_id):
            VehicleEnv.set_vehicle_route(self, self.veh_id, cur_routelist)
            # else:
            #     self.update = False
            #     return
            # while VehicleEnv.arrive_excution_zone(self,self.veh_id):
            #     VehicleEnv.set_vehicle_route(self, self.veh_id, cur_routelist)
            # if edge_choice in cur_routelist:
            #     return self.reset()
    # if edge_choice == self.Arrived:
    #     print('arrive the destionat')
    #     VehicleEnv.unsubscribe_vehicle(self, self.veh_id)
        elif edge_choice == 'NoneConnection':
            self.update = False
            return 'NoneConnection'
        else:
            self.update = False
            edge_choice =self.edge_choice_keeper
            VehicleEnv.set_vehicle_route(self, self.veh_id, self.curr_routelist_keeper )
            self.actionList = None
            print('self action list' , self.actionList)
        print('edge_choice------', edge_choice)
        print('self.curr_routelist_keeper',self.curr_routelist_keeper)
        crash = NetworkEnv.check_collision(self)
        self.edge_choice = edge_choice
        return edge_choice


    def computer_observation(self, edge_choice):

        edge_len = VehicleEnv.edge_length(self, edge_choice)
        edge_avg_speed = VehicleEnv.get_edge_speed(self, edge_choice)
        avg_speed = traci.edge.getLastStepMeanSpeed(edge_choice)
        state = self.get_state()
        self.state = state
        # average_travel_time = VehicleEnv.get_travel_time(self, rl_actions)
        veh_num = state[0]
        dest_pos = state[2]
        veh_2D_pos = state[3]
        print(dest_pos)
        print(veh_2D_pos)
        dist_veh_des = np.linalg.norm(np.array([dest_pos, veh_2D_pos], dtype=float))
        print(dist_veh_des)
        if avg_speed == 0:
            update_traveling_time = edge_len / veh_num
        else:
            update_traveling_time = edge_len / edge_avg_speed
        # 2D observation
        observation = [update_traveling_time, dist_veh_des, state[4]]

        return np.array(observation)

    def compute_reward(self, edge_choice):
        '''
        instant reward is the next state traveling time which is the traveling time
        after apply an action
        :return:
        '''
        obser = self.computer_observation(edge_choice)
        future_traveling_time = obser[0]
        dis_to_destination = obser[1]
        if edge_choice ==self.dest_edge:
            arrive = 100000000000
        else: arrive = 0
        if edge_choice == False:
            return -100000
        weight = 0.1
        wait_time = VehicleEnv.get_wait_time(self,self.veh_id)
        instant_reward = np.negative(future_traveling_time + weight*np.negative(dis_to_destination) )+ arrive - wait_time *100
        print('instant reward', instant_reward)
        return instant_reward

    def _sumo_step(self):
        traci.simulationStep()


    def close(self):
        traci.close()
        try:
            self.disp.stop()
        except AttributeError:
            pass

    def __del__(self):
        self.close()

    # def render(self, mode='human'):
    #     if self.virtual_display:
    #         # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
    #         #                          f"temp/img{self.sim_step}.jpg",
    #         #                          width=self.virtual_display[0],
    #         #                          height=self.virtual_display[1])
    #         img = self.disp.grab()
    #         if mode == 'rgb_array':
    #             return np.array(img)
    #         return img

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_conn{}_run{}'.format(self.label, run) + '.csv', index=False)

    # Below functions are for discrete state space

    def _discretize_density(self, density):
        return min(int(density * 10), 9)


    def render (self, mode="human"):
        s = "position: {:2d}  reward: {:2d} "
        print(s.format(self.state, self.reward))

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    # def initSimulator(self, withGUI, portnum):
    #     # Path to the sumo binary
    #     if withGUI:
    #         sumoBinary = "/Users/corinnajiang/sumo-gui"
    #     else:
    #         sumoBinary = "/Users/corinnajiang/sumo"
    #
    #     sumoConfig = "data/test.sumocfg"
    #
    #     # Call the sumo simulator
    #     sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
    #                                     "--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
    #                                     "--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout,
    #                                    stderr=sys.stderr)
    #
    #     # Initialize the simulation
    #     traci.init(portnum)
    #     return traci



