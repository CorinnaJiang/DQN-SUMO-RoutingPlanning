from abc import ABC

from vehicle import VehicleEnv
import numpy as np
import traci
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple
from flow.envs import Env
from Network import NetworkEnv


class MyEnv(Env, VehicleEnv, NetworkEnv):
    def __init__(self):
    # def __init__(self, env_params, sim_params, network, simulator):
        # Env.__init__(self,env_params, sim_params, network, simulator)
        self.veh_id = "vehicle_0"
        self.Arrived = 'gneE19'
        # self.step(0)

    def step(self, rl_actions):
        choose_route_index = rl_actions

        # try:
        #     pos = traci.vehicle.getPosition(self.veh_id)
        # except traci.TraCIException:
        #     VehicleEnv.unsubscribe_vehicle(self, self.veh_id)
        edge_choice = VehicleEnv.choose_rl_routes(self, self.veh_id, choose_route_index)
        # if edge_choice == self.Arrived:
        #     print('arrive the destionat')
        #     VehicleEnv.unsubscribe_vehicle(self, self.veh_id)
        cur_routelist = VehicleEnv.assign_rl_route(self, self.veh_id, choose_route_index)
        VehicleEnv.set_vehicle_route(self, self.veh_id, cur_routelist)
        print('edge_choice------',edge_choice)
        crash = NetworkEnv.check_collision(self)

        if edge_choice is not None:
            update_traveling_time, dist_veh_des = self._apply_rl_actions(edge_choice)

            states = self.get_state()

            self.state = np.asarray(states).T

            next_observation = np.copy(states)

            done = (edge_choice == self.Arrived)

            infos = {}

            reward = self.compute_reward(edge_choice, fail=crash)
            print('')

            return next_observation, reward, done, infos
        # elif edge_choice == self.Arrived:
        #     return False
        return None

    @property
    def action_space(self):
        actionList = VehicleEnv.generate_routeList(self, veh_id=self.veh_id)
        Discrete(len(actionList))

    @property
    def observation_space(self):
        vehicle_number = Box(
            low=0,
            high=np.inf,
            shape=(1,),
            dtype=np.float32)
        mean_speed = Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32)

        dis_to_des = Box(
            low=0,
            high=np.inf,
            shape=(1,),
            dtype=np.float32)

        return Tuple((vehicle_number, mean_speed, dis_to_des))

    def get_state(self):
        '''
        state1 : the numbers of vehicle in the edges
        state2 : the average driving speeds int hte edges
        state3 : vehicle position
        state4 : traveling time
        :return:
        '''
        # choose_edge = rl_actions
        vehicle_number = VehicleEnv.vehicle_number_inedge(self, self.veh_id)
        # mean_speed = traci.edge.getLastStepMeanSpeed(choose_edge)
        vehicle_position = VehicleEnv.get_vehicle_position(self, self.veh_id)
        # average_travel_time = VehicleEnv.get_travel_time(self, choose_edge)
        destination_pos = NetworkEnv.get_edge_2D_position(self, self.Arrived)
        vehicle_2D_position = VehicleEnv.get_veh_2D_position(self, self.veh_id)

        state = [vehicle_number,
                 vehicle_position,
                 destination_pos,
                 vehicle_2D_position]

        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        '''
        After appling an action, how the state will change
        :param rl_actions: the actinon (choose_edgeID) of RL vehicle
        :return: the change of the state-- traveling time which can be used
        in reward calcaulation
        '''
        edge_len = VehicleEnv.edge_length(self, rl_actions)
        edge_avg_speed = VehicleEnv.get_edge_speed(self, rl_actions)
        state = self.get_state()
        avg_speed = traci.edge.getLastStepMeanSpeed(rl_actions)
        # average_travel_time = VehicleEnv.get_travel_time(self, rl_actions)
        veh_num= state[0]
        dest_pos = state[2][1]
        veh_2D_pos = state[3]
        print(dest_pos)
        print(veh_2D_pos)
        dist_veh_des = np.linalg.norm(np.array([dest_pos, veh_2D_pos], dtype= float))
        print(dist_veh_des)
        if avg_speed == 0 :
            update_traveling_time = edge_len / veh_num
        else:
            update_traveling_time = edge_len /edge_avg_speed

        return update_traveling_time, dist_veh_des


    def compute_reward(self, rl_actions, **kwargs):
        '''
        instant reward is the next state traveling time which is the traveling time
        after apply an action
        :return:
        '''
        future_traveling_time, dist_veh_des = self._apply_rl_actions(rl_actions)
        instant_reward = np.negative(future_traveling_time)

        return instant_reward

