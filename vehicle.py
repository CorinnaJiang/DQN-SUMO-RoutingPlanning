import numpy as np
import traci
import os
import sys
import optparse
import random
import traci.constants as tc
import xml.etree.ElementTree as ET

# start_edge = 'gneE13'
# des_edge = 'gneE19'
# # curr_routelist = list()


class VehicleEnv:
    #
    def __init__(self, veh_id, Start_edge, Des_edge, routlist, netdir):

        self.veh_id = veh_id
        self.start_edge = Start_edge
        self.dest_edge = Des_edge
        self.curr_routelist = routlist
        self.netdir = netdir

        # self.set_vehicle_route(self.veh_id)

    def get_vehicle_info(self, veh_id):
        try:
            pos = traci.vehicle.getPosition(veh_id)
        except traci.TraCIException:
            pass
        # traci.vehicle.setActionStepLength(self.veh_id, 0.01, True)
        # for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, [
            tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION,
            tc.VAR_ROAD_ID,
            tc.VAR_SPEED,
            tc.VAR_EDGES,
            tc.VAR_POSITION,
            tc.VAR_ANGLE,
            tc.VAR_SPEED_WITHOUT_TRACI,
            tc.VAR_FUELCONSUMPTION,
            tc.VAR_DISTANCE,
            tc.VAR_MAXSPEED,
            tc.VAR_LASTACTIONTIME,
        ])
        info = traci.vehicle.getSubscriptionResults(veh_id)
        return info

    def unsubscribe_vehicle(self, veh_id):
        return traci.vehicle.unsubscribe(veh_id)

    def generate_routeList(self, veh_id):
        info = self.get_vehicle_info(veh_id)
        curroutelist = list()
        curroutelist.clear()
        cur_edges = info[tc.VAR_ROAD_ID]
        # print('cur_edges',info)
        if cur_edges[0] == ":":
            return []
        lanes = info[tc.VAR_LANE_INDEX]
        valid_next_edgelist = self.check_next_edge(cur_edges, lanes)
        if len(valid_next_edgelist)==0:
            return False
        print('valid_next_edgelist',valid_next_edgelist)
        # print(next_edge[0][0])
        # print(genertate_connect()['next'][next_edge[0][0]][lanes])
        for num in range(len(valid_next_edgelist)):
            nexted = valid_next_edgelist[num][0]
            curroutelist.append(nexted)
            # if valid_next_edgelist[num][0][0] == ':':
            #     nexted = generate_connection()['next'][valid_next_edgelist[num][0]][lanes]
            #     curroutelist.append(nexted[0][0])
            # else:
            #     curroutelist.append(valid_next_edgelist[0][0])
        print("curr:", curroutelist)

        return curroutelist

    def check_next_edge(self, edges, lane):
        """See parent class."""
        try:
            return self.generate_connection()['next'][edges][lane]
        except KeyError:
            return []

    def generate_connection(self):
        tree = ET.parse(self.netdir)
        root = tree.getroot()
        next_conn_data = dict()
        prev_conn_data = dict()

        for connection in root.findall('connection'):
            from_edge = connection.attrib['from']
            from_lane = int(connection.attrib['fromLane'])
            to_edge = connection.attrib['to']
            to_lane = int(connection.attrib['toLane'])

            # if from_edge[0] != ":":
            #     # if the edge is not an internal link, then get the next
            #     # edge/lane pair from the "via" element
            #     # via = connection.attrib['via'].rsplit('_', 1)
            #     # to_edge = via[0]
            #     # to_lane = int(via[1])
            #     to_edge = connection.attrib['to']
            #     to_lane = int
            # else:
            #     to_edge = connection.attrib['to']
            #     to_lane = int(connection.attrib['toLane'])

            if from_edge not in next_conn_data:
                next_conn_data[from_edge] = dict()

            if from_lane not in next_conn_data[from_edge]:
                next_conn_data[from_edge][from_lane] = list()

            if to_edge not in prev_conn_data:
                prev_conn_data[to_edge] = dict()

            if to_lane not in prev_conn_data[to_edge]:
                prev_conn_data[to_edge][to_lane] = list()

            next_conn_data[from_edge][from_lane].append((to_edge, to_lane))
            prev_conn_data[to_edge][to_lane].append((from_edge, from_lane))

        connection_data = {'next': next_conn_data, 'prev': prev_conn_data}
        return connection_data

    def generate_edgesList(self):
        tree = ET.parse(self.netdir)
        root = tree.getroot()
        net_data = dict()

        types_data = dict()

        for typ in root.findall('type'):
            type_id = typ.attrib['id']
            types_data[type_id] = dict()

            if 'speed' in typ.attrib:
                types_data[type_id]['speed'] = float(typ.attrib['speed'])
            else:
                types_data[type_id]['speed'] = None

            if 'numLanes' in typ.attrib:
                types_data[type_id]['numLanes'] = int(typ.attrib['numLanes'])
            else:
                types_data[type_id]['numLanes'] = None

        for edge in root.findall('edge'):
            edge_id = edge.attrib['id']

            # create a new key for this edge
            net_data[edge_id] = dict()

            # check for speed
            if 'speed' in edge:
                net_data[edge_id]['speed'] = float(edge.attrib['speed'])
            else:
                net_data[edge_id]['speed'] = None

            # if the edge has a type parameters, check that type for a
            # speed and parameter if one was not already found
            if 'type' in edge.attrib and edge.attrib['type'] in types_data:
                if net_data[edge_id]['speed'] is None:
                    net_data[edge_id]['speed'] = \
                        float(types_data[edge.attrib['type']]['speed'])

            # collect the length from the lane sub-element in the edge, the
            # number of lanes from the number of lane elements, and if needed,
            # also collect the speed value (assuming it is there)
            net_data[edge_id]['lanes'] = 0
            for i, lane in enumerate(edge):
                net_data[edge_id]['lanes'] += 1
                if i == 0:
                    net_data[edge_id]['length'] = float(lane.attrib['length'])
                    if net_data[edge_id]['speed'] is None \
                            and 'speed' in lane.attrib:
                        net_data[edge_id]['speed'] = float(
                            lane.attrib['speed'])

            # if no speed value is present anywhere, set it to some default
            if net_data[edge_id]['speed'] is None:
                net_data[edge_id]['speed'] = 30
        return net_data

    def choose_routes(self, veh_id):
        """See parent class."""
        # to hand the case of a single vehicle
        # if type(veh_ids) == str:
        #     veh_ids = [veh_ids]
        #     route_choices = [route_choices]
        #
        # for i, veh_id in enumerate(veh_ids):
        #     if route_choices[i] is not None:
        #         traci.vehicle.setRoute(
        #             vehID=veh_id, edgeList=route_choices[i])
        routelist = self.generate_routeList(veh_id)
        print("currlist ---", routelist)
        # choose_routelist = list()
        travel_time = dict()
        print(routelist)
        if routelist is None:
            return None
        elif self.Des_edge in routelist:
            return self.Des_edge
        else:
            for edgeId_choice in routelist:
                travel_time[edgeId_choice] = self.get_travel_time(edgeId_choice)
            choose_edgeId = min(travel_time, key=lambda x: travel_time[x])
            # choose_routelist.append(minTravel_edgeId)
            return choose_edgeId

    def choose_rl_routes(self, veh_id, index):
        routelist = self.generate_routeList(veh_id)
        print('---routelist ',routelist)
        if len(routelist) == 0:
            print('routelist is none')
            return None
        elif routelist == False :
            return 'NoneConnection'
        elif self.start_edge in routelist:
            print('des_edge', self.start_edge)
            return self.start_edge
        elif self.dest_edge in routelist:
            return self.dest_edge
        else:
            print('what is wrong', routelist)
            edge_id_choice = routelist[index]
            return edge_id_choice

    def assign_rl_route(self, veh_id, edge_choice):
        if self.start_edge not in self.curr_routelist :
            # assign the start poiont
            self.curr_routelist.append(self.start_edge)
        # arrive decision and make choice
        elif self.arrive_decision_zone(veh_id):
            print('arrive the excution_zone')
            # edge_choice = self.choose_rl_routes(veh_id,index)
            if edge_choice is None:
                return self.curr_routelist
            elif edge_choice not in self.curr_routelist:
                self.curr_routelist.append(edge_choice)
            else: return self.curr_routelist
            # else:
            #     return False
        # traci.vehicle.setRoute(veh_id, curr_routelist)
        print('curr_routelist:',self.curr_routelist )
        return self.curr_routelist

    def assign_route(self, veh_id):
        # routing_ids = []
        # routing_actions = []
        # for veh_id in self.k.vehicle.get_ids():
        #     if self.k.vehicle.get_routing_controller(veh_id) \
        #             is not None:
        #         routing_ids.append(veh_id)
        #         route_contr = self.k.vehicle.get_routing_controller(
        #             veh_id)
        #         routing_actions.append(route_contr.choose_route(self))
        #
        # choose_routes(routing_ids, routing_actions)
        if self.start_edge not in self.curr_routelist :
            # assign the start poiont
            self.curr_routelist .append(self.start_edge)
        # arrive decision and make choice
        if self.des_edge in self.curr_routelist :
            return self.curr_routelist
        elif self.arrive_decision_zone(veh_id):
            edge_choice = self.choose_routes(veh_id)
            if edge_choice is None:
                return self.curr_routelist
            if edge_choice not in self.curr_routelist :
                self.curr_routelist .append(edge_choice)
        # traci.vehicle.setRoute(veh_id, curr_routelist)
        print(self.curr_routelist )
        return self.curr_routelist

    def set_vehicle_route(self, veh_id, routelist):
        # routlist = self.assign_route(veh_id)
        # if self.arrive_excution_zone(veh_id):
        traci.vehicle.setRoute(veh_id, routelist)

    # def set_vehicle_route(self, veh_id):
    #     routelist = self.assign_route(veh_id)
    #     traci.vehicle.setRoute(veh_id,routelist)

    def edge_length(self, edge_id):
        edge_list = self.generate_edgesList()
        try:
            return edge_list[edge_id]['length']
        except KeyError:
            print('Error in edge length with key', edge_id)
            return -1001

    def get_vehicle_position(self, veh_id):
        info = self.get_vehicle_info(veh_id)
        return info[tc.VAR_LANEPOSITION]

    def get_vehicle_edge(self, veh_id):
        info = self.get_vehicle_info(veh_id)
        get_edge = info[tc.VAR_ROAD_ID]
        if get_edge[0] == ":":
            return None
        return get_edge

    def distance_to_intersection(self, veh_id):
        edge_id = self.get_vehicle_edge(veh_id)
        edge_len = self.edge_length(edge_id)
        relative_pos = self.get_vehicle_position(veh_id)
        dist = edge_len - relative_pos
        return dist, edge_len

    def arrive_decision_zone(self, veh_id):
        dist, edge_len = self.distance_to_intersection(veh_id)
        # if the decision zone is too large
        # vehicle may make one more choice
        # decision_dist = edge_len / 5
        speed = self.get_veh_speed(self.veh_id)
        Actiontime = self.get_action_length(self.veh_id)
        Watitime = self.get_wait_time(self.veh_id)
        decision_dist = np.minimum(edge_len, speed * Actiontime * 1.5)
        print('waitme ', Watitime)
        print('decision_dist', decision_dist)
        return bool(dist <= decision_dist or dist==0 or decision_dist==0)

    def arrive_excution_zone(self, veh_id):
        dist, edge_len = self.distance_to_intersection(veh_id)
        speed = self.get_veh_speed(self.veh_id)
        Actiontime = self.get_action_length(self.veh_id)
        # Watitime = self.get_wait_time(self.veh_id)
        # decision_dist = np.minimum(edge_len, Maxspeed * Actiontime * 2)
        Execution_dist = np.minimum(edge_len/2, speed * Actiontime*1.5)
        print('Execution_dist', Execution_dist)
        return bool(dist <= Execution_dist)


    def get_travel_time(self, edge_id):
        try:
            return traci.edge.getTraveltime(edge_id)
        except KeyError:
            print('Edge not exist', edge_id)
            return 0

    def vehicle_number_inedge(self, veh_id):
        edge = self.get_vehicle_edge(veh_id)
        number_inedge = traci.edge.getLastStepVehicleNumber(edgeID=edge)
        return number_inedge

    def get_veh_2D_position(self, veh_id):
        info = self.get_vehicle_info(veh_id)
        return info[tc.VAR_POSITION]

    def get_veh_maxspeed(self, veh_id):
        info = self.get_vehicle_info(veh_id)
        return info[tc.VAR_MAXSPEED]

    def get_veh_speed(self, veh_id):
        info = self.get_vehicle_info(veh_id)
        return info[tc.VAR_SPEED]

    def get_veh_lastactiontime(self, veh_id):
        info = self.get_vehicle_info(veh_id)
        return info[tc.VAR_LASTACTIONTIME]


    def get_edge_speed(self, edge_id):
        return self.generate_edgesList()[edge_id]['speed']

    def get_action_length(self, veh_id):
        return traci.vehicle.getActionStepLength(veh_id)

    def get_wait_time(self,veh_id):
        return traci.vehicle.getWaitingTime(veh_id)
