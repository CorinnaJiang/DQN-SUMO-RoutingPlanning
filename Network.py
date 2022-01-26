import sumolib
import traci

class NetworkEnv():

    def get_edge_2D_position(self, edge_id, dir):
        net = sumolib.net.readNet(dir)
        pos = net.getEdge(edge_id).getShape()
        return pos

    def check_collision(self):
        """See parent class."""
        return traci.simulation.getStartingTeleportNumber() != 0



