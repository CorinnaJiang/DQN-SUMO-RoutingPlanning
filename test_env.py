from MainEnv import SumoRouteEnv
from gym.utils.env_checker import check_env


def test_api():
    env = SumoRouteEnv(net_file = 'data/test.net.xml',
                       route_file= 'data/test.rou.xml',
                       sim_file = "data/test.sumocfg",
                       veh_id='vehicle_0',
                       Start_edge='gneE13',
                       Destination_edge='gneE19',
                       use_gui= True)
    env.reset()
    check_env(env)
    print('Create environment successfully!')
    env.close()

if __name__ == '__main__':
    test_api()

