import matplotlib.pyplot as plt
import json
import numpy as np
import os



## this script illustrate the recorded data into animated graph
# the target must be at the centre of the graph (0,0)
# the brush must show the trajectory with a heat color for the velocity
# the brush must show the trajectory with a heat color for the velocity
# for an episode, extract the position and velocity data
class DataFrame(object):
    """[summary]

        Args:
            current_time_step (int): [description]
            current_time (float): [description]
            data (dict): [description]
        """

    def __init__(self, current_time_step: int, current_time: float, data: dict) -> None:
        self.current_time_step = current_time_step
        self.current_time = current_time
        self.data = data

    def get_dict(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return {"current_time": self.current_time, "current_time_step": self.current_time_step, "data": self.data}

    def __str__(self) -> str:
        return str(self.get_dict())

    @classmethod
    def init_from_dict(cls, dict_representation: dict):
        """[summary]

        Args:
            dict_representation (dict): [description]

        Returns:
            DataFrame: [description]
        """
        frame = object.__new__(cls)
        frame.current_time_step = dict_representation["current_time_step"]
        frame.current_time = dict_representation["current_time"]
        frame.data = dict_representation["data"]
        return frame


class FrameStruct: 
    def __init__(self, log_path: str):
        # raw data from files
        self.timesteps =  np.array([])
        self.brush_positions = np.array([])
        self.brush_orientations = np.array([])
        self.debris_position = np.array([])
        self.target_positions = np.array([])
        # processed data
        self.target_brush_positions = np.array([])
        self.target_brush_distance = np.array([])
        self.target_brush_speed = np.array([])
        self.forces_xy = np.array([])
        
        self.target_debris_positions = np.array([])
        self.target_debris_distance = np.array([])
        self.target_debris_speed = np.array([])

        self.log_path = log_path
        self.data_keyword = "Isaac Sim Data"


    def load_data(self):
        _data_frames = []
        print("log path: " + self.log_path)
        with open(self.log_path) as json_file:
            json_data = json.load(json_file)
            data_frames = json_data[self.data_keyword ]
            data_frames = [DataFrame.init_from_dict(dict_representation=data_frame) for data_frame in data_frames]
            _data_frames = data_frames
        #print("data frame" + str(data_frames))
        return _data_frames
    
    def process_frame(self) -> None:
        # get the data frame
        timesteps = []
        brush_positions = []
        brush_orientations = []
        debris_position = []
        target_positions = []
        data_frames = self.load_data()
        self.forces = []
        f = 15 # 15Hz timestep
        timeskip = 20
        
        data_frames_len = self.get_num_data_frame(data_frames)
        #print("number of data frames :  " + str(data_frames[1]))
        data_frames.pop(0) # temporary
        data_frames_len = self.get_num_data_frame(data_frames)
        for i in range(data_frames_len):
            data_frame = data_frames[i]
            timesteps.append(np.array([data_frame.current_time]))
            brush_positions.append(np.array(data_frame.data["brush_position"]))
            brush_orientations.append(np.array(data_frame.data["brush_orientation"]))
            debris_position.append(np.array(data_frame.data["debris_position"]))
            target_positions.append(np.array(data_frame.data["target_position"]))
            self.forces.append(np.array(data_frame.data["ee_force_sensor"]))
        
        self.timesteps = data_frames_len
        self.total_time = data_frames_len * (1/f) * (1/timeskip)
        print("data frame len: " + str(data_frames_len))
        self.brush_positions = np.array(brush_positions)
        self.brush_orientations = np.array(brush_orientations)
        self.debris_position = np.array(debris_position)
        self.target_positions = np.array(target_positions)
        self.forces = np.array(self.forces)



        self.target_brush_positions = self.brush_positions - self.target_positions
        
        self.target_debris_positions = self.debris_position - self.target_positions

        self.target_debris_positions_xy = (self.debris_position[:2] - self.target_positions[:2]) * 100
        self.shortest_length = np.linalg.norm(self.target_debris_positions_xy[0])
        

        for i in range(data_frames_len):
            self.target_brush_distance = np.append(self.target_brush_distance, np.linalg.norm(self.target_brush_positions[i][:2])) 
            
            self.target_debris_distance = np.append(self.target_debris_distance, np.linalg.norm(self.target_debris_positions[i][:2])) 
        
        for i in range(int(data_frames_len/20)):
            self.forces_xy = np.append(self.forces_xy, np.linalg.norm(self.forces[i*20][:2]))
        

    def get_time_steps(self):
        return self.timesteps
    
    def get_total_time(self):
        return self.total_time
    
    def get_num_data_frame(self, _data_frame) -> int:
        return len(_data_frame)
    
    def get_target_brush_positions(self):
        return self.target_brush_positions 
    
    def get_target_brush_distance(self):
        return self.target_brush_distance

    def get_target_brush_speed(self):
        return self.target_brush_speed

    def get_target_debris_positions(self):
        return self.target_debris_positions_xy
    
    def get_target_debris_distance(self):
        return self.target_debris_distance
    

    def get_target_debris_speed(self):
        return self.target_debris_speed
    
    def get_normal_force(self):
        return self.forces_xy
    
    def dist_normalized(self, c_min, c_max, dist, what="object"):
        normed_min, normed_max = 0, 1
        if (dist - c_min) == 0 or (c_max - c_min) == 0:
            print("dist: ", dist, "c_min: ", c_min, "c_max: ", c_max, "for: ", what)
        x_normed = (dist - c_min) / (c_max - c_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return round(x_normed, 4)

    
class EpisodeAnalysis:
    def __init__(self, base_path: str):
        self.episode = FrameStruct(base_path)
        self.episode.load_data()
        self.episode.process_frame()
        self.debris_travel_time = 0
        self.success = 0
        self.optimal_travel_length = 0
        self.optimal_travel_time = 0
        self.debris_path_length = 0
        self.speed_error_mean = 0
        self.speed_error_std = 0
        self.contact_rate = 0
        self.target_speed = 0

        self.target_debris_success_tolerance = 0.1
        self.target_debris_speed_command = np.sqrt(6)
        self.target_debris_speed_tolerance = np.sqrt(1)

    def calculate_travel_time(self) -> float:
        travel_time = self.episode.get_total_time()
        return travel_time
    
    def calculate_debris_path_length(self) -> float:
       debris_target_pos = self.episode.get_target_debris_distance()
       debris_target_pos_len = len(debris_target_pos)
       prev_debris_target_pos = debris_target_pos[0]
       cumul = 0
       delta = 0
       for i in range(int(debris_target_pos_len / 20)):
            if i == 0:
                delta = 0
            else:
                delta = np.absolute(debris_target_pos[i*20] - debris_target_pos[(i*20) - 1])
            cumul = delta + cumul
       return cumul

    def calculate_success(self) -> float: 
        success = 0

        target_debris_dists = self.episode.get_target_debris_distance()
        target_debris_dists_len = len(target_debris_dists) / 20
        if target_debris_dists_len < 256: 
            success = 1
        else : 
            success = 0
        return success
    
    def calculate_shortest_path(self) -> float: 
        target_debris_dist = self.episode.get_target_debris_distance()
        shortest_path = target_debris_dist[0]
        #print("shortest length: " + str(self.shortest_length))
        return shortest_path
    
    def calculate_mean_std_force(self) -> float: 
        total_force = 0
        forces = self.episode.get_normal_force()
        for i in range(len(forces)):
            total_force = total_force + forces[i]
        force_mean = total_force / len(forces)
        total_diff = 0
        for i in range(len(forces)):
            total_diff = total_diff + pow(forces[i] - force_mean,2)
        std = np.sqrt(total_diff/len(forces))
        return force_mean, std



    
    def calculate_contact_rate(self) -> float:
        
        contact_rate = 0
        contact_tolerance = 0.1
        brush_dist = self.episode.get_target_brush_distance()
        debris_dist = self.episode.get_target_debris_distance()
        diff = brush_dist - debris_dist
        contact = np.ones(len(diff))
        contact = np.where(diff > contact_tolerance, 0 ,1)
        contact = np.where(contact < -contact_tolerance, 1, contact)
        cumul = 0
        for i in range(len(contact)):
            cumul = contact[i] + cumul
        contact_rate = cumul / len(contact)
        return contact_rate
    
    def calculate_velocity_error(self):
        debris_speed = self.episode.get_target_debris_speed()
        speed_error = np.array([])
        for i in range(len(debris_speed)):
            np.append(speed_error, self.target_debris_speed_command - debris_speed[i])
        cumul = 0
        for i in range(len(speed_error)):
            cumul = speed_error + cumul
        debris_speed_mean = cumul / len(debris_speed)

        cumul_std = 0
        for i in range(len(len(speed_error))): 
            cumul_std = (debris_speed[i] - debris_speed_mean) + cumul_std
        debris_speed_std = np.sqrt(cumul_std / len(debris_speed))

        return debris_speed_mean, debris_speed_std
    
    def calculate_optimal_travel_time(self) -> float:
        debris_dist = self.episode.get_target_debris_distance()
        debris_dist_init = debris_dist[0]
        speed = self.target_debris_speed_command
        return debris_dist_init / speed

    def calculate_optimal_travel_distance(self, optimal_travel_time) -> float:
        # get initial position
        debris_pos = self.episode.get_target_debris_distance()
        return debris_pos[0] 
    
    def calculate_travel_time_ratio(self, optimal_travel_time, travel_time):
        # the closer the ratio is to the value 1, the better. 
        return travel_time / optimal_travel_time
    
    def calculate_travel_distance_ratio(self, optimal_travel_distance, travel_distance):
        # the closer the ratio is to the value 1, the better. 
        return travel_distance / optimal_travel_distance 



class DataPlot:
    def __init__(self, base_path):
        self.episode_analyses = []
        self.base_path = base_path

    def plot_episode_trajectories(self, iteration, plt):

        debris_trajectory =  self.episode_analyses[iteration].episode.get_target_debris_positions()
        debris_trajectory_x = debris_trajectory[:,0]
        debris_trajectory_y = debris_trajectory[:,1]
        debris_optimal_trajectory_x = np.array([debris_trajectory[0][0], 0])
        debris_optimal_trajectory_y = np.array([debris_trajectory[0][1], 0])

        # plot  optimal trajectory line
        plt.plot(debris_optimal_trajectory_x, debris_optimal_trajectory_y,  
                color=(0,0,0.5), # blue
                label= "Debris_trajectory_ep_" + str(iteration)) 
        
        # plot real episode trajectory line
        plt.plot(debris_trajectory_x, debris_trajectory_y,  
            color=(0,0.5,0), # green
            label= "Debris_opt_trajectory_ep_" + str(iteration)) 
        
        # mark the episode number
        plt.text(debris_trajectory_x[0] + 0.1, debris_trajectory_y[0] + 0.1, str(iteration))

        plt.plot(debris_trajectory_x[0], debris_trajectory_y[0],  
            color=(0,0,0), # black
            marker='^')
        
        plt.plot(debris_trajectory_x[len(debris_trajectory_x) - 1], debris_trajectory_y[len(debris_trajectory_y) - 1],  
            color=(0,0,0), # black
            marker='v')

        return
    
    def plot_targets(self, plt):
        # plot 6 circles from 5 to 0.5 (5,4,3,2,1,0.5)
        circles = []
        for i in range(5):
            circles.append(plt.Circle((0, 0), i + 1, color='r', fill=False, linestyle = "--"))
            plt.gca().add_patch(circles[i])

        target = plt.Circle((0,0), 0.5, color='r', fill=True)
        target = plt.Rectangle((-8,-8), 16,16, color = 'black', fill = False)
        plt.gca().add_patch(target)

    def calculate_success_rate(self) -> float:
        success_rate = 0
        for i in range(len(self.episode_analyses)):
            success_rate = self.episode_analyses[i].calculate_success() + success_rate
        success_rate = success_rate / len(self.episode_analyses)
        return success_rate
    
    def calculate_contact_rate(self): 
        contact_rate = 0
        for i in range(len(self.episode_analyses)):
            contact_rate = self.episode_analyses[i].calculate_contact_rate() + contact_rate
        contact_rate = contact_rate / len(self.episode_analyses)
        return contact_rate
    
    def get_travel_times(self):
        travel_time_total = 0
        for i in range(len(self.episode_analyses)):
            travel_time_total = self.episode_analyses[i].calculate_travel_time() + travel_time_total
        #print("travel time total: " + str(travel_time_total))
        #print("nb episodes: " + str(len(self.episode_analyses)))
        travel_time_mean = travel_time_total / len(self.episode_analyses)
        travel_time_diff_total = 0
        for i in range(len(self.episode_analyses)):
            travel_time_diff_total = pow(self.episode_analyses[i].calculate_travel_time() - travel_time_mean, 2) + travel_time_diff_total
        travel_time_std = np.sqrt(travel_time_diff_total/len(self.episode_analyses) )
        return travel_time_mean, travel_time_std 
    
    def get_spl(self): 
        t = 0
        for i in range(len(self.episode_analyses)):
            success_rate = self.episode_analyses[i].calculate_success()
            shortest_path = self.episode_analyses[i].calculate_shortest_path()
            actual_path = self.episode_analyses[i].calculate_debris_path_length()
            t = t +  (success_rate * (shortest_path / max(actual_path, shortest_path)))
        spl = t / len(self.episode_analyses)
        return spl
    
    def get_forces(self):
        f = 0
        total_mean = 0
        total_std = 0
        for i in range(len(self.episode_analyses)):
            mean, std =  self.episode_analyses[i].calculate_mean_std_force()
            total_mean = total_mean + mean
            total_std = total_std + std
        mean = total_mean / len(self.episode_analyses)
        std = total_std / len(self.episode_analyses)
        return mean, std

    
    def get_travel_distances(self):
        travel_dist_total = 0
        for i in range(len(self.episode_analyses)):
            travel_dist_total = travel_dist_total + self.episode_analyses[i].calculate_debris_path_length()
        travel_dist_mean = travel_dist_total / len(self.episode_analyses)
        travel_dist_diff = 0
        for i in range(len(self.episode_analyses)):
            travel_dist_diff = travel_dist_diff + pow(self.episode_analyses[i].calculate_debris_path_length() - travel_dist_mean,2)
        travel_dist_std = np.sqrt(travel_dist_diff / len(self.episode_analyses))
        return travel_dist_mean, travel_dist_std
    
    def get_velocity_error(self):
        velocity = 0

    def plot_analysis(self, plt):
        print("------ Policy performance analysis ------")
        str_1 = "1) Policy Success Rate: " + str(self.calculate_success_rate())
        print(str_1)
        print("2) Policy Contact Rate: " + str(self.calculate_contact_rate()))
        t_mean, t_std= self.get_travel_times()
        spl = self.get_spl()
        travel_mean, travel_std = self.get_travel_distances()
        normal_mean, normal_std = self.get_forces()
        print("3) Policy Travel time :" )
        print("     3.1) travel time mean: " + str(t_mean))
        print("     3.2) travel time std: " + str(t_std))
        print("4) Policy Travel Distance :" )
        print("     4.1) travel distance  mean: " + str(travel_mean * 100))
        print("     4.2) travel distance  std: " + str(travel_std * 100))
        print("5) SPL :" + str(spl) )
        print("6) Measured Normal Force :" )
        print("     4.1) Normal force mean: " + str(normal_mean))
        print("     4.2) Normal force std: " + str(normal_std))
        print("------ End of analysis ------")
        plt.text(-8, 8 + 0.1, str_1)
        #plt.text(-8, 8  - 1 , str_2)

    def run(self):
        # initialise plot

        plt.grid(True)

        plt.rcParams["figure.figsize"] = [5, 5]
        plt.rcParams["figure.autolayout"] = False

        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')

        target_x = np.arange(-5,5,0.5)
        target_y = np.arange(-5,5,0.5)

        target_x_inv = np.arange(5,-5,0.5)
        target_y_inv = np.arange(5,-5,0.5)

        ep = 1
        begin_filename = "episode_"
        end_name = "_log.json"
        isFile = True

        while(isFile):
            file_path = base_path + begin_filename + str(ep) + end_name
            if os.path.isfile(file_path) == True:
                self.episode_analyses.append(EpisodeAnalysis(file_path))
                self.plot_episode_trajectories(ep - 1, plt)
                ep = ep + 1
                plt.plot(target_x,target_y, color='none')
                plt.plot(target_x_inv,target_y_inv, color='none')
                #plt.set_aspect('equal', 'box')
                plt.xlabel('x_position ')
                plt.ylabel('y_position')
                plt.title("brush and debris trajectory / Unit: dm / 1dm -> 10cm")
            elif os.path.isfile(file_path) == False:
                break
        self.plot_analysis(plt)
        self.plot_targets(plt)
        #plt.legend()
        plt.show()



base_path = "/home/btabia/git/residual_soft_push/datalog/play_TQC17/" # 691 = > 90% / 712 => 80% , 762 => 97%

dataplot = DataPlot(base_path=base_path)

dataplot.run()



