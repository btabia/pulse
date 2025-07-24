import numpy as np
import torch as th
import cv2
import matplotlib.pyplot as plt

class Mapping():
    def __init__(self, size_x, size_y, min_bound, max_bound):
        self.size_x = size_x
        self.size_y = size_y
        self.grid = np.zeros((size_x, size_y))
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.delta = max_bound - min_bound
        self.step_x = (self.delta[0] / self.size_x)
        self.step_y = (self.delta[1] / self.size_y)
        self.radius = np.sqrt(pow(self.step_x,2) + pow(self.step_y,2))


    def set_target_map(self, target_pos, target_radius):
        from skimage.draw import disk
        target_grid = np.zeros((self.size_x, self.size_y))
        t_x =  int(((target_pos[0][0]/self.delta[0]) - self.min_bound[0]) * self.size_x)
        t_y =  int(((target_pos[0][1]/self.delta[1]) - self.min_bound[1]) * self.size_y)
        radius = int(target_radius / self.step_x)
        rr, cc = disk((t_y, t_x), radius)
        target_grid[cc,rr]=1
        target_grid = cv2.normalize(target_grid, None, 0, 255, cv2.NORM_MINMAX)
        #fig = plt.imshow(target_grid)
        #plt.savefig('tmap.png')
        target_grid = th.from_numpy(target_grid).int()
        #target_grid = target_grid.int()
        #target_grid = target_grid.i
        #print("target grid: " + str(target_grid))
        return target_grid

    def update(self, pcd):
        self.grid = np.zeros((self.size_x, self.size_y))
        #self.queries = th.zeros(self.size_x * self.size_y)
        #for i in range(self.size_x):
        #    for j in range(self.size_y):
        #        self.queries[i][j] = th.Tensor([(i * self.step_x) + self.min_bound[0], (i * self.step_y) + self.min_bound[1], 0])
        h_limit = 0.08
        for i in range(self.size_x):
            for j in range(self.size_y):
                cgp = np.zeros(2)
                _max = 0
                cgp[0] = (i * self.step_x) + self.min_bound[0]
                cgp[1] = (j * self.step_y) + self.min_bound[1] 
                shape = np.shape(pcd)
                for k in range(shape[0]):
                    norm = np.sqrt(pow(cgp[0] - (pcd[k][0]),2) + pow(cgp[1] - (pcd[k][1]),2))
                    if norm < self.radius: 
                        if max(_max, pcd[k][2]) > 0.02:
                            _max = 0
                        else:
                            _max = max(_max, pcd[k][2])
                self.grid[i][j] = _max
        self.grid = cv2.normalize(self.grid, None, 0, 255, cv2.NORM_MINMAX)
        return self.grid
    

                    
        