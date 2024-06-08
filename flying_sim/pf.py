import numpy as np
import scipy
import math
# run PF with env_config.dt = 0.1 in config.py
class ParticleFilter:
    
    def __init__(self, drone):
        self.drone = drone
        self.nPF = int(1e3)             # Number of samples for particle filter
        self.Q = 1e-4*np.eye(6)
        self.R = 5e-2*np.eye(3)
        self.x_pf = np.zeros((6, ))             # Attitude state estimate, x_pf = [phi, theta, psi, p, q, r]
        self.init_prior = 1e-3*np.eye(6)
        self.X_bar = np.random.multivariate_normal(np.zeros((6, )), self.init_prior, self.nPF)      # Initialize particles
        self.weights = np.ones(self.nPF) / self.nPF         # Initialize particle weights
    
    def f_disc(self, x, input):
        # x = [phi, theta, psi, p, q, r]
        
        kinematics = np.array([[1, np.tan(x[1]) * np.sin(x[0]), np.tan(x[1]) * np.cos(x[0])],
                               [0, np.cos(x[0]), -np.sin(x[0])],
                               [0, np.sin(x[0]) / np.cos(x[1]), np.cos(x[0]) / np.cos(x[1])]])
        d_pose = kinematics @ x[3:]
        pose = x[:3] + self.drone.dt*d_pose
        
        moment = self.drone.get_moment(input)   # Resulting moment from inputs in body reference frame
        d_angular_velocity = np.array([((self.drone.I[1, 1] - self.drone.I[2, 2]) * x[5] * x[4] + moment[0]) / self.drone.I[0, 0],
                                       ((self.drone.I[2, 2] - self.drone.I[0, 0]) * x[5] * x[3] + moment[1]) / self.drone.I[1, 1],
                                       ((self.drone.I[0, 0] - self.drone.I[1, 1]) * x[4] * x[3] + moment[2]) / self.drone.I[2, 2]])
        angular_velocity = x[3:] + self.drone.dt*d_angular_velocity
        
        return np.hstack((pose, angular_velocity))
    
    
    # def f_disc1(self, x, input):
    #     # x = [phi, theta, psi, p, q, r]
        
    #     Rbe = self.drone.rotationBodytoEarth(x[0:3])
        
    #     kinematics = np.array([[1, np.tan(self.x_pf[1]) * np.sin(self.x_pf[0]), np.tan(self.x_pf[1]) * np.cos(self.x_pf[0])],
    #                         [0, np.cos(self.x_pf[0]), -np.sin(self.x_pf[0])],
    #                         [0, np.sin(self.x_pf[0]) / np.cos(self.x_pf[1]), np.cos(self.x_pf[0]) / np.cos(self.x_pf[1])]])
    #     d_pose = kinematics @ x[3:]
    #     pose = x[:3] + self.drone.dt*d_pose
        
    #     moment = self.drone.get_moment(input)   # Resulting moment from inputs in body reference frame
    #     d_angular_velocity = np.array([((self.drone.I[1, 1] - self.drone.I[2, 2]) * x[5] * x[4] + moment[0]) / self.drone.I[0, 0],
    #                                 ((self.drone.I[2, 2] - self.drone.I[0, 0]) * x[5] * x[3] + moment[1]) / self.drone.I[1, 1],
    #                                 ((self.drone.I[0, 0] - self.drone.I[1, 1]) * x[4] * x[3] + moment[2]) / self.drone.I[2, 2]])
    #     angular_velocity = x[3:] + self.drone.dt*d_angular_velocity
    
    #     return np.hstack((pose, angular_velocity))
        
        
    def g_disc(self, x):
        return x[3:]
        
    def attitude_estimate_PF(self, att_meas, input):
        # att_meas = [ax, ay, az, p, q, r]
        # accel = att_meas[:3]        # ax, ay, az measurements
        angles = att_meas[6:]       # p, q, r measurements
        X_bar = self.X_bar
        weights = self.weights
        Y_bar = np.zeros((self.nPF, 3))
        
        for i in range(self.nPF):
            
            # Prediction
            X_bar[i] = self.f_disc(X_bar[i], input) + np.random.multivariate_normal(np.zeros((6, )), self.Q)
            
            # Update
            Y_bar[i] = self.g_disc(X_bar[i])
            
            # Importance Sampling
            weights[i] = scipy.stats.multivariate_normal.pdf(angles, mean=Y_bar[i], cov=self.R)
        
        # Update
        weights = weights / np.sum(weights)
        self.x_pf = np.average(X_bar, weights=weights, axis=0)
                
        # Importance resampling
        # if (1/self.nPF)*np.sum(np.square(weights - np.ones(self.nPF) / self.nPF)) > 1: # Resampling condition 1
        # if (1/self.nPF)*np.sum(np.square(np.linalg.norm(X_bar - self.x_pf, axis=1))) > 1: # Resampling condition 2
        if True:    # Always resample
            # print((1/self.nPF)*np.sum(np.square(weights - np.ones(self.nPF) / self.nPF)))     # 1
            # print((1/self.nPF)*np.sum(np.square(np.linalg.norm(X_bar - self.x_pf, axis=1))))      # 2
            idx = np.array(np.random.choice(self.nPF, self.nPF, p=weights))
            self.X_bar = X_bar[idx]
            self.weights = np.ones(self.nPF) / self.nPF
        
        return self.x_pf
    