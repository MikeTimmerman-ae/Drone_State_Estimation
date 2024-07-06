import numpy as np
from flying_sim.pf import ParticleFilter
from flying_sim.mkf import MultiplicativeKF


class Sensors:

    def __init__(self, drone):
        self.drone = drone
        self.gyro_noise = 5e-2
        self.acc_noise = 1e-2
        self.mag_noise = 1e-2
        self.pos_noise = 1e-0
        self.vel_noise = 1e-1

    def sensor_measurements(self, input):
        ang_vel, acc, mag = self.IMU(input)
        pos, vel_e = self.GPS()
        return np.hstack((pos, vel_e, acc, mag, ang_vel))

    def IMU(self, input):
        # Gyroscope model
        ang_vel = self.drone.angular_velocity + np.random.normal(1e-3, self.gyro_noise)

        # Accelerometer model
        f_b = self.drone.get_force(self.drone.state, input) / self.drone.m                              # specific force in body-frame
        g_b = self.drone.rotationBodytoEarth(self.drone.attitude).T @ np.array([0, 0, self.drone.g])    # gravity in body-frame
        acc = f_b + g_b + np.random.normal(0, self.acc_noise)
        mag = self.drone.rotationBodytoEarth(self.drone.attitude).T @ np.array([1, 0, 0])

        return ang_vel, acc, mag

    def GPS(self):
        # GPS model
        pos = self.drone.position + np.random.normal(0, self.pos_noise)
        vel_e = self.drone.velocity_e + np.random.normal(0, self.vel_noise)

        return pos, vel_e


class Estimation:

    def __init__(self, drone):
        self.drone = drone
        self.state_estimate = np.zeros((12, ))
        self.pose_cov = 1e-3 * np.eye(6)
        self.pose_Q = 1e-3 * np.eye(6)
        self.pose_R = 1e-1 * np.eye(6)
        self.att_cov = 1e-3 * np.eye(6)
        self.pf = ParticleFilter(self.drone)
        self.mkf = MultiplicativeKF(self.drone)

    @property
    def attitude(self):
        return self.state_estimate[0:3]

    @property
    def angular_velocity(self):
        return self.state_estimate[3:6]

    @property
    def position(self):
        return self.state_estimate[6:9]

    @property
    def velocity_e(self):
        return self.state_estimate[9:]

    def estimate_state(self, meas, input):
        pose_est = self.position_estimate(meas[:6], input)
        att_est = self.attitude_estimate(meas[6:], input)
        self.state_estimate = np.hstack((att_est, pose_est))

    def position_estimate(self, pose_meas, input):
        pose_est = self.state_estimate[6:]              # x_k
        att_est = self.state_estimate[:6]

        A = np.eye(6) + self.drone.dt * np.diag([1, 1, 1], 3)           # State transition matrix
        C = np.eye(6)                                                   # Measurement matrix

        # Update step
        T_acc = self.drone.kf * np.sum(input) / self.drone.m
        acc_e = self.drone.rotationBodytoEarth(att_est[:3]) @ np.array([0, 0, T_acc]) + np.array([0, 0, self.drone.g])
        pose_est = A @ pose_est + self.drone.dt * np.hstack((np.zeros(3,), acc_e))      # x_{k+1} = A x_k + g(x, u)
        self.pose_cov = A @ self.pose_cov @ A.T + self.pose_Q

        # Correction step
        K = self.pose_cov @ C.T @ np.linalg.inv(self.pose_R + C @ self.pose_cov @ C.T)
        pose_est = pose_est + K @ (pose_meas - C @ pose_est)
        self.pose_cov = self.pose_cov - K @ C @ self.pose_cov

        return pose_est

    def attitude_estimate(self, att_meas, input):

        # return np.hstack((np.zeros(3,), att_meas[3:]))        # No attitude estimation
        # return self.pf.attitude_estimate_PF(att_meas, input)    # PF attutude estimation
        return self.mkf.estimate_attitude(att_meas)