import numpy as np
from flying_sim.drone import Drone
from configs.config import Config


class MultiplicativeKF:

    def __init__(self, drone):
        self.q_est = np.array([1, 0, 0, 0])
        self.dt = drone.dt
        self.att_err = np.zeros((3, ))
        self.cov = 1e-2 * np.eye(3)
        self.Q = 5e-2 * np.eye(3)
        self.Racc = np.diag([1e-2, 1e-2, 1e-2])
        self.Rmag = np.diag([1e-2, 1e-2, 1e-2])
        self.mag_model = np.array([1, 0, 0])

    def estimate_attitude(self, att_meas):
        acc = att_meas[:3]
        mag_field = att_meas[3:6]
        ang_vel = att_meas[6:]

        # Time update step
        F = self.transition_matrix(ang_vel)
        self.att_err = np.zeros((3,))                                                   # Reset attitude error
        self.cov = F @ self.cov @ F.T + self.Q                                          # Propagate state covariance
        q_est_ = self.q_est + self.dt * self.kinematics(ang_vel, self.q_est)            # Propagate quaternion attitude
        self.q_est = q_est_ / np.linalg.norm(q_est_)

        # Measurement update step
        # Determine attitude error vectors
        Rbe = self.rotationEarthtoBody(self.q_est)

        n_err = np.cross(acc / np.linalg.norm(acc), Rbe @ np.array([0, 0, 1]))
        ang_err = np.arccos(np.dot(acc / np.linalg.norm(acc), Rbe @ np.array([0, 0, 1])))
        self.att_err = ang_err * n_err

        n_err = np.cross(mag_field, Rbe @ self.mag_model)
        ang_err = np.arccos(np.dot(mag_field, Rbe @ self.mag_model))
        self.att_err = ang_err * n_err

        # # 1. Accelerometer update
        K = self.cov @ np.linalg.inv(self.Racc * (1 + 10 * (1 - np.linalg.norm(acc) / 9.81) ** 2) + self.cov)
        self.cov = self.cov - K @ self.cov
        self.att_err = K @ self.att_err
        self.q_est = self.quaternion_multiply(self.axisangle2quat(self.att_err), self.q_est)

        # # 2. Magnetic field update
        K = self.cov @ np.linalg.inv(self.Rmag + self.cov)
        self.cov = self.cov - K @ self.cov
        self.att_err = K @ self.att_err
        self.q_est = self.quaternion_multiply(self.axisangle2quat(self.att_err), self.q_est)

        att_est = self.quat2euler(self.q_est)
        return np.hstack((att_est, ang_vel))

    def transition_matrix(self, ang_vel):
        return -np.array([
            [0, -ang_vel[2], ang_vel[1]],
            [ang_vel[2], 0, -ang_vel[0]],
            [-ang_vel[1], ang_vel[0], 0],
        ])

    def euler2quat(self, euler_angles):
        phi, theta, psi = euler_angles
        return np.array([
            np.cos(phi/2) * np.cos(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.sin(theta/2) * np.sin(psi/2),
           -np.cos(phi/2) * np.sin(theta/2) * np.sin(psi/2) + np.cos(theta/2) * np.cos(psi/2) * np.sin(phi/2),
            np.cos(phi/2) * np.cos(psi/2) * np.sin(theta/2) + np.sin(phi/2) * np.cos(theta/2) * np.sin(psi/2),
            np.cos(phi/2) * np.cos(theta/2) * np.sin(psi/2) - np.sin(phi/2) * np.cos(psi/2) * np.sin(theta/2)
        ])

    def quat2euler(self, q):
        return np.array([
            np.arctan2(2 * q[2] * q[3] + 2 * q[0] * q[1], q[3]**2 - q[2]**2 - q[1]**2 + q[0]**2),
           -np.arcsin(2 * q[1] * q[3] - 2 * q[0] * q[2]),
            np.arctan2(2 * q[1] * q[2] + 2 * q[0] * q[3], q[1]**2 + q[0]**2 - q[3]**2 - q[2]**2)
        ])

    def axisangle2quat(self, vec):
        arg = np.linalg.norm(vec)
        return np.array([
            np.cos(arg / 2),
            vec[0] / arg * np.sin(arg / 2),
            vec[1] / arg * np.sin(arg / 2),
            vec[2] / arg * np.sin(arg / 2),
        ])

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
                         x2 * w1 + w2 * x1 - y2 * z1 + z2 * y1,
                         y2 * w1 + w2 * y1 + x2 * z1 - z2 * x1,
                         z2 * w1 + w2 * z1 - x2 * y1 + y2 * x1])

    def rotationEarthtoBody(self, quat) -> np.ndarray:
        return np.array([
            [quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2, 2*quat[1]*quat[2]+2*quat[0]*quat[3], 2*quat[1]*quat[3]-2*quat[0]*quat[2]],
            [2*quat[1]*quat[2]-2*quat[0]*quat[3], quat[0]**2-quat[1]**2+quat[2]**2-quat[3]**2, 2*quat[2]*quat[3]+2*quat[0]*quat[1]],
            [2*quat[1]*quat[3]+2*quat[0]*quat[2], 2*quat[2]*quat[3]-2*quat[0]*quat[1], quat[0]**2-quat[1]**2-quat[2]**2+quat[3]**2]
        ])

    def kinematics(self, ang_vel, quat):
        """ Get quaternion rates matrix from angular velocity and quaternion """

        return 1/2 * self.quaternion_multiply(quat, np.hstack((0, ang_vel)))


config = Config()
drone = Drone(config)
mkf = MultiplicativeKF(drone)

# R = mkf.rotationEarthtoBody(np.array([1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4)]))
