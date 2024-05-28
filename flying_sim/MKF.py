import numpy as np


class MKF:

    def __init__(self, dt=0.01):
        self.q_est = np.zeros((4, ))
        self.dt = dt

    def estimate_attitude(self, ang_vel):

        # Time update step
        self.q_est = self.q_est + self.dt * self.kinematics(ang_vel, self.q_est)
        self.q_est = self.q_est / np.linalg.norm(self.q_est)

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
           -np.sin(2 * q[1] * q[3] - 2 * q[0] * q[2]),
            np.arctan2(2 * q[1] * q[2] + 2 * q[0] * q[3], q[1]**2 + q[0]**2 - q[3]**2 - q[2]**2)
        ])

    def kinematics(self, ang_vel, quat):
        """ Get quaternion rates matrix from angular velocity and quaternion """
        W = np.array([
            [-quat[1], quat[0], -quat[3], quat[2]],
            [-quat[2], quat[3], quat[0], -quat[1]],
            [-quat[3], -quat[2], quat[1], quat[0]]
        ])
        dqdt = 1/2 * W.T @ ang_vel
        return dqdt


mkf = MKF()

print(mkf.quat2euler(mkf.euler2quat(np.array([np.pi/3, np.pi/6, np.pi/4]))))














