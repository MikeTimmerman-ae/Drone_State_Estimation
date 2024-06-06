import numpy as np
import scipy
import scipy.stats

def unsented_transform(x, P, lam=2.):
    n = len(x)
    print(x)
    sigma_points = np.zeros((2*n+1, n))
    weights = np.zeros((2*n+1, ))

    sigma_points[0] = x
    weights[0] = lam / (lam + n)
    cov = scipy.linalg.sqrtm((lam + n) * P)
    for i in range(n):
        sigma_points[i+1] = sigma_points[0] + cov[:, i]
        weights[i+1] = 1 / (2 * (lam + n))
        sigma_points[n+i+1] = sigma_points[0] - cov[:, i]
        weights[n+i+1] = 1 / (2 * (lam + n))
    return sigma_points, weights

def inverse(sigma_points, weights):
    print(sigma_points.shape)
    print(weights.shape)
    x_est = np.average(sigma_points, weights=weights, axis=0)
    P = np.sum(np.array([w * np.outer(sigma_point - x_est, sigma_point - x_est) for w, sigma_point in zip(weights, sigma_points)]), axis=0) 
    print(x_est)
    return x_est, P

class unscentedKalmanFilter:

    def __init__(self, drone):
        self.drone = drone

        self.x = 6                      # state dimension
        self.y = 3                      # measurement dimension

        self.numSigma = 2 * self.x + 1
        # self.lamb = 3 - self.x 
        # self.a = 1e-3
        # self.lamb = self.a**2 * (self.x) - self.x
        self.lamb = 2
        self.W0 = self.lamb / (self.x + self.lamb)
        self.Wi = 1 / (2 * (self.x + self.lamb))

        self.Q = 1e-3*np.eye(6)
        self.R = 1e-3*np.eye(3)

        self.x_pf = np.zeros((self.x, ))                    # Attitude state estimate, x_pf = [phi, theta, psi, p, q, r]
        self.P = 1e-2*np.eye(6)
        self.xbar = np.zeros((self.x, self.numSigma))
        self.xbarF = np.zeros((self.x, self.numSigma))
        self.ybar = np.zeros((self.y, self.numSigma))
        self.yhat = np.zeros((self.y, ))
        self.Py = np.zeros((self.y, self.y))
        self.Pxy = np.zeros((self.x, self.y))

    def f_disc(self, x, input):

        kinematics = np.array([[1, np.tan(self.x_pf[1]) * np.sin(self.x_pf[0]), np.tan(self.x_pf[1]) * np.cos(self.x_pf[0])],
                               [0, np.cos(self.x_pf[0]), -np.sin(self.x_pf[0])],
                               [0, np.sin(self.x_pf[0]) / np.cos(self.x_pf[1]), np.cos(self.x_pf[0]) / np.cos(self.x_pf[1])]])
        d_pose = kinematics @ x[3:]
        pose = x[:3] + self.drone.dt*d_pose

        moment = self.drone.get_moment(input)   # Resulting moment from inputs in body reference frame
        d_angular_velocity = np.array([((self.drone.I[1, 1] - self.drone.I[2, 2]) * x[5] * x[4] + moment[0]) / self.drone.I[0, 0],
                                       ((self.drone.I[2, 2] - self.drone.I[0, 0]) * x[5] * x[3] + moment[1]) / self.drone.I[1, 1],
                                       ((self.drone.I[0, 0] - self.drone.I[1, 1]) * x[4] * x[3] + moment[2]) / self.drone.I[2, 2]])
        angular_velocity = x[3:] + self.drone.dt*d_angular_velocity

        return np.hstack((pose, angular_velocity))


    def g_disc(self, x):
        return x[3:]
    
    # def unscented_transform(x, P, lam=2.):
    #     n = x.shape[0]
    #     sigma_points = np.zeros((2*n+1, n))
    #     weights = np.zeros((2*n+1, ))

    #     sigma_points[0] = x
    #     weights[0] = lam / (lam + n)
    #     cov = sqrtm((lam + n) * P)
    #     for i in range(n):
    #         sigma_points[i+1] = sigma_points[0] + cov[:, i]
    #         weights[i+1] = 1 / (2 * (lam + n))
    #         sigma_points[n+i+1] = sigma_points[0] - cov[:, i]
    #         weights[n+i+1] = 1 / (2 * (lam + n))
    #     return sigma_points, weights
    
    def attitude_est_UKF(self, att_meas, input):
        sigma_points, weights = unsented_transform(self.x_pf, self.P)
        sigma_points = np.array([self.f_disc(sigma_point,  input) for sigma_point in sigma_points])
        x_est, P = inverse(sigma_points, weights)
        P = P + 1e-3*np.eye(6)

        # Update step
        sigma_points, weights, = unsented_transform(x_est, P)
        sigma_ys = np.array([self.g_disc(sigma_point) for sigma_point in sigma_points])
        y_hat, Sigma_y = inverse(sigma_ys, weights)
        Sigma_y = Sigma_y + self.R
        Sigma_xy = np.sum(np.array([w * np.outer(sigma_point - x_est, sigma_y - y_hat) for w, sigma_point, sigma_y in zip(weights, sigma_points, sigma_ys)]), axis=0)

        x_est = x_est + Sigma_xy @ np.linalg.inv(Sigma_y) @ (att_meas[3:] - y_hat)
        P = P - Sigma_xy @ np.linalg.inv(Sigma_y) @ Sigma_xy.T
        self.x_pf = x_est
        self.P = P
        return x_est