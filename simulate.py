from flying_sim.drone import Drone
from flying_sim.controllers import AttitudeController, ControlAllocation, PositionController
from flying_sim.trajectory import Trajectory
from flying_sim.estimation import Sensors, Estimation
from configs.config import Config
from scipy.spatial.transform import Rotation

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import time

matplotlib.use('TkAgg')


class Simulate:
    def __init__(self, trajectory_type="train", wind="no-wind"):
        self.config = Config()
        self.trajectory_type = trajectory_type
        self.wind = wind == "wind"

        # Initialize drone, controllers and trajectory
        self.drone = Drone(self.config)
        self.position_controller = PositionController(self.drone)
        self.attitude_controller = AttitudeController(self.drone)
        self.control_allocation = ControlAllocation(self.drone)
        self.trajectory = Trajectory(self.config)
        self.sensors = Sensors(self.drone)
        self.estimation = Estimation(self.drone)

        self.reset()

    def reset(self):
        # Reset drone
        self.drone.reset()

        # Logging signals
        self.states = self.drone.state
        self.aux_states = np.array([0, 0, 0, 0, 0, 0])
        self.inputs = np.array([0, 0, 0, 0])
        self.angular_vel_ref = np.array([0, 0, 0])
        self.attitude_ref = np.array([[0, 0, 0]])
        self.acceleration_ref = np.array([0, 0, 0])
        self.acceleration_des = np.array([0, 0, 0])
        self.velocity_ref = np.array([0, 0, 0])
        self.position_ref = np.array([0, 0, 0])
        self.measurements = np.zeros((15, ))
        self.state_estimates = np.zeros((12, ))
        self.estimate_covariance = np.zeros((1, ))
        self.time = [0]

    def simulate(self):
        start_time = time.time() # Runtime timer
        if self.trajectory_type == "random":
            self.trajectory.random_spline_trajectory()
        elif self.trajectory_type == "hover":
            self.trajectory.hover_trajectory()

        # Initialize time and control inputs
        control_moment = np.zeros((3,))
        thrust_des = -self.drone.m * self.drone.g

        # Simulation loop
        while self.time[-1] < self.config.traj_config.tf * 1.1:
            # Reference trajectory
            pos_ref = self.trajectory.position_ref(self.time[-1])
            vel_ref = self.trajectory.velocity_ref(self.time[-1])
            acc_ref = self.trajectory.acceleration_ref(self.time[-1])

            # Position controller
            des_lin_acc = self.position_controller.get_desired_lin_acc(self.estimation.position, self.estimation.velocity_e, self.drone.lin_acc, pos_ref, vel_ref, acc_ref)
            att_des, thrust_des = self.position_controller.get_desired_attitude(self.estimation.attitude, thrust_des, self.drone.lin_acc, des_lin_acc)
            yaw_des = 0     # np.arctan2(drone.state[10], drone.state[9])
            des_attitude = np.array([att_des[0], att_des[1], yaw_des])        # Desired attitude

            # Attitude controller
            des_angular_vel = self.attitude_controller.get_des_angular_vel(self.estimation.attitude, des_attitude)
            control_moment = self.attitude_controller.get_control_moment(control_moment, self.estimation.angular_velocity, des_angular_vel)
            control_input = self.control_allocation.get_control_input(control_moment, thrust_des)

            # Time step for drone
            self.drone.step(control_input, wind=self.wind)

            # Acquire sensor measurements
            sensor_measurements = self.sensors.sensor_measurements(control_input)

            # Update state estimate
            self.estimation.estimate_state(sensor_measurements, control_input)

            # Log states and time
            self.states = np.vstack((self.states, self.drone.state.copy()))
            self.aux_states = np.vstack((self.aux_states, np.hstack((self.drone.velocity_e, self.drone.lin_acc))))
            self.inputs = np.vstack((self.inputs, control_input))
            self.angular_vel_ref = np.vstack((self.angular_vel_ref, des_angular_vel))
            self.attitude_ref = np.vstack((self.attitude_ref, des_attitude))
            self.acceleration_des = np.vstack((self.acceleration_des, des_lin_acc))
            self.acceleration_ref = np.vstack((self.acceleration_ref, acc_ref))
            self.velocity_ref = np.vstack((self.velocity_ref, vel_ref))
            self.position_ref = np.vstack((self.position_ref, pos_ref))
            self.measurements = np.vstack((self.measurements, sensor_measurements))
            self.state_estimates = np.vstack((self.state_estimates, self.estimation.state_estimate))
            self.estimate_covariance = np.vstack((self.estimate_covariance, np.linalg.norm(np.sqrt(np.diag(self.estimation.mkf.cov)))))
            self.time.append(self.time[-1] + self.drone.dt)

            # Termination condition
            if np.linalg.norm(self.trajectory.waypoints[-1] - self.drone.position) < 0.5:
                print(f"Drone reach last waypoint {self.time[-1]} sec.")
                break

        print("Runtime: %s seconds" % (time.time() - start_time)) # Print runtime

    def evaluate(self, n=100):
        max_control_dev = []
        rmse_control_pos = []
        rmse_estimate_att = []
        max_vel = []
        rms_vel = []
        max_estimate_att = []
        for i in range(n):
            self.simulate()
            # Control Position Error
            pos_dev = np.linalg.norm(self.states[:, 6:9] - self.position_ref, axis=1)
            max_control_dev.append(np.max(pos_dev))
            rmse_control_pos.append(np.sqrt((pos_dev ** 2).mean()))

            # Velocity Quantification
            velocity = np.linalg.norm(self.aux_states[:, 0:3], axis=1)
            max_vel.append(np.max(velocity))
            rms_vel.append(np.sqrt((velocity ** 2).mean()))

            # Attitude Estimation Error
            att_est_dev = []
            for att_true, att_est in zip(self.states[:, :3], self.state_estimates[:, :3]):
                R1 = self.drone.rotationBodytoEarth(att_true).T         # inertial to true
                R2 = self.drone.rotationBodytoEarth(att_est)            # estimate to inertial
                Rerr = R1 @ R2
                att_est_dev.append(np.linalg.norm(Rotation.from_matrix(Rerr).as_rotvec()) * 180 / np.pi)
            max_estimate_att.append(np.max(att_est_dev))
            rmse_estimate_att.append(np.sqrt((np.array(att_est_dev) ** 2).mean()))

            self.reset()
            print("Simulation Count: ", i+1)

        print("============================================")
        print("max(||p - p_ref||) :", np.array(max_control_dev).mean())
        print("RMSE(||p - p_ref||) :", np.array(rmse_control_pos).mean())
        print(r"max(||$\eta_{true}$ - $\eta_{est}$||) :", np.array(max_estimate_att).mean())
        print(r"RMSE(||$\eta_{true}$ - $\eta_{est}$||) :", np.array(rmse_estimate_att).mean())
        print("max(||v||) :", np.array(max_vel).mean())
        print("RMS(||v||) :", np.array(rms_vel).mean())
        print("============================================")

    def plot(self):
        self.simulate()
        # np.savetxt("data/position_disturbance_conventional.csv", self.states[:, 6:9])

        # Plot states
        fig1, ax = plt.subplots(3, 2)

        ax[0, 0].plot(self.time, self.states[:, 0] * 180 / np.pi, label="state")
        ax[0, 0].plot(self.time, self.attitude_ref[:, 0] * 180 / np.pi, label="reference")
        ax[0, 0].plot(self.time, self.state_estimates[:, 0] * 180 / np.pi, label="estimate")
        ax[0, 0].set_ylabel("Roll Angle [deg]")
        ax[0, 0].grid()

        ax[1, 0].plot(self.time, self.states[:, 1] * 180 / np.pi, self.time, self.attitude_ref[:, 1] * 180 / np.pi)
        ax[1, 0].plot(self.time, self.state_estimates[:, 1] * 180 / np.pi)
        ax[1, 0].set_ylabel("Pitch Angle [deg]")
        ax[1, 0].grid()

        ax[2, 0].plot(self.time, self.states[:, 2] * 180 / np.pi, self.time, self.attitude_ref[:, 2] * 180 / np.pi)
        ax[2, 0].plot(self.time, self.state_estimates[:, 2] * 180 / np.pi)
        ax[2, 0].set_ylabel("Yaw Angle [deg]")
        ax[2, 0].grid()

        ax[0, 1].plot(self.time, self.states[:, 3] * 180 / np.pi, self.time, self.angular_vel_ref[:, 0] * 180 / np.pi)
        ax[0, 1].plot(self.time, self.state_estimates[:, 3] * 180 / np.pi)
        ax[0, 1].set_ylabel("Roll Rate [deg/s]")
        ax[0, 1].grid()

        ax[1, 1].plot(self.time, self.states[:, 4] * 180 / np.pi, self.time, self.angular_vel_ref[:, 1] * 180 / np.pi)
        ax[1, 1].plot(self.time, self.state_estimates[:, 4] * 180 / np.pi)
        ax[1, 1].set_ylabel("Pitch Rate [deg/s]")
        ax[1, 1].grid()

        ax[2, 1].plot(self.time, self.states[:, 5] * 180 / np.pi, self.time, self.angular_vel_ref[:, 2] * 180 / np.pi)
        ax[2, 1].plot(self.time, self.state_estimates[:, 5] * 180 / np.pi)
        ax[2, 1].set_ylabel("Yaw Rate [deg/s]")
        ax[2, 1].grid()

        ax[0, 0].legend()
        plt.tight_layout()


        fig2, ax = plt.subplots(3, 3)
        ax[0, 0].plot(self.time, self.states[:, 6], label='True States')
        ax[0, 0].plot(self.time, self.position_ref[:, 0], label='Reference')
        ax[0, 0].plot(self.time, self.state_estimates[:, 6], label='Estimated States')
        ax[0, 0].scatter(self.trajectory.time, self.trajectory.waypoints[:, 0], label='Waypoints')
        ax[0, 0].set_ylabel("X-position [m]")
        ax[0, 0].grid()

        ax[1, 0].plot(self.time, self.states[:, 7], self.time, self.position_ref[:, 1])
        ax[1, 0].plot(self.time, self.state_estimates[:, 7])
        ax[1, 0].scatter(self.trajectory.time, self.trajectory.waypoints[:, 1])
        ax[1, 0].set_ylabel("Y-position [m]")
        ax[1, 0].grid()

        ax[2, 0].plot(self.time, self.states[:, 8], self.time, self.position_ref[:, 2])
        ax[2, 0].plot(self.time, self.state_estimates[:, 8])
        ax[2, 0].scatter(self.trajectory.time, self.trajectory.waypoints[:, 2])
        ax[2, 0].set_ylabel("Z-position [m]")
        ax[2, 0].set_xlabel("Time [s]")
        ax[2, 0].grid()

        ax[0, 1].plot(self.time, self.aux_states[:, 0], self.time, self.velocity_ref[:, 0])
        ax[0, 1].plot(self.time, self.state_estimates[:, 9])
        ax[0, 1].set_ylabel("X-velocity [m/s]")
        ax[0, 1].grid()

        ax[1, 1].plot(self.time, self.aux_states[:, 1], self.time, self.velocity_ref[:, 1])
        ax[1, 1].plot(self.time, self.state_estimates[:, 10])
        ax[1, 1].set_ylabel("Y-velocity [m/s]")
        ax[1, 1].grid()

        ax[2, 1].plot(self.time, self.aux_states[:, 2], self.time, self.velocity_ref[:, 2])
        ax[2, 1].plot(self.time, self.state_estimates[:, 11])
        ax[2, 1].set_ylabel("Z-velocity [m/s]")
        ax[2, 1].set_xlabel("Time [s]")
        ax[2, 1].grid()

        ax[0, 2].plot(self.time, self.aux_states[:, 3], label="State")
        ax[0, 2].plot(self.time, self.acceleration_ref[:, 0], label="Ref")
        ax[0, 2].plot(self.time, self.acceleration_des[:, 0], label="Des")
        ax[0, 2].set_ylabel("X-acceleration [m/s2]")
        ax[0, 2].grid()

        ax[1, 2].plot(self.time, self.aux_states[:, 4], self.time, self.acceleration_ref[:, 1], self.time, self.acceleration_des[:, 1])
        ax[1, 2].set_ylabel("Y-acceleration [m/s2]")
        ax[1, 2].grid()

        ax[2, 2].plot(self.time, self.aux_states[:, 5], self.time, self.acceleration_ref[:, 2], self.time, self.acceleration_des[:, 2])
        ax[2, 2].set_ylabel("Z-acceleration [m/s2]")
        ax[2, 2].set_xlabel("Time [s]")
        ax[2, 2].grid()

        plt.tight_layout()

        # Plot inputs
        fig3, ax = plt.subplots(4, 1)

        ax[0].plot(self.time, np.sqrt(np.abs(self.inputs[:, 0])) * 60 / (2 * np.pi))
        ax[0].plot(self.time, self.drone.max_rotor_speed * np.ones(len(self.time),) * 60 / (2 * np.pi))
        ax[0].set_ylabel("rotational velocity 1 [rpm]")
        ax[0].grid()

        ax[1].plot(self.time, np.sqrt(np.abs(self.inputs[:, 1])) * 60 / (2 * np.pi))
        ax[1].set_ylabel("rotational velocity 2 [rad/s]")
        ax[1].grid()

        ax[2].plot(self.time, np.sqrt(np.abs(self.inputs[:, 2])) * 60 / (2 * np.pi))
        ax[2].set_ylabel("rotational velocity 3 [rad/s]")
        ax[2].grid()

        ax[3].plot(self.time, np.sqrt(np.abs(self.inputs[:, 3])) * 60 / (2 * np.pi))
        ax[3].set_ylabel("rotational velocity 4 [rad/s]")
        ax[3].grid()
        plt.tight_layout()

        # Plot 3d position trajectory
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(self.states[:, 6], self.states[:, 7], self.states[:, 8])
        ax.plot(self.position_ref[:, 0], self.position_ref[:, 1], self.position_ref[:, 2])
        ax.scatter(self.trajectory.waypoints[:, 0], self.trajectory.waypoints[:, 1], self.trajectory.waypoints[:, 2])
        ax.set_xlabel("X-position [m]")
        ax.set_ylabel("Y-position [m]")
        ax.set_zlabel("Z-position [m]")

        # Plot attitude estimation performance
        att_est_dev_angle = np.zeros((len(self.states), ))
        att_est_dev_euler = np.zeros(self.states[:, :3].shape)
        for i, (att_true, att_est) in enumerate(zip(self.states[:, :3], self.state_estimates[:, :3])):
            R1 = self.drone.rotationBodytoEarth(att_true).T  # inertial to true
            R2 = self.drone.rotationBodytoEarth(att_est)  # estimate to inertial
            Rerr = R1 @ R2
            att_est_dev_angle[i] = np.linalg.norm(Rotation.from_matrix(Rerr).as_rotvec())
            att_est_dev_euler[i, :] = Rotation.from_matrix(Rerr).as_euler('xyz')

        fig5, ax = plt.subplots(3, 1)
        ax[0].plot(self.time, att_est_dev_euler[:, 0] * 180 / np.pi)
        ax[0].set_ylabel("Roll Angle Error [deg]")
        ax[0].grid()

        ax[1].plot(self.time, att_est_dev_euler[:, 1] * 180 / np.pi)
        ax[1].set_ylabel("Pitch Angle Error [deg]")
        ax[1].grid()

        ax[2].plot(self.time, att_est_dev_euler[:, 2] * 180 / np.pi)
        ax[2].set_ylabel("Yaw Angle Error [deg]")
        ax[2].set_xlabel("Time [s]")
        ax[2].grid()

        fig6 = plt.figure()
        plt.plot(self.time[1:], att_est_dev_angle[1:] * 180 / np.pi, label='Estimation Error')
        plt.plot(self.time[1:], self.estimate_covariance[1:] * 180 / np.pi, label='Uncertainty')
        plt.plot(self.time[1:], -self.estimate_covariance[1:] * 180 / np.pi)
        plt.ylabel("Axis-Angle Error [deg]")
        plt.grid()

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Simulation',
        description='This program simulates the trajectory controller, either summarizing some metrics or plotting results',
        epilog='Text at the bottom of help')
    parser.add_argument("--action", default="plot", choices=["plot", "metrics"])
    parser.add_argument('--trajectory', default="random", choices=["figure-8", "random", "hover"])
    parser.add_argument('--wind_condition', default="no-wind", choices=["no-wind", "wind"])
    parser.add_argument("--num_eval", default=100, type=int)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()

    if type(args.seed) is int:
        print(f"Set seed to {args.seed}")
        np.random.seed(args.seed)

    # Run simulations
    simulate = Simulate(trajectory_type=args.trajectory, wind=args.wind_condition)
    if args.action == "plot":
        print(f"Plot simulation results on {args.trajectory} trajectory.")
        simulate.plot()
    elif args.action == "metrics":
        print(f"Calculate evaluation metrics on {args.trajectory} trajectory over {args.num_eval} simulations")
        simulate.evaluate(args.num_eval)
