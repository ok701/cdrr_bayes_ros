#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

import numpy as np
import argparse
import time

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', etc., depending on your setup
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm


class BayesOptHWNode(Node):
    """
    A ROS2 Node that demonstrates a one-run-at-a-time Bayesian optimization approach
    with real-time Matplotlib plotting of the GP model and Expected Improvement.

    Changes from original:
      - Using RBF instead of Matern.
      - No scaling of error or force (treat mm as is).
    """

    def __init__(self, freq=10.0, show_plot=True, max_runs=10):
        super().__init__('bayes_opt_hw_node')

        # ---------------- Parameters ----------------
        self.freq = freq               # frequency of the main loop
        self.show_plot = show_plot     # whether to show GP plots
        self.max_runs = max_runs       # how many runs we want to do (optional)

        # ---------------- Publishers & Subscribers ----------------
        self.ee_error_sub = self.create_subscription(
            Int32,
            '/ee_error',
            self.ee_error_callback,
            10
        )
        self.trigger_sub = self.create_subscription(
            Int32,
            '/trigger',
            self.trigger_callback,
            10
        )
        self.manipulated_var_pub = self.create_publisher(
            Int32,
            '/manipulated_var',
            10
        )

        # ------------- Internal State for Triggers & Data -------------
        self.trigger_state = 0         # last known trigger value
        self.collecting_data = False   # True if capturing errors
        self.ee_error_list = []        # store errors for each run (no scaling)

        # ------------- Storage for the Bayesian Optimization -------------
        self.X = []    # list of tested forces (float)
        self.y = []    # list of costs for those forces (float)
        self.run_count = 0

        # Default cable force = 0 as our first test
        self.next_cable_force = 0.0

        # ------------- Set up a Gaussian Process Regressor -------------
        # Now using RBF instead of Matern, no scaling in the data
        kernel = ConstantKernel(1.0) * RBF(length_scale=10.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-3,
            n_restarts_optimizer=5,
            random_state=42
        )

        # ------------- Create a timer to spin at 'freq' Hz -------------
        period = 1.0 / self.freq
        self.timer = self.create_timer(period, self.main_loop_callback)

        self.get_logger().info(
            f"BayesOptHWNode started (freq={self.freq} Hz, show_plot={self.show_plot}, max_runs={self.max_runs})"
        )

        # Immediately publish the “initial” cable force as a starting point
        self.publish_cable_force(self.next_cable_force)

        # ------------- If plotting, set up Matplotlib -------------
        if self.show_plot:
            plt.ion()  # enable interactive mode
            self.fig, (self.ax_gp, self.ax_acq) = plt.subplots(2, 1, figsize=(6, 8))
            self.fig.tight_layout(pad=3.0)
            self.fig.show()

    # ------------------------------------------------------------------
    #                           Main loop
    # ------------------------------------------------------------------
    def main_loop_callback(self):
        """
        Periodic tasks can be handled here. Typically, no-op for this example.
        """
        pass

    # ------------------------------------------------------------------
    #                       Subscriber Callbacks
    # ------------------------------------------------------------------
    def ee_error_callback(self, msg: Int32):
        """
        /ee_error (std_msgs/Int32).
        No scaling; treat as mm (or any unit you like).
        """
        error_val = float(msg.data)
        if self.collecting_data:
            self.ee_error_list.append(error_val)

    def trigger_callback(self, msg: Int32):
        """
        /trigger (std_msgs/Int32).
        - 0->1: start recording data
        - 1->0: stop recording, compute cost, do BO iteration, propose next force
        """
        new_trigger = msg.data

        # 0->1: start a new run
        if new_trigger == 1 and self.trigger_state == 0:
            self.start_recording()

        # 1->0: finish the run, do Bayesian optimization
        elif new_trigger == 0 and self.trigger_state == 1:
            self.stop_recording_and_optimize()

        self.trigger_state = new_trigger

    # ------------------------------------------------------------------
    #                       Data Recording
    # ------------------------------------------------------------------
    def start_recording(self):
        """Start collecting ee_error for a new run."""
        self.get_logger().info(f"*** Trigger 0->1: Starting run #{self.run_count + 1} data collection.")
        self.collecting_data = True
        self.ee_error_list = []

    def stop_recording_and_optimize(self):
        """
        Stop collecting, compute cost, store the result,
        run Bayesian optimization, propose next force, publish it.
        """
        self.get_logger().info(f"*** Trigger 1->0: Ending run #{self.run_count + 1}. Processing data.")
        self.collecting_data = False


        tested_force = self.next_cable_force

        # 1) Compute cost from the recorded data
        cost_val = self.compute_cost_from_data(self.ee_error_list, tested_force)
        n_samples = len(self.ee_error_list)
        self.get_logger().info(
            f"   - Collected {n_samples} error samples. Cost = {cost_val:.5f}"
        )

        # 2) Store the (force, cost) pair
        # tested_force = self.next_cable_force
        self.X.append(tested_force)
        self.y.append(cost_val)

        # Increment run count
        self.run_count += 1

        # 3) If we haven't reached max_runs, do a Bayesian optimization iteration
        if self.run_count < self.max_runs:
            self.get_logger().info(f"   - Running Bayesian optimization iteration #{self.run_count} ...")

            # Fit GP to all data so far
            X_data = np.array(self.X).reshape(-1, 1)
            y_data = np.array(self.y)
            self.gp.fit(X_data, y_data)

            # Optional: update the live plot
            if self.show_plot:
                self.update_plot(X_data, y_data)

            # Propose next cable force
            self.next_cable_force = self.propose_next_force(X_data, y_data)
            self.get_logger().info(f"   - Proposed next cable force: {self.next_cable_force:.2f}")

            # 4) Publish new cable force for next run
            self.publish_cable_force(self.next_cable_force)
        else:
            # If we've done enough runs, we can finalize or publish a default (e.g., 0).
            self.get_logger().info(f"Reached max_runs={self.max_runs}. No further optimization.")
            self.publish_cable_force(0.0)

    # ------------------------------------------------------------------
    #                       Cost Computation
    # ------------------------------------------------------------------
    def compute_cost_from_data(self, error_list, cable_force):
        """
        Example: cost = mean squared error (MSE/1000) + cable_force
        Adjust to your preference (e.g. absolute force, linear error, etc.).
        """
        if len(error_list) == 0:
            return 0.0

        # Example using MSE (scaled by 1/1000) plus raw cable_force
        mse_part = float(np.mean(np.square(error_list)) / 100.0)
        weight = 0.5
        return weight*mse_part + (1 - weight)*cable_force

    # ------------------------------------------------------------------
    #                 Bayesian Optimization Helpers
    # ------------------------------------------------------------------
    def propose_next_force(self, X_data, y_data):
        """
        1) Evaluate EI on a discrete grid of forces (e.g., -40 to 40).
        2) Return the force that maximizes EI.
        """
        candidate_forces = np.linspace(-40, 40, 101).reshape(-1, 1)
        ei_values = self.expected_improvement(candidate_forces, X_data, y_data, self.gp)
        best_idx = np.argmax(ei_values)
        best_candidate = candidate_forces[best_idx, 0]
        return float(best_candidate)

    def expected_improvement(self, X_new, X, y, model, xi=0.01):
        """
        Compute the Expected Improvement at points X_new.
        We assume we're *minimizing* y, so the best is y_min.
        """
        mu, sigma = model.predict(X_new, return_std=True)
        y_min = np.min(y)
        with np.errstate(divide='warn'):
            improvement = (y_min - mu) - xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    # ------------------------------------------------------------------
    #                     Live Plotting Methods
    # ------------------------------------------------------------------
    def update_plot(self, X_data, y_data):
        """
        Update the GP mean/std and EI plots in real time.
        """
        if not self.show_plot:
            return  # no-op if disabled

        self.ax_gp.clear()
        self.ax_acq.clear()

        # 1) GP mean ± std
        X_plot = np.linspace(-40, 40, 200).reshape(-1, 1)
        mu, std = self.gp.predict(X_plot, return_std=True)

        self.ax_gp.plot(X_plot, mu, 'b-', label='GP Mean')
        self.ax_gp.fill_between(
            X_plot.ravel(), mu - std, mu + std,
            alpha=0.2, color='blue', label='GP ±1σ'
        )
        # Observations
        self.ax_gp.scatter(X_data, y_data, c='r', label='Data')

        self.ax_gp.set_title("Gaussian Process Model of the Cost")
        self.ax_gp.set_xlabel("Cable Force (N)")
        self.ax_gp.set_ylabel("Cost (lower is better)")
        self.ax_gp.legend()
        self.ax_gp.grid(True)

        # 2) EI (acquisition function)
        ei_values = self.expected_improvement(X_plot, X_data, y_data, self.gp)
        self.ax_acq.plot(X_plot, ei_values, 'g-', label='Expected Improvement')
        self.ax_acq.set_title("Acquisition Function (EI)")
        self.ax_acq.set_xlabel("Cable Force (N)")
        self.ax_acq.set_ylabel("EI")
        self.ax_acq.legend()
        self.ax_acq.grid(True)

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.2)  # small pause to allow update

    # ------------------------------------------------------------------
    #              Helper: Publish Cable Force (no scaling)
    # ------------------------------------------------------------------
    def publish_cable_force(self, force_value):
        """
        Publish the cable force as an integer to /manipulated_var with no scaling.
        """
        out_msg = Int32()
        # Force is cast to int (rounding can be refined if needed)
        out_msg.data = int(force_value)
        self.manipulated_var_pub.publish(out_msg)
        self.get_logger().info(
            f"Published new force={force_value:.2f} to /manipulated_var."
        )


# ----------------------------------------------------------------------
#                          Main Entry Point
# ----------------------------------------------------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="Bayesian Opt HW Multi-Run Demo Node")
    parser.add_argument("--freq", type=float, default=10.0, help="Loop frequency (Hz)")
    parser.add_argument("--show-plot", action='store_true', help="Show GP plot after each iteration")
    parser.add_argument("--max-runs", type=int, default=10, help="How many runs to do before stopping")
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = BayesOptHWNode(
        freq=parsed_args.freq,
        show_plot=parsed_args.show_plot,
        max_runs=parsed_args.max_runs
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down BayesOptHWNode.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Int32

# import numpy as np
# import argparse
# import time

# import matplotlib
# matplotlib.use('TkAgg')  # or 'Qt5Agg', etc.
# import matplotlib.pyplot as plt

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel
# from scipy.stats import norm


# class BayesOptHWNode(Node):
#     """
#     A ROS2 Node that demonstrates a one-run-at-a-time Bayesian optimization approach
#     with real-time Matplotlib plotting of the GP model and Expected Improvement.
#     """

#     def __init__(self, freq=10.0, show_plot=True, max_runs=10):
#         super().__init__('bayes_opt_hw_node')

#         # ---------------- Parameters ----------------
#         self.freq = freq
#         self.show_plot = show_plot
#         self.max_runs = max_runs

#         # ---------------- Publishers & Subscribers ----------------
#         self.ee_error_sub = self.create_subscription(
#             Int32,
#             '/ee_error',
#             self.ee_error_callback,
#             10
#         )
#         self.trigger_sub = self.create_subscription(
#             Int32,
#             '/trigger',
#             self.trigger_callback,
#             10
#         )
#         self.manipulated_var_pub = self.create_publisher(
#             Int32,
#             '/manipulated_var',
#             10
#         )

#         # ------------- Internal State -------------
#         self.trigger_state = 0
#         self.collecting_data = False
#         self.ee_error_list = []

#         # ------------- Bayesian Optimization Data -------------
#         self.X = []   # forces tested
#         self.y = []   # cost results
#         self.run_count = 0

#         # ---------------- Initial Forces ----------------
#         # First 3 trials: [15 N, 0 N, -15 N], in that order
#         self.initial_forces = [15, 0, -15]

#         # Start with 15 N for the very first run
#         self.next_cable_force = 15.0

#         # ------------- Set up Gaussian Process -------------
#         kernel = ConstantKernel(1.0) * RBF(length_scale=10.0)
#         self.gp = GaussianProcessRegressor(
#             kernel=kernel,
#             alpha=1e-3,
#             n_restarts_optimizer=5,
#             random_state=42
#         )

#         # ------------- Timer -------------
#         period = 1.0 / self.freq
#         self.timer = self.create_timer(period, self.main_loop_callback)

#         self.get_logger().info(
#             f"BayesOptHWNode started (freq={self.freq} Hz, show_plot={self.show_plot}, max_runs={self.max_runs})"
#         )

#         # Publish the first force (15 N) immediately
#         self.publish_cable_force(self.next_cable_force)

#         # ------------- If plotting, set up Matplotlib -------------
#         if self.show_plot:
#             plt.ion()
#             self.fig, (self.ax_gp, self.ax_acq) = plt.subplots(2, 1, figsize=(6, 8))
#             self.fig.tight_layout(pad=3.0)
#             self.fig.show()

#     # ------------------------------------------------------------------
#     #                           Main loop
#     # ------------------------------------------------------------------
#     def main_loop_callback(self):
#         pass

#     # ------------------------------------------------------------------
#     #                       Subscriber Callbacks
#     # ------------------------------------------------------------------
#     def ee_error_callback(self, msg: Int32):
#         """
#         /ee_error (std_msgs/Int32).
#         """
#         if self.collecting_data:
#             self.ee_error_list.append(float(msg.data))

#     def trigger_callback(self, msg: Int32):
#         """
#         /trigger (std_msgs/Int32).
#         - 0->1: start data collection
#         - 1->0: stop, compute cost, do BO iteration (unless in first 3 runs)
#         """
#         new_trigger = msg.data

#         # 0->1: start a new run
#         if new_trigger == 1 and self.trigger_state == 0:
#             self.start_recording()

#         # 1->0: finish the run, do Bayesian optimization (or pick from initial forces)
#         elif new_trigger == 0 and self.trigger_state == 1:
#             self.stop_recording_and_optimize()

#         self.trigger_state = new_trigger

#     # ------------------------------------------------------------------
#     #                       Data Recording
#     # ------------------------------------------------------------------
#     def start_recording(self):
#         self.get_logger().info(f"*** Trigger 0->1: Starting run #{self.run_count + 1}.")
#         self.collecting_data = True
#         self.ee_error_list = []

#     def stop_recording_and_optimize(self):
#         self.get_logger().info(f"*** Trigger 1->0: Ending run #{self.run_count + 1}.")
#         self.collecting_data = False

#         # Compute cost for this run
#         tested_force = self.next_cable_force
#         cost_val = self.compute_cost_from_data(self.ee_error_list, tested_force)
#         n_samples = len(self.ee_error_list)

#         self.get_logger().info(
#             f"   - Collected {n_samples} samples | Force={tested_force:.1f} | Cost={cost_val:.5f}"
#         )

#         # Store data and increment run count
#         self.X.append(tested_force)
#         self.y.append(cost_val)
#         self.run_count += 1

#         # If we haven't reached max_runs, pick the next force
#         if self.run_count < self.max_runs:
#             self.get_logger().info(f"   - Deciding next force (run #{self.run_count + 1}) ...")

#             # ----------------- First 3 runs: use initial forces -----------------
#             if self.run_count < 3:
#                 # Use the next fixed force from [15, 0, -15]
#                 self.next_cable_force = float(self.initial_forces[self.run_count])
#                 self.get_logger().info(
#                     f"     * Using fixed force for run #{self.run_count + 1}: {self.next_cable_force:.2f}"
#                 )
#             else:
#                 # ----------------- Subsequent runs: Bayesian Optimization -----------------
#                 X_data = np.array(self.X).reshape(-1, 1)
#                 y_data = np.array(self.y)
#                 self.gp.fit(X_data, y_data)

#                 # Update plot if enabled
#                 if self.show_plot:
#                     self.update_plot(X_data, y_data)

#                 # Propose next cable force from EI
#                 self.next_cable_force = self.propose_next_force(X_data, y_data)
#                 self.get_logger().info(f"     * BO proposed force: {self.next_cable_force:.2f}")

#             # Publish new cable force
#             self.publish_cable_force(self.next_cable_force)
#         else:
#             self.get_logger().info(f"Reached max_runs={self.max_runs}. No further optimization.")
#             self.publish_cable_force(0.0)

#     # ------------------------------------------------------------------
#     #                       Cost Computation
#     # ------------------------------------------------------------------
#     def compute_cost_from_data(self, error_list, cable_force):
#         """
#         Example: cost = (MSE/100) * 0.5 + cable_force * 0.5
#         Adjust weighting or formula as you prefer.
#         """
#         if len(error_list) == 0:
#             return 0.0

#         mse_part = np.mean(np.square(error_list)) / 100.0
#         weight = 0.5
#         return weight * mse_part + (1 - weight) * cable_force

#     # ------------------------------------------------------------------
#     #                 Bayesian Optimization Helpers
#     # ------------------------------------------------------------------
#     def propose_next_force(self, X_data, y_data):
#         """
#         1) Evaluate EI on a discrete grid of forces (e.g., -40 to 40).
#         2) Return the force that maximizes EI.
#         """
#         candidate_forces = np.linspace(-40, 40, 101).reshape(-1, 1)
#         ei_values = self.expected_improvement(candidate_forces, X_data, y_data, self.gp)
#         best_idx = np.argmax(ei_values)
#         return float(candidate_forces[best_idx, 0])

#     def expected_improvement(self, X_new, X, y, model, xi=0.01):
#         """
#         Compute the Expected Improvement at points X_new.
#         Minimizing y, so best is y_min.
#         """
#         mu, sigma = model.predict(X_new, return_std=True)
#         y_min = np.min(y)
#         with np.errstate(divide='warn'):
#             improvement = (y_min - mu) - xi
#             Z = improvement / sigma
#             ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
#             ei[sigma == 0.0] = 0.0
#         return ei

#     # ------------------------------------------------------------------
#     #                     Live Plotting Methods
#     # ------------------------------------------------------------------
#     def update_plot(self, X_data, y_data):
#         if not self.show_plot:
#             return

#         self.ax_gp.clear()
#         self.ax_acq.clear()

#         # 1) GP mean ± std
#         X_plot = np.linspace(-40, 40, 200).reshape(-1, 1)
#         mu, std = self.gp.predict(X_plot, return_std=True)

#         self.ax_gp.plot(X_plot, mu, 'b-', label='GP Mean')
#         self.ax_gp.fill_between(
#             X_plot.ravel(), mu - std, mu + std,
#             alpha=0.2, color='blue', label='GP ±1σ'
#         )
#         # Observations
#         self.ax_gp.scatter(X_data, y_data, c='r', label='Data')

#         self.ax_gp.set_title("Gaussian Process Model of the Cost")
#         self.ax_gp.set_xlabel("Cable Force (N)")
#         self.ax_gp.set_ylabel("Cost (lower is better)")
#         self.ax_gp.legend()
#         self.ax_gp.grid(True)

#         # 2) EI (acquisition function)
#         ei_values = self.expected_improvement(X_plot, X_data, y_data, self.gp)
#         self.ax_acq.plot(X_plot, ei_values, 'g-', label='Expected Improvement')
#         self.ax_acq.set_title("Acquisition Function (EI)")
#         self.ax_acq.set_xlabel("Cable Force (N)")
#         self.ax_acq.set_ylabel("EI")
#         self.ax_acq.legend()
#         self.ax_acq.grid(True)

#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()
#         plt.pause(0.2)

#     # ------------------------------------------------------------------
#     #              Helper: Publish Cable Force
#     # ------------------------------------------------------------------
#     def publish_cable_force(self, force_value):
#         out_msg = Int32()
#         out_msg.data = int(force_value)
#         self.manipulated_var_pub.publish(out_msg)
#         self.get_logger().info(
#             f"Published new force={force_value:.2f} to /manipulated_var."
#         )


# def main(args=None):
#     parser = argparse.ArgumentParser(description="Bayesian Opt HW Multi-Run Demo Node")
#     parser.add_argument("--freq", type=float, default=10.0, help="Loop frequency (Hz)")
#     parser.add_argument("--show-plot", action='store_true', help="Show GP plot after each iteration")
#     parser.add_argument("--max-runs", type=int, default=10, help="How many runs to do before stopping")
#     parsed_args = parser.parse_args()

#     rclpy.init(args=args)
#     node = BayesOptHWNode(
#         freq=parsed_args.freq,
#         show_plot=parsed_args.show_plot,
#         max_runs=parsed_args.max_runs
#     )

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.get_logger().info("Shutting down BayesOptHWNode.")
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
