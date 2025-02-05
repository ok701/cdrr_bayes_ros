import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray

import numpy as np
import argparse

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', etc., depending on your setup
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm

import sys


class BayesOptHWNode(Node):
    """
    A ROS2 Node that demonstrates a one-run-at-a-time Bayesian optimization approach,
    with:
      1) Repeats per force (default = 3)
      2) Outlier exclusion
      3) Averaging valid costs before feeding to the GP
      4) Logging all trial costs and outliers.

    Subscribes:
      - /mes_pos (Int32)
      - /ref_pos (Int32)
      - /trigger (Int32)

    Publishes:
      - /ref_force (Int32): the proposed cable force
      - /score (Float32MultiArray): [strength_score, accuracy_score] at ~freq Hz
    """

    def __init__(self, freq=10.0, show_plot=True, max_runs=10):
        super().__init__('bayes_opt_hw_node')

        # ---------------- Parameters ----------------
        self.freq = freq
        self.show_plot = show_plot
        self.max_runs = max_runs

        # How many times to repeat the same force
        self.repeats_per_force = 3
        # Count how many times we've repeated the current force so far
        self.current_repeat_count = 0
        # Store costs for the current force (all repeats)
        self.current_force_costs = []

        # ---------------- Publishers & Subscribers ----------------
        self.mes_pos_sub = self.create_subscription(
            Int32,
            '/mes_pos',
            self.mes_pos_callback,
            10
        )

        self.ref_pos_sub = self.create_subscription(
            Int32,
            '/ref_pos',
            self.ref_pos_callback,
            10
        )

        self.trigger_sub = self.create_subscription(
            Int32,
            '/trigger',
            self.trigger_callback,
            10
        )

        self.ref_force_pub = self.create_publisher(
            Int32,
            '/ref_force',
            10
        )

        # Publisher for real-time scores (and final scores).
        self.score_pub = self.create_publisher(
            Float32MultiArray,
            '/score',
            10
        )

        # ------------- Internal State -------------
        self.trigger_state = 0
        self.collecting_data = False
        self.ee_error_list = []

        self.mes_pos = None
        self.ref_pos = None

        # For Bayesian Optimization:
        self.X = []
        self.y = []
        self.run_count = 0

        # First force
        self.next_cable_force = 0.0

        # ------------- Set up a Gaussian Process Regressor -------------
        kernel = ConstantKernel(1.0) * RBF(length_scale=50.0, length_scale_bounds=(10.0, 100.0))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=10,
            n_restarts_optimizer=5,
            random_state=42
        )

        # Domain for searching new forces
        self.lower_bound = -30.0
        self.upper_bound = 15.0

        # Keep track of the last final run’s scores (for real-time if no new data)
        self.last_run_strength = 0.0
        self.last_run_accuracy = 0.0

        # ------------- Timer -------------
        period = 1.0 / self.freq
        self.timer = self.create_timer(period, self.main_loop_callback)

        self.get_logger().info(
            f"BayesOptHWNode started (freq={self.freq} Hz, show_plot={self.show_plot}, max_runs={self.max_runs})"
        )

        # Publish the initial cable force
        self.publish_ref_force(self.next_cable_force)

        # ------------- Plotting (if enabled) -------------
        if self.show_plot:
            plt.ion()
            self.fig, (self.ax_gp, self.ax_acq) = plt.subplots(2, 1, figsize=(6, 8))
            self.fig.tight_layout(pad=3.0)
            self.fig.show()

    # ------------------------------------------------------------------
    #                           Main loop
    # ------------------------------------------------------------------
    def main_loop_callback(self):
        """
        Periodic tasks at self.freq (e.g., 10 Hz).
        We'll publish real-time strength_score and accuracy_score here.
        """
        strength_score, accuracy_score = self.compute_realtime_scores()
        self.publish_realtime_scores(strength_score, accuracy_score)

    # ------------------------------------------------------------------
    #                       Subscriber Callbacks
    # ------------------------------------------------------------------
    def mes_pos_callback(self, msg: Int32):
        self.mes_pos = float(msg.data)
        self.compute_and_store_error()

    def ref_pos_callback(self, msg: Int32):
        self.ref_pos = float(msg.data)
        self.compute_and_store_error()

    def trigger_callback(self, msg: Int32):
        new_trigger = msg.data

        if new_trigger == 1 and self.trigger_state == 0:
            # 0->1: start a new run
            self.start_recording()
        elif new_trigger == 0 and self.trigger_state == 1:
            # 1->0: finish the run, process data
            self.stop_recording_and_optimize()

        self.trigger_state = new_trigger

    # ------------------------------------------------------------------
    #                       Data Recording
    # ------------------------------------------------------------------
    def start_recording(self):
        self.get_logger().info(f"*** Trigger 0->1: Starting run #{self.run_count + 1} data collection.")
        self.collecting_data = True
        self.ee_error_list = []

    def stop_recording_and_optimize(self):
        self.get_logger().info(f"*** Trigger 1->0: Ending run #{self.run_count + 1}. Processing data.")
        self.collecting_data = False

        tested_force = self.next_cable_force

        # 1) Compute cost from this single run
        cost_val, strength_score, accuracy_score = self.compute_cost_from_data(self.ee_error_list, tested_force)
        self.last_run_strength = strength_score
        self.last_run_accuracy = accuracy_score

        n_samples = len(self.ee_error_list)
        self.get_logger().info(f"   - Collected {n_samples} error samples. Cost = {cost_val:.5f}")

        # 2) Store this run's cost
        self.current_force_costs.append(cost_val)

        # 3) Increase the repeat count
        self.current_repeat_count += 1
        self.get_logger().info(f"   - current_repeat_count = {self.current_repeat_count}/{self.repeats_per_force}")

        # 4) If we still need more repeats, just re-publish the SAME force
        if self.current_repeat_count < self.repeats_per_force:
            self.get_logger().info("   - Still repeating the same force; no new optimization step.")
            self.publish_ref_force(self.next_cable_force)
            return

        # ----------------------------------------------------------------
        # We have completed all repeats for this force
        # ----------------------------------------------------------------
        self.get_logger().info(f"=== Repeats complete for force={tested_force:.2f}. ===")

        # (a) Log **all trial costs** for these repeats
        all_cost_str = ", ".join(f"{c:.5f}" for c in self.current_force_costs)
        self.get_logger().info(f"   - All trial costs: [{all_cost_str}]")

        # (b) Remove outliers
        filtered_costs = self.remove_outliers(self.current_force_costs, k=1.3)

        # Identify outliers simply via set difference (note: duplicates can complicate this)
        outliers = list(set(self.current_force_costs) - set(filtered_costs))
        if len(outliers) > 0:
            outlier_str = ", ".join(f"{c:.5f}" for c in outliers)
            self.get_logger().warn(f"   - Outliers detected: [{outlier_str}]")
        else:
            self.get_logger().info("   - No outliers detected.")

        if len(filtered_costs) == 0:
            self.get_logger().warn("   - ALL repeats were outliers! Using raw data anyway.")
            filtered_costs = self.current_force_costs

        # (c) Average the remaining (valid) costs
        final_cost_for_gp = float(np.mean(filtered_costs))
        self.get_logger().info(f"   - Average cost after outlier removal: {final_cost_for_gp:.5f}")

        # Clear the temp lists for the next force
        self.current_repeat_count = 0
        self.current_force_costs = []

        # 5) Store (force, averaged cost) as a single data point in GP
        self.X.append(tested_force)
        self.y.append(final_cost_for_gp)

        # Publish the final scores from the last run
        self.publish_realtime_scores(self.last_run_strength, self.last_run_accuracy)

        # 6) Bayesian optimization iteration
        self.run_count += 1
        if self.run_count < self.max_runs:
            self.get_logger().info(f"   - Running Bayesian optimization iteration #{self.run_count} ...")
            X_data = np.array(self.X).reshape(-1, 1)
            y_data = np.array(self.y)
            self.gp.fit(X_data, y_data)

            # Update the plot if desired
            if self.show_plot:
                self.update_plot(X_data, y_data)

            # Propose next cable force
            self.next_cable_force = self.propose_next_force(X_data, y_data)
            self.get_logger().info(f"   - Proposed next cable force: {self.next_cable_force:.2f}")

            # Publish new force
            self.publish_ref_force(self.next_cable_force)
        else:
            self.get_logger().info(f"Reached max_runs={self.max_runs}. No further optimization.")
            self.publish_ref_force(0.0)

    # ------------------------------------------------------------------
    #             Compute and Store Error Each Time We Get Data
    # ------------------------------------------------------------------
    def compute_and_store_error(self):
        if self.collecting_data and (self.mes_pos is not None) and (self.ref_pos is not None):
            error_val = self.ref_pos - self.mes_pos
            self.ee_error_list.append(error_val)

    # ------------------------------------------------------------------
    #                       Cost Computation
    # ------------------------------------------------------------------
    def compute_cost_from_data(self, error_list, cable_force):
        """
        Returns (cost_val, strength_score, accuracy_score) 
        after a single run has ended.
        """
        if len(error_list) == 0:
            return 0.0, 0.0, 0.0

        # Example: accuracy_score
        accuracy_score = np.exp(-np.mean(np.square(error_list)) / 15000) * 100

        # Example: strength_score
        mapped_force = (100 / (self.lower_bound - self.upper_bound)) * (cable_force - self.upper_bound)
        strength_score = mapped_force

        # Example cost: combination of MSE part and cable_force
        mse_part = float(np.mean(np.square(error_list)) / 100)
        weight = 0.4
        cost_val = weight * mse_part + (1 - weight) * cable_force

        return cost_val, strength_score, accuracy_score

    # ------------------------------------------------------------------
    #                Real-Time Score Computation
    # ------------------------------------------------------------------
    def compute_realtime_scores(self):
        """
        Compute (strength_score, accuracy_score) in real time.
        If not collecting data or no data yet, return last final scores.
        """
        if not self.collecting_data or len(self.ee_error_list) == 0:
            return self.last_run_strength, self.last_run_accuracy

        # Real-time approximate => use the running mean error so far
        accuracy_score = np.exp(-np.mean(np.square(self.ee_error_list)) / 15000) * 100
        cable_force = self.next_cable_force
        mapped_force = (100 / (self.lower_bound - self.upper_bound)) * (cable_force - self.upper_bound)
        strength_score = mapped_force

        return strength_score, accuracy_score

    # ------------------------------------------------------------------
    #                     Publish Real-Time Scores
    # ------------------------------------------------------------------
    def publish_realtime_scores(self, strength_score, accuracy_score):
        msg = Float32MultiArray()
        msg.data = [strength_score, accuracy_score]
        self.score_pub.publish(msg)

    # ------------------------------------------------------------------
    #                   Remove Outliers from a List
    # ------------------------------------------------------------------
    def remove_outliers(self, cost_list, k=1):
        """
        Exclude values that are more than k standard deviations from the mean.
        Returns a filtered list of costs.
        """
        if len(cost_list) < 2:
            return cost_list  # with 0 or 1 item, can't detect outliers

        mean_val = np.mean(cost_list)
        std_val = np.std(cost_list)
        # self.get_logger().info(f"   ----------------------------------------------------- StdDev = {std_val:.5f}")

        if std_val == 0:
            # All costs are identical => no outliers by this criterion
            return cost_list

        filtered = []
        for c in cost_list:
            # self.get_logger().info(f"   ----------------------------------------------------- c - mean_val = {abs(c - mean_val):.5f}")
            # self.get_logger().info(f"   ----------------------------------------------------- k * std_val = {k * std_val:.5f}") 
            if abs(c - mean_val) <= k * std_val:
                filtered.append(c)          
            # self.get_logger().info(f"   ----------------------------------------------------- filtered = {filtered}")
        return filtered

    # ------------------------------------------------------------------
    #                Bayesian Optimization: Next Force
    # ------------------------------------------------------------------
    def propose_next_force(self, X_data, y_data):
        # Evaluate EI on a grid in [lower_bound, upper_bound]
        global_candidates = np.linspace(self.lower_bound, self.upper_bound, 200).reshape(-1, 1)
        ei_values = self.expected_improvement(global_candidates, X_data, y_data, self.gp, xi=0.001)
        
        # Find the global max of EI
        best_idx_global = np.argmax(ei_values)
        F_global_best = float(global_candidates[best_idx_global, 0])
        return F_global_best

    def expected_improvement(self, X_new, X, y, model, xi=0.01):
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
        if not self.show_plot:
            return

        self.ax_gp.clear()
        self.ax_acq.clear()

        X_plot = np.linspace(-40, 40, 200).reshape(-1, 1)
        mu, std = self.gp.predict(X_plot, return_std=True)

        # Plot GP mean & std
        self.ax_gp.plot(X_plot, mu, 'b-', label='GP Mean')
        self.ax_gp.fill_between(
            X_plot.ravel(), mu - std, mu + std,
            alpha=0.2, color='blue', label='GP ±1σ'
        )
        self.ax_gp.scatter(X_data, y_data, c='r', label='Data')
        self.ax_gp.set_title("Gaussian Process Model of the Cost")
        self.ax_gp.set_xlabel("Cable Force (N)")
        self.ax_gp.set_ylabel("Cost (lower is better)")
        self.ax_gp.legend()
        self.ax_gp.grid(True)

        # Plot acquisition
        ei_values = self.expected_improvement(X_plot, X_data, y_data, self.gp)
        self.ax_acq.plot(X_plot, ei_values, 'g-', label='Expected Improvement')
        self.ax_acq.set_title("Acquisition Function (EI)")
        self.ax_acq.set_xlabel("Cable Force (N)")
        self.ax_acq.set_ylabel("EI")
        self.ax_acq.legend()
        self.ax_acq.grid(True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.2)

    # ------------------------------------------------------------------
    #              Helper: Publish Cable Force
    # ------------------------------------------------------------------
    def publish_ref_force(self, force_value):
        out_msg = Int32()
        out_msg.data = int(force_value)
        self.ref_force_pub.publish(out_msg)
        self.get_logger().info(
            f"Published new force={force_value:.2f} to /ref_force."
        )

# ----------------------------------------------------------------------
#                          Main Entry Point
# ----------------------------------------------------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="Bayesian Opt HW Multi-Run Demo Node")
    parser.add_argument("--freq", type=float, default=10.0, help="Loop frequency (Hz)")
    parser.add_argument("--show-plot", action='store_true', help="Show GP plot after each iteration")
    parser.add_argument("--max-runs", type=int, default=10, help="How many runs (i.e. sets of repeats) to do")

    known_args, _ = parser.parse_known_args()
    rclpy.init(args=sys.argv)

    node = BayesOptHWNode(
        freq=known_args.freq,
        show_plot=known_args.show_plot,
        max_runs=known_args.max_runs
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
