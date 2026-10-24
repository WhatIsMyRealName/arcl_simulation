#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "robot_msgs/action/rock_centering.hpp"
#include "robot_msgs/msg/robot_speed.hpp"
#include "ros2_aruco_interfaces/msg/aruco_markers.hpp"

using namespace std::chrono_literals;

class RockCentering : public rclcpp::Node {
    public:
        RockCentering() : Node("rock_centering") {

            aruco_markers_sub_ = create_subscription<ros2_aruco_interfaces::msg::ArucoMarkers>(
                "/aruco_markers", 10,
                std::bind(&RockCentering::aruco_markers_callback, this, std::placeholders::_1)
            );

            action_server_ = rclcpp_action::create_server<robot_msgs::action::RockCentering>(
                this, "/rock_centering",
                std::bind(&RockCentering::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
                std::bind(&RockCentering::handle_cancel, this, std::placeholders::_1),
                std::bind(&RockCentering::handle_accepted, this, std::placeholders::_1)
            );

            robot_speed_pub_ = create_publisher<robot_msgs::msg::RobotSpeed>("/speed_command", 10);

            RCLCPP_INFO(get_logger(), "RockCentering launched successfully");
        }

    private:
        enum class Outcome { Succeeded, Aborted, Canceled };

        // Réglages
        int marker_id_ = 13;
        double x_ref_ = 0.0;
        double z_ref_ = 0.2;

        double kp_ = 0.8;
        double kp_theta_ = 1.0;

        double v_max_ = 0.25;
        double omega_max_ = 1.0;

        double tol_pos_ = 0.01;
        double tol_theta_ = 0.08;

        int stable_ticks_needed_ = 10;
        double marker_timeout_ = 0.5;

        double centering_timeout_ = 15.0;
        int feedback_every_ticks_ = 5;

        // Dernière détection du tag
        bool has_pose_ = false;
        geometry_msgs::msg::Pose pose_;
        rclcpp::Time last_seen_;

        // État du centrage
        std::shared_ptr<rclcpp_action::ServerGoalHandle<robot_msgs::action::RockCentering>> active_goal_;
        int stable_ticks_ = 0;
        int tick_count_ = 0;
        rclcpp::Time start_time_;

        rclcpp::Subscription<ros2_aruco_interfaces::msg::ArucoMarkers>::SharedPtr aruco_markers_sub_;
        rclcpp_action::Server<robot_msgs::action::RockCentering>::SharedPtr action_server_;
        rclcpp::Publisher<robot_msgs::msg::RobotSpeed>::SharedPtr robot_speed_pub_;
        rclcpp::TimerBase::SharedPtr aruco_centering_timer_;


        void aruco_markers_callback(const ros2_aruco_interfaces::msg::ArucoMarkers::SharedPtr msg) {
            bool found = false;
            double best_z = 1e9;
            for (size_t i = 0; i < msg->marker_ids.size() && i < msg->poses.size(); ++i) {
                if (msg->marker_ids[i] != marker_id_) continue;

                if (msg->poses[i].position.z < best_z) {
                    best_z = msg->poses[i].position.z;
                    pose_ = msg->poses[i];
                    found = true;
                }
            }
            if (found) {
                has_pose_ = true;
                last_seen_ = now();
            }
        }

        rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID & /*uuid*/, std::shared_ptr<const robot_msgs::action::RockCentering::Goal> /*goal*/) {
            if (active_goal_) {
                RCLCPP_WARN(get_logger(), "Goal rejected: centering already in progress");
                return rclcpp_action::GoalResponse::REJECT;
            }
            RCLCPP_INFO(get_logger(), "Goal received");
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }

        rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<rclcpp_action::ServerGoalHandle<robot_msgs::action::RockCentering>> /*goal_handle*/) {
            RCLCPP_INFO(get_logger(), "Cancel requested");
            return rclcpp_action::CancelResponse::ACCEPT;
        }

        void handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<robot_msgs::action::RockCentering>> goal_handle) {
            active_goal_ = goal_handle;
            stable_ticks_ = 0;
            tick_count_ = 0;
            start_time_ = now();
            aruco_centering_timer_ = create_wall_timer(20ms, std::bind(&RockCentering::aruco_centering_loop, this));
            RCLCPP_INFO(get_logger(), "Centering started on marker %d", marker_id_);
        }

        void aruco_centering_loop() {
            if (!active_goal_) return;

            if (active_goal_->is_canceling()) {
                finish_centering(Outcome::Canceled, "Centering canceled");
                return;
            }

            const double elapsed = (now() - start_time_).seconds();
            if (elapsed > centering_timeout_) {
                finish_centering(Outcome::Aborted, "Centering timeout");
                return;
            }

            ++tick_count_;
            auto feedback = std::make_shared<robot_msgs::action::RockCentering::Feedback>();
            feedback->elapsed = elapsed;

            if (!has_pose_ || (now() - last_seen_).seconds() > marker_timeout_) {
                stable_ticks_ = 0;
                publish_speed(0.0, 0.0, 0.0);
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Marker %d not visible, waiting", marker_id_);
                if (tick_count_ % feedback_every_ticks_ == 0) {
                    feedback->marker_visible = false;
                    active_goal_->publish_feedback(feedback);
                }
                return;
            }

            const auto & p = pose_.position;
            const auto & q = pose_.orientation;

            // Normale du tag (axe z du tag) dans le repère optique = 3e colonne de la matrice de rotation.
            const double nx = 2.0 * (q.x * q.z + q.w * q.y);
            const double nz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);

            const double err_fwd = p.z - z_ref_;              // > 0 : trop loin -> avancer
            const double err_lat = -(p.x - x_ref_);           // tag à droite (x > 0) -> aller à droite (vy < 0)
            const double err_theta = std::atan2(nx, -nz);     // 0 quand la caméra est face au tag

            const double vx = std::clamp(kp_ * err_fwd, -v_max_, v_max_);
            const double vy = std::clamp(kp_ * err_lat, -v_max_, v_max_);
            const double omega = std::clamp(kp_theta_ * err_theta, -omega_max_, omega_max_);

            if (tick_count_ % feedback_every_ticks_ == 0) {
                feedback->marker_visible = true;
                feedback->err_fwd = err_fwd;
                feedback->err_lat = err_lat;
                feedback->err_theta = err_theta;
                active_goal_->publish_feedback(feedback);
            }

            const bool ok = std::fabs(err_fwd) < tol_pos_ && std::fabs(err_lat) < tol_pos_ && std::fabs(err_theta) < tol_theta_;
            stable_ticks_ = ok ? stable_ticks_ + 1 : 0;

            if (stable_ticks_ >= stable_ticks_needed_) {
                finish_centering(Outcome::Succeeded, "Centered on ArUco");
                return;
            }
            publish_speed(vx, vy, omega);
        }

        void publish_speed(double vx, double vy, double omega) {
            robot_msgs::msg::RobotSpeed msg;
            const double scale = std::max(1.0, std::hypot(vx, vy) / v_max_);
            msg.vx = vx / scale;
            msg.vy = vy / scale;
            msg.omega = omega;
            robot_speed_pub_->publish(msg);
        }

        void finish_centering(Outcome outcome, const std::string & message) {
            if (aruco_centering_timer_) {
                aruco_centering_timer_->cancel();
                aruco_centering_timer_.reset();
            }
            publish_speed(0.0, 0.0, 0.0);

            auto result = std::make_shared<robot_msgs::action::RockCentering::Result>();
            result->success = (outcome == Outcome::Succeeded);
            result->message = message;

            switch (outcome) {
                case Outcome::Succeeded: active_goal_->succeed(result);  break;
                case Outcome::Aborted:   active_goal_->abort(result);    break;
                case Outcome::Canceled:  active_goal_->canceled(result); break;
            }

            active_goal_.reset();
            RCLCPP_INFO(get_logger(), "Centering finished: %s", message.c_str());
        }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RockCentering>());
    rclcpp::shutdown();
    return 0;
}