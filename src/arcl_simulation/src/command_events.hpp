#pragma once

#include <chrono>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

// Optional command telemetry. Callers guard calls so disabled telemetry does not
// construct identifiers, serialize parameters or maintain request bookkeeping.
class CommandEvents
{
public:
  static std::unique_ptr<CommandEvents> create_if_enabled(rclcpp::Node * node) {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.read_only = true;
    descriptor.description = "Publish command telemetry; configured only at startup";
    if (!node->declare_parameter("publish_command_events", false, descriptor)) return nullptr;
    return std::make_unique<CommandEvents>(node);
  }

  explicit CommandEvents(rclcpp::Node * node) : node_(node) {
    publisher_ = node->create_publisher<std_msgs::msg::String>("/agent/command_events", 100);
  }

  template<typename UUID>
  static std::string goal_id(const UUID & uuid) {
    std::ostringstream out;
    for (auto byte : uuid) out << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return out.str();
  }

  template<typename UUID>
  void move(const UUID & uuid, double x, double y, double theta) {
    std::ostringstream parameters;
    parameters << std::setprecision(17) << "{\"x\":" << x << ",\"y\":" << y << ",\"theta\":" << theta << '}';
    publish(goal_id(uuid), "move", "accepted", parameters.str());
  }

  void cancel() {
    publish(next_id("cancel"), "cancel", "accepted", "{\"operation_type\":\"MOVE\"}");
  }

  void speed(double vx, double vy, double omega, bool valid, bool accepted) {
    const auto id = next_id("speed");
    std::ostringstream parameters;
    if (valid) parameters << std::setprecision(17) << "{\"vx\":" << vx << ",\"vy\":" << vy << ",\"omega\":" << omega << '}';
    publish(id, "speed", accepted ? "accepted" : "rejected", valid ? parameters.str() : "{}");
    if (accepted) speed_id = id;
  }

  void begin_request(const std::shared_ptr<rmw_request_id_t> & header,
    bool enabled, const std::string & group, bool accepted) {
    const std::string id = request_key(header);
    const std::string command = group == "wrist" ? "wrist" : "suction";
    const std::string parameters = group == "wrist"
      ? std::string("{\"target_angle_rad\":") + (enabled ? "1.5707963267948966}" : "0}")
      : std::string("{\"group\":") + (group == "all" ? "\"all\"" : group)
        + ",\"enabled\":" + (enabled ? "true}" : "false}");
    publish(id, command, accepted ? "accepted" : "rejected", parameters);
    if (accepted) requests_[id] = command;
  }

  void finish_request(const std::shared_ptr<rmw_request_id_t> & header,
    bool success, const std::string & message) {
    const auto key = request_key(header);
    auto entry = requests_.find(key);
    if (entry == requests_.end()) return;
    publish(key, entry->second, success ? "succeeded" : "failed", "{}", message);
    requests_.erase(entry);
  }

  void publish(const std::string & id, const std::string & command,
    const std::string & status, const std::string & parameters = "{}",
    const std::string & message = "") {
    std_msgs::msg::String event;
    std::ostringstream out;
    out << std::setprecision(17) << "{\"format\":\"arcl-command-event\",\"id\":" << quote(id)
        << ",\"command\":" << quote(command) << ",\"status\":" << quote(status)
        << ",\"timestamp\":" << node_->now().seconds() << ",\"parameters\":" << parameters
        << ",\"message\":" << quote(message) << '}';
    event.data = out.str();
    publisher_->publish(event);
  }

  std::string navigation_id;
  std::string speed_id;

private:
  std::string next_id(const std::string & prefix) {
    return prefix + '-' + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
      + '-' + std::to_string(++counter_);
  }

  static std::string request_key(const std::shared_ptr<rmw_request_id_t> & header) {
    std::ostringstream out;
    for (auto byte : header->writer_guid) out << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    out << '-' << header->sequence_number;
    return out.str();
  }

  static std::string quote(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
      if (c == '"' || c == '\\') out << '\\' << c;
      else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
      else out << c;
    }
    out << '"';
    return out.str();
  }

  rclcpp::Node * node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  std::map<std::string, std::string> requests_;
  unsigned long counter_ = 0;
};
