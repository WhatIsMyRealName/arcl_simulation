cmake_minimum_required(VERSION 3.8)
project(arcl_simulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp)
find_package(std_msgs)
find_package(std_srvs REQUIRED)
find_package(robot_msgs)
find_package(sensor_msgs)
find_package(visualization_msgs)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ros2_aruco_interfaces REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)


add_executable(robot_control src/robot_control.cpp)
target_include_directories(robot_control PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(robot_control rclcpp std_msgs robot_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs rclcpp_action)

add_executable(beacon_localization src/beacon_localization.cpp)
target_include_directories(beacon_localization PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(beacon_localization rclcpp sensor_msgs geometry_msgs nav_msgs visualization_msgs robot_msgs tf2_ros)

add_executable(wheel_odometry src/wheel_odometry.cpp)
target_include_directories(wheel_odometry PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(wheel_odometry rclcpp sensor_msgs nav_msgs)

add_executable(rock_centering src/rock_centering.cpp)
ament_target_dependencies(rock_centering rclcpp ros2_aruco_interfaces robot_msgs rclcpp_action)

add_executable(strategy src/strategy.cpp)
ament_target_dependencies(strategy rclcpp robot_msgs rclcpp_action std_msgs std_srvs tf2_ros tf2 geometry_msgs)

add_executable(ventouse_service src/ventouse_service.cpp)
target_compile_features(ventouse_service PRIVATE cxx_std_17)
ament_target_dependencies(ventouse_service rclcpp std_msgs std_srvs)

add_executable(otos_simulator src/otos_simulator.cpp)
target_include_directories(otos_simulator PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(otos_simulator rclcpp nav_msgs)

add_executable(table_markers src/table_markers.cpp)
ament_target_dependencies(table_markers rclcpp visualization_msgs)

install(TARGETS robot_control beacon_localization wheel_odometry rock_centering strategy ventouse_service table_markers otos_simulator
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch rviz urdf worlds models
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip linters which require source copyright headers that are not present yet.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
