cmake_minimum_required(VERSION 3.8)
project(deported_base_calculus)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp)
find_package(geometry_msgs)
find_package(tf2 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ros2_aruco_interfaces REQUIRED)

add_executable(cam_tracking src/cam_tracking.cpp)
ament_target_dependencies(cam_tracking rclcpp ros2_aruco_interfaces geometry_msgs tf2 visualization_msgs)

install(TARGETS cam_tracking 
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
