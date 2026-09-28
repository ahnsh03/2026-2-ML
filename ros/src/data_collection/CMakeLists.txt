cmake_minimum_required(VERSION 3.0.2)
project(data_collection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  diagnostic_msgs
  morai_msgs
  message_filters
  rosbag
  rosgraph
)

catkin_python_setup()

catkin_package()

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_capture_mode.py)
  catkin_add_nosetests(test/test_capture_sync.py)
  catkin_add_nosetests(test/test_mask_baker.py)
  catkin_add_nosetests(test/test_perception_curation.py)
  catkin_add_nosetests(test/test_perception_dataset.py)
endif()

catkin_install_python(PROGRAMS
  scripts/capture_mode_collector_node.py
  scripts/morai_udp_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
