set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Core)
find_package(ce30_driver)

add_executable(
  ce30_pointcloud_viewer
    main.cpp
    point_cloud_viewer.cpp
    point_cloud_viewer.h
    exit_code.h
)

target_link_libraries(
  ce30_pointcloud_viewer
    ce30_pcviz
    ${CE30_DRIVER_LIBRARIES}
    Qt5::Core)

install(TARGETS ce30_pointcloud_viewer
  # IMPORTANT: Add the ce30_pointcloud_viewer executable to the "export-set"
  EXPORT ce30_vizTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)