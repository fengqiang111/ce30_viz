configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

add_library(
  ce30_pcviz
    SHARED
      ce30_pcviz.cpp
      ce30_pcviz.h
      config.h.in
      export.h
)

set_target_properties(ce30_pcviz PROPERTIES
  PUBLIC_HEADER "ce30_pcviz.h;${CMAKE_CURRENT_BINARY_DIR}/config.h")

install(TARGETS ce30_pcviz
  # IMPORTANT: Add the ce30_pcviz library to the "export-set"
  EXPORT ce30_pcvizTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/ce30_pcviz"
    COMPONENT dev)