## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -- Installation values
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(include_install_dir "include")
set(generated_dir "${PROJECT_BINARY_DIR}/generated")
set(
  version_config
  "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake"
  )
set(
  project_config
  "${generated_dir}/${PROJECT_NAME}Config.cmake"
  )
set(targets_export_name "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

## -- Global installation rules
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${version_config}" COMPATIBILITY SameMajorVersion
  )

## -- Configuration files installation
configure_package_config_file(
  "cmake/${PROJECT_NAME}Config.cmake.in"
  "${project_config}"
  INSTALL_DESTINATION "${config_install_dir}"
  )
install(
  FILES "${project_config}"
  DESTINATION "${config_install_dir}"
  )
install(
  FILES "${version_config}"
  DESTINATION "${config_install_dir}"
  )
install(
  EXPORT "${targets_export_name}"
  NAMESPACE "${namespace}"
  DESTINATION "${config_install_dir}"
  )

## eof - $RCSfile$
