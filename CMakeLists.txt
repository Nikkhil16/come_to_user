cmake_minimum_required(VERSION 3.8)
project(come_to_user)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/come_to_user_script.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  package_runner.json
  README.md
  requirements.txt
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
