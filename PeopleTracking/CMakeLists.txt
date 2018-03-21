#This basically describes how you're able to add OpenCV to your own program, if you're actually #using CMake.

#The text in quotes has to be replaced with your actual values/project names, for example, if my #target is called myopencvthing and I'd want to use OpenCV 2.0 or newer, then I'd setup something #like this:

# First tell CMake the minimal version of CMake expected
cmake_minimum_required(VERSION 2.8)

# Define a project
project(myopencvproject)


# Tell CMake to look for OpenCV 2.0 (and tell it that it's required, not optional)
find_package(OpenCV REQUIRED)


#defines a target that uses C++11
set (CMAKE_CXX_STANDARD 11)
#use thread libraries in project
find_package( Threads )


# Tell CMake to add the OpenCV include directory for the preprocessor
include_directories(${OpenCV_INCLUDE_DIRS})


# Add the source files to a variable
set( SOURCES
	peopledetect.cpp
	#main.cpp
)

set( HEADERS
	#main.hpp
)


# Define the actual executable target and the source files
add_executable(myopencvthing ${SOURCES} ${HEADERS} )
#add_executable(myopencvthing)

# Finally, add the dependencies of our executable (i.e. OpenCV):
target_link_libraries(myopencvthing ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

