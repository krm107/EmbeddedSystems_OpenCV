# CMAKE file for generating a "release version" executable for the ROOBockey OPENCV 3.0.0 project
# This template was taken from: http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html
# Also Looking off here: http://stackoverflow.com/questions/33749216/compiling-an-opencv-project-using-cmake-on-cygwin-with-opencv-installed-for-win


# To use this file to build the C++ OpenCV3 Project:
# cd into the project directory (where this file is located) and type cmake, then make
# 		EXAMPLE:
# cd ~/Desktop/ROOBockey
# cmake <project folder directory goes here>
# make


#Help for using cmake with Wiring Pi:
#A:HELP) https://www.raspberrypi.org/forums/viewtopic.php?t=39198&p=323138
#http://stackoverflow.com/questions/30424236/add-wiringpi-lib-to-cmake-on-raspberrypi

#FOLLOWING STEPS FROM raspberrypi.org help forum:
#0) INSTALL WIRING PI using the setup here:
#	http://wiringpi.com/download-and-install/
#1) make a file located in "/usr/share/cmake-2.8/Modules" called "FindWiringPi.cmake"
#2) fill the file with the following lines to allow CMAKE to find the WiringPi module:
#	find_library(WIRINGPI_LIBRARIES NAMES wiringPi)
#	find_path(WIRINGPI_INCLUDE_DIRS NAMES wiringPi.h)
#	include(FindPackageHandleStandardArgs)
#	find_package_handle_standard_args(wiringPi DEFAULT_MSG WIRINGPI_LIBRARIES WIRINGPI_INCLUDE_DIRS)



cmake_minimum_required(VERSION 2.8)
project( ROOBockey )


#add some flags, pthread was for a different projet, lwiringPi should let me include <wiringPi.h>
SET( CMAKE_CXX_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++11 -pthread -I/usr/local/include -L/usr/local/lib -lwiringPi ")
SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++11 -pthread -I/usr/local/include -L/usr/local/lib -lwiringPi ")
SET( WiringPi_DIR "/usr/share/cmake-2.8/Modules")
set( CMAKE_MODULE_PATH "/usr/local/lib/cmake/${CMAKE_MODULE_PATH}")


# Locate libraries and headers (locates .cmake files)
find_package( OpenCV REQUIRED )
find_package( Threads REQUIRED ) 
#find_package( WiringPi REQUIRED )


# Include headers
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${WIRINGPI_INCLUDE_DIRS}) 


set( NAME_SRC
	main.cpp
	ObjectTracking.cpp
	Beacons.cpp
	defs.cpp
	GPIO_UART.cpp
	Xbox360Controller.cpp
)

set( NAME_HEADERS
	main.hpp
	ObjectTracking.hpp
	Beacons.hpp
	defs.hpp
	GPIO_UART.hpp
	Xbox360Controller.hpp
)


# NOW ADD THE OUTPUT EXECUTABLE FILE NAME FOLLOWED BY ALL SOURCE FILES TO THE CMAKE OUTPUT
add_executable( ROOBockey ${NAME_SRC} ${NAME_HEADERS} )


# Link against libraries
target_link_libraries(ROOBockey ${OpenCV_LIBS} )
target_link_libraries(ROOBockey ${WIRINGPI_LIBRARIES})
target_link_libraries(ROOBockey ${CMAKE_THREAD_LIBS_INIT})
