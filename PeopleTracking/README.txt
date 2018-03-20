Helpful link for WiringPi with CMAKE: 
http://stackoverflow.com/questions/30424236/add-wiringpi-lib-to-cmake-on-raspberrypi
In order to assemble the CMake items, I placed a file called "WiringPiConfig.cmake" into
/usr/share/cmake-2.8/Modules

Then I used the included CMakeLists.txt included with the project that points to the directory of the project:
cmake /home/pi/Downloads/ROOBockey-master 

then type:
make

then to run the program
./myopencvthing
