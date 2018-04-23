#include <iostream>
#include <stdexcept>
#include <ctime>
#include <thread>
#include <pthread.h>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include "personObj.h"
#include <fstream> //used to write to text file using "std::ofstream"
#include<sstream> //used to create a directory for sotring pictures and video using "stringstream"
#include <sys/stat.h> //used to detect if directories for pictures and video were already created


//Working with Eclipse and OpenCV3.0
//https://www.eclipse.org/forums/index.php/t/59314/
//https://github.com/facebook/C3D/issues/253


using namespace cv;
using namespace std;


//#define FRAME_WIDTH 160
//#define FRAME_HEIGHT 120
#define FRAME_WIDTH 320
#define FRAME_HEIGHT 240
#define BLACK cv::Scalar(0.0, 0.0, 0.0)
#define WHITE cv::Scalar(255.0, 255.0, 255.0)
#define YELLOW cv::Scalar(0.0, 255.0, 255.0)
#define GREEN cv::Scalar(0.0, 200.0, 0.0)
#define RED cv::Scalar(0.0, 0.0, 255.0)
#define BLUE cv::Scalar(255, 0.0, 0.0)

//Tunable Parameters for object detection
int THRESHOLD = 110; //default value of 30
int scFxThous = 1100; //value on trackbar is 1000x scaled up (default 1.1 is 1100 scaled)
int MIN_NEIGHBORS = 5;
int RECT_AREA_SIZE_MOVING_CLOSER = 300; //value on trackbar is 1000x scaled up (default 2.0 is 2000 scaled)
int NUM_NOT_DETECTED = 100;
int STOP_PROGRAM = 0;

bool seeDebugFramesOutput = true;
bool displayFramesPerSecond = true;

string videoFileName;

// Create the outputfilestream
std::ofstream txtLogFileWrite("txtLogFileWrite.txt");


//This function gets called whenever a trackbar position is changed
void on_trackbar( int, void* ){ }


//trackbars are used to tune the parameters for detecting after applying a "convex hull"
void createTrackbars()
{
	std::time_t timeNow = std::time(NULL);
	txtLogFileWrite << "trackbarsChanged: " << to_string(timeNow) << endl;

	string trackbarWindowName = "Trackbars";

	//create window for trackbars
	namedWindow(trackbarWindowName,0);
	//create trackbars and insert them into window
	//3 parameters are:
	//the address of the variable that is changing when the trackbar is moved(eg.THRESHOLD),
	//the max value the trackbar can move (eg. 255 for 8 bit register),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)

	createTrackbar( "THRESHOLD", trackbarWindowName, &THRESHOLD, 255, on_trackbar );
	createTrackbar( "scFxThous", trackbarWindowName, &scFxThous, 5000, on_trackbar ); //value on trackbar is 1000x scaled up (default 1.1 is 1100 scaled)
	createTrackbar( "MIN_NEIGHBORS", trackbarWindowName, &MIN_NEIGHBORS, 10, on_trackbar );
	createTrackbar( "RECT_AREA_CLOSER", trackbarWindowName, &RECT_AREA_SIZE_MOVING_CLOSER, 3000, on_trackbar );
	createTrackbar( "NUM_NOT_DETECTED", trackbarWindowName, &NUM_NOT_DETECTED, 3000, on_trackbar );
	createTrackbar( "STOP_PROGRAM", trackbarWindowName, &STOP_PROGRAM, 1, on_trackbar );
}


//attempt to find people using "Cascade Classifier" in the provides image
void findBodies(cv::Mat img, personObj &personObjReference, CascadeClassifier tempBodyCascade)
{
    Mat frame_gray;
    std::vector<Rect> rectangleBodies; //detected people from "Cascade Classifier" are represented as rectangles

    //Get images into grayscale
    cvtColor( img, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect bodies within the provided image frame
	//void CascadeClassifier::detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1,
    	//	int minNeighbors=3, int flags=0, Size minSize=Size(),Size maxSize=Size() )
//    body_cascade.detectMultiScale( img, bodies, 1.1, 1, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    tempBodyCascade.detectMultiScale( img, rectangleBodies, (double)scFxThous/1000.0, MIN_NEIGHBORS, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

	if(rectangleBodies.size() > 0) //person was detected
	{
		personObjReference = personObj(rectangleBodies[0]); //assume person is the first object detected in "rectangleBodies" vector
		personObjReference.stillBeingTracked = true;
		int thickness=2, lineType=8, shift=0;
		rectangle( img, rectangleBodies[5], BLUE, thickness, lineType, shift );
		personObjReference.numConsecutiveFramesWithoutAMatch = 0;
	}
	else if(personObjReference.numConsecutiveFramesWithoutAMatch > NUM_NOT_DETECTED) //wait number of frames before saying person is not detected
	{
		//person not found in frame --> set flags to indicate person was not found
		personObjReference.stillBeingTracked = false;
		personObjReference.numConsecutiveFramesWithoutAMatch = 0;
	}
	else //person not detected using "tempBodyCascade.detectMultiScale()"
	{
		personObjReference.numConsecutiveFramesWithoutAMatch++;
	}
}


std::string convertDateTime(std::time_t rawtime)
{
	  struct tm * timeinfo;
	  char buffer[80];
	  time (&rawtime);
	  timeinfo = localtime(&rawtime);

	  strftime(buffer, sizeof(buffer), "%Y:%m:%d - %I:%M:%S",timeinfo);
	  std::string str(buffer);
	  return str;
}


void cameraOperations(int cameraNum, FileStorage XmlClassFile)
{
	int tick = 0;
	int fps = 0;
	long frameCounter = 0;
	std::time_t timeBegin = std::time(0);
	cv::Mat frame1;
	cv::Mat frame2;
	std::time_t timeFirstPersonObjDetected = std::time(0);
	float state = 0.0; //state machine 1)Nothing2)Takepicture6secondsDetected3)Picture3secondsMovingCloser4)VideoMovingCloserTime
	double rollAvgRectAreaInitial = 0.0; //used to detect area of person's rectangle on last iteration of state machine
	double rollAvgRectAreaNew = 0.0;
	double rollAvgNewMinInitial = 0.0;


	cout << "Thread Start " << cameraNum << endl;

	/* Using Marc's Cascade Classifier in a thread-safe manner */
	String body_cascade_name;
	CascadeClassifier body_cascade;

	if (!body_cascade.read(XmlClassFile.getFirstTopLevelNode()))
		cout << "ERROR: Cascade XML CANNOT LOAD in thread:" << cameraNum << endl;

	cout << "XML loaded thread:" << cameraNum << endl;


	personObj personFound( cv::Rect(0,0,0,0) ); //create public personObj using constructor

	//Set source of video: "camera 0" is the builtin laptop webcam, "camera 1" is usb webcam
	VideoCapture vc1(cameraNum); //Open the Default Camera

	if (!vc1.isOpened() )
		exit(EXIT_FAILURE); //Check if we succeeded in receiving images from camera. If not, quit program.

	vc1.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); //Set height and width of capture frame
	vc1.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

	//flag to only capture 2 frames at runtime, each loop after will only get 1 frame
	bool firstTimeThrough = true;

	//convey which thread is being run at a time, but only every 1000ms
	unsigned long temp1 = 0;

	cout << "Cam Open Thread:" << cameraNum << endl;


	while(true)
	{
		if(firstTimeThrough==true)
		{
			vc1 >> frame1; //get a first frame from camera to get difference of images
		}
		vc1 >> frame2; //get a second frame from camera to get difference of images

		//Output Image after threshold (Thresholding is a Low-pass filter)
		cv::Mat imgThreshNorm;

		//Filter image further
		cv::threshold(frame2, imgThreshNorm, THRESHOLD, 255.0, CV_THRESH_BINARY);


//Using Marc's method of "Cascade Classifier"

		//Detect and draw bodies using "Cascade Classifier"
		findBodies(imgThreshNorm, personFound, body_cascade);

		//Show detected objects using "imshow()"
		if(seeDebugFramesOutput)
		{
			//give window a title to separate different cameras when called on different threads
			string windowName2 = "cam" + to_string(cameraNum) + ": windowName3:imgThreshNorm";
			cv::imshow(windowName2, imgThreshNorm);
		}


		//BEGIN STATE MACHINE LOGIC:
		//	state machine logic for detected person approaching camera
		if(personFound.stillBeingTracked == true)
		{
			if(state >= 2.0 && state < 3.0)
			{
				if(state == 2.0)
				{
					//find baseline detected person area the first time in State#2
					rollAvgRectAreaInitial = personFound.dblCurrentArea;
					timeFirstPersonObjDetected = std::time(0);
					state = 2.1;
					txtLogFileWrite << "State2Init: " << convertDateTime(timeFirstPersonObjDetected) << endl;
					continue;
				}

				rollAvgRectAreaNew = personFound.rollingAverageCalc(personFound.dblCurrentArea);
				rollAvgNewMinInitial = rollAvgRectAreaNew - rollAvgRectAreaInitial;

				std::time_t timePresent1 = std::time(0);
				//STATE2: save image every 6 seconds
				if (timePresent1 - timeFirstPersonObjDetected >= 2)
				{
					rollAvgRectAreaInitial = rollAvgRectAreaNew;
					cout << "S2: 6secPic" << "Diff" << rollAvgNewMinInitial << "Init" << rollAvgRectAreaInitial << endl;
					timeFirstPersonObjDetected = std::time(0);
					state = 2.2;
					std::time_t timeNow = std::time(NULL);
					txtLogFileWrite << "State2: 6SecPic T" << to_string(cameraNum) << ":" << convertDateTime(timeNow) << endl;
					string pictureFileName = "./outputPictures/S2outputPic T" + to_string(cameraNum) + ":" + convertDateTime(timeNow) + ".jpg";
					imwrite(pictureFileName, frame1);
				}
			}
			if( ( state == 2.2  &&  rollAvgNewMinInitial >= (double)RECT_AREA_SIZE_MOVING_CLOSER/1000.0 )
					|| ( state >= 3.0 && state < 4.0 ) )
			{
				if(state == 2.2)
				{
					//find baseline detected person area the first time in State#3
					rollAvgRectAreaInitial = personFound.dblCurrentArea;
					timeFirstPersonObjDetected = std::time(0);
					state = 3.1;
					std::time_t timeNow = std::time(NULL);
					txtLogFileWrite << "State3Init: " << convertDateTime(timeFirstPersonObjDetected) << endl;
					continue;
				}

				rollAvgRectAreaNew = personFound.rollingAverageCalc(personFound.dblCurrentArea);
				rollAvgNewMinInitial = rollAvgRectAreaNew - rollAvgRectAreaInitial;

				std::time_t timePresent2 = std::time(0);
				//STATE3: moving toward camera; take higher resolution picture every 3 seconds
				if (timePresent2 - timeFirstPersonObjDetected >= 1)
				{
					rollAvgRectAreaInitial = rollAvgRectAreaNew;
					cout << "S3: 3secPic" << endl;
					timeFirstPersonObjDetected = std::time(0);
					state = 3.2;
					std::time_t timeNow = std::time(NULL);
					txtLogFileWrite << "State3: 3SecPic T" << to_string(cameraNum) << ":" << convertDateTime(timeNow) << endl;
					string pictureFileName = "./outputPictures/S3outputPic T" + to_string(cameraNum) + ":" + convertDateTime(timeNow) + ".jpg";
					imwrite(pictureFileName, frame1);
				}
			}
			if( ( state == 3.2  &&  rollAvgNewMinInitial >= (double)RECT_AREA_SIZE_MOVING_CLOSER/1000.0 )
					|| (state >= 4.0) )
			{
				std::time_t timeNow = std::time(NULL);

				if(state == 3.2) //only create 1 video file when transitioning from state 3 to state 4
				{
					//https://stackoverflow.com/questions/24195926/opencv-write-webcam-output-to-avi-file?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
					videoFileName = "./outputVideo/VideoCam T" + to_string(cameraNum) + convertDateTime(timeNow) + ".avi";
					state = 4.0;
				}

				cout << "S4: Video" << endl;
				timeFirstPersonObjDetected = timeNow;
				txtLogFileWrite << "State4: Video T" << to_string(cameraNum) << ":" << to_string(timeNow) << endl;
				VideoWriter outputVideo(videoFileName, CV_FOURCC('M','J','P','G'), 10, Size(FRAME_WIDTH, FRAME_HEIGHT), true);
				outputVideo.write(frame1);
			}

		}
		else //personFound.stillBeingTracked == false
		{
			//Person not detected
			//Next time person is detected, the state machine will be on state#2
			state = 2.0;
		}


		if(displayFramesPerSecond == true)
		{
			//update frame count value
			frameCounter++;
			std::time_t timeNow = std::time(0) - timeBegin;

			if (timeNow - tick >= 1)
			{
			    tick++;
			    fps = frameCounter;
			    frameCounter = 0;
			}

			if(temp1 > 50)
			{
				//convey which thread is being run at a time, but only every 1000ms
				cout << "camNum: " << cameraNum << "fps: " << fps << endl;
				temp1 = 0;
			}

			firstTimeThrough = false;
			temp1++;
		}


		//delay for 1 milliseconds to keep "imshow()" from locking up
		char c = (char)waitKey(1);
		if( c == 27 || c == 'q' || c == 'Q' )
		{
			cout << "Escape Pressed - Exiting Program" << endl;
			std::terminate();
		}

		if(STOP_PROGRAM > 0) //if user selects to terminate/stop the program, stop all threads
			std::terminate();
        }
}




int main(void)
{
	//identify which cameras are detected over USB
	cv::VideoCapture temp_camera;
	int maxTested = 10;
	for (int i = 0; i < maxTested; i++)
	{
		cv::VideoCapture temp_camera(i);
		bool res = (temp_camera.isOpened());
		temp_camera.release();
		if (res)
		{
			cout << "CamDetected:" << i << endl;
		}
	}


	//delete folder already in existence with old contents
	system("rm -r ./outputPictures");
	system("rm -r ./outputVideo");
	//create new folder that is empty
	system("mkdir -p ./outputPictures");
	system("mkdir -p ./outputVideo");


	//run camera operations in separate threads
	//Thread info from "http://www.cplusplus.com/reference/thread/thread/"
	int camera0 = 1; //default: 0
	int camera1 = 1; //default: 1
	int camera2 = 2; //default: 2
	int camera3 = 3; //default: 3
	int camnum = 1; //how many cameras are used (1,2,3, or 4)?

	//create slider bars for object filtering
	createTrackbars();


//https://stackoverflow.com/questions/7285480/opencv-cascadeclassifier-c-interface-in-multiple-threads
	//If you are working with LBP cascade of with Haar cascade stored in new format
	//	then you can avoid reading cascade from file system for each new thread:
//https://stackoverflow.com/questions/15429035/multithreaded-face-detection-stops-working
	//Further detectMultiScale is already multithreaded / parallelized. In the docs it says
	//	The function is parallelized with the TBB library.
	//Load cascade into memory:
//https://docs.opencv.org/2.4/modules/core/doc/xml_yaml_persistence.html
	//Load XML file into code: FileStorage fs("test.yml", FileStorage::READ);
	cv::FileStorage CascadeClassFileXML("haarcascade_upperbody.xml", cv::FileStorage::READ);
	if (!CascadeClassFileXML.isOpened())
	{
		cout << "ERROR: XML CascadeClassifier not loaded correctly" << endl;
		return -1;
	}

	std::thread first (cameraOperations, camera0, CascadeClassFileXML); //spawn new thread that calls cameraOperations(camera0)
	first.join(); //main loop will not "return 0" until the thread stops

	if(camnum>1)
	{
		std::thread second (cameraOperations, camera1, CascadeClassFileXML); //spawn new thread that calls cameraOperations(camera1)
		second.join(); //main loop will not "return 0" until the thread stops
	}
	if(camnum>2)
	{
		std::thread third (cameraOperations, camera2, CascadeClassFileXML); //spawn new thread that calls cameraOperations(camera1)
		third.join(); //main loop will not "return 0" until the thread stops
	}
	if(camnum>3)
	{
		std::thread fourth (cameraOperations, camera3, CascadeClassFileXML); //spawn new thread that calls cameraOperations(camera1)
		fourth.join();
	}
	cout << "camera threads running" << endl;

	//Makes the main thread wait for the new thread to finish execution, therefore blocks its own execution.
//	if(camnum>1)
//		second.join();
//	if(camnum>2)
//		third.join();
//	if(camnum>3)
//		fourth.join();

	return 0;
}
