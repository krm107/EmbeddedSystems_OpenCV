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


//Working with Eclipse and OpenCV3.0
//https://www.eclipse.org/forums/index.php/t/59314/
//https://github.com/facebook/C3D/issues/253


using namespace cv;
using namespace std;


#define FRAME_WIDTH 320
#define FRAME_HEIGHT 240
#define BLACK cv::Scalar(0.0, 0.0, 0.0)
#define WHITE cv::Scalar(255.0, 255.0, 255.0)
#define YELLOW cv::Scalar(0.0, 255.0, 255.0)
#define GREEN cv::Scalar(0.0, 200.0, 0.0)
#define RED cv::Scalar(0.0, 0.0, 255.0)
#define BLUE cv::Scalar(255, 0.0, 0.0)

//Tunable Parameters for object detection
int THRESHOLD = 90; //default value of 30
int scFxThous = 1100; //value on trackbar is 1000x scaled up (default 1.1 is 1100 scaled)
int MIN_NEIGHBORS = 3;
int RECT_AREA_SIZE_MOVING_CLOSER = 2000; //value on trackbar is 1000x scaled up (default 2.0 is 2000 scaled)
int NUM_NOT_DETECTED = 50;
int STOP_PROGRAM = 0;

bool seeDebugFramesOutput = true;
bool displayFramesPerSecond = true;



//This function gets called whenever a trackbar position is changed
void on_trackbar( int, void* ){ }


//trackbars are used to tune the parameters for detecting after applying a "convex hull"
void createTrackbars(){

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
	createTrackbar( "RECT_AREA_CLOSER", trackbarWindowName, &RECT_AREA_SIZE_MOVING_CLOSER, 10000, on_trackbar );
	createTrackbar( "NUM_NOT_DETECTED", trackbarWindowName, &NUM_NOT_DETECTED, 1000, on_trackbar );
	createTrackbar( "STOP_PROGRAM", trackbarWindowName, &STOP_PROGRAM, 1, on_trackbar );
}


//attempt to find people using "Cascade Classifier" in the provides image
void findBodies(cv::Mat img, personObj personObjReference, CascadeClassifier tempBodyCascade)
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

    static int numFramesNotDetectingPerson = 0;

	if(rectangleBodies.size() > 0) //person was detected
	{
		personObjReference.stillBeingTracked = true;
		personObjReference = personObj(rectangleBodies[0]); //assume person is the first object detected in "rectangleBodies" vector
		int thickness=2, lineType=8, shift=0;
		rectangle( img, rectangleBodies[0], BLUE, thickness, lineType, shift );
	}
	else if(numFramesNotDetectingPerson > NUM_NOT_DETECTED) //wait number of frames before saying person is not detected
	{
		//person not found in frame --> set flags to indicate person was not found
		personObjReference.stillBeingTracked = false;
		personObjReference.numConsecutiveFramesWithoutAMatch++;
	}
	else{}
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
	double LastPersonArea = 0.0; //used to detect area of person's rectangle on last iteration of state machine
	double rollAvgRectArea;
	double rectMovingCloserDiff;


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

	//https://stackoverflow.com/questions/24195926/opencv-write-webcam-output-to-avi-file?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	VideoWriter outputVideo("outputVideo.avi", CV_FOURCC('M','J','P','G'), 10, Size(FRAME_WIDTH, FRAME_HEIGHT), true);


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


		//state machine logic for detected person approaching camera
		if(personFound.stillBeingTracked == true)
		{
			//Only compare object moving closer in mode 2.2 or 3 or 4
			if(LastPersonArea != 0.0)
			{
				//Compute area of detected person rectangle between the last time a person was detected
				//	This signifies the person is getting large from the camera view and must be approaching the camera
				rectMovingCloserDiff = personFound.dblCurrentArea - LastPersonArea;
				rollAvgRectArea = personFound.rollingAverageCalc(rectMovingCloserDiff);
			}

			if(state >= 2.0 && state < 3.0)
			{
				if(state == 2.0)
				{
					LastPersonArea = personFound.dblCurrentArea;
					timeFirstPersonObjDetected = std::time(0);
					state = 2.1;
					break;
				}

				std::time_t timePresent1 = std::time(0);
				//STATE2: save image every 6 seconds
				if (timePresent1 - timeFirstPersonObjDetected >= 6)
				{
					cout << "S2: 6secPic" << endl;
					imwrite("outputPicture.jpg", imgThreshNorm);
					state = 2.2;
				}
			}
			else if( ( state == 2.2  &&  rollAvgRectArea >= (double)RECT_AREA_SIZE_MOVING_CLOSER/1000.0 )
					|| ( state >= 3.0 && state < 4.0 ) )
			{
				if(state == 2.2)
				{
					LastPersonArea = personFound.dblCurrentArea;
					timeFirstPersonObjDetected = std::time(0);
					state = 3.1;
					break;
				}

				std::time_t timePresent2 = std::time(0);
				//STATE3: moving toward camera; take higher resolution picture every 3 seconds
				if (timePresent2 - timeFirstPersonObjDetected >= 3)
				{
					cout << "S3: 3secPic" << endl;
					imwrite("outputPicture.jpg", imgThreshNorm);
					state = 3.2;
				}
			}
			else if( ( state == 3.2  &&  rollAvgRectArea >= (double)RECT_AREA_SIZE_MOVING_CLOSER/1000.0 )
					|| (state >= 4.0) )
			{
				if(state == 3.2)
				{
					LastPersonArea = personFound.dblCurrentArea;
					timeFirstPersonObjDetected = std::time(0);
					state = 4.1;
					break;
				}

				std::time_t timePresent2 = std::time(0);
				//STATE4: moving toward camera; save frames to video (.AVI format video)
				if (timePresent2 - timeFirstPersonObjDetected >= 3)
				{
					cout << "S4: Video" << endl;
					outputVideo.write(imgThreshNorm);
					state = 4.2;
				}
			}
			else
			{
				//never gets here
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
	//run camera operations in separate threads
	//Thread info from "http://www.cplusplus.com/reference/thread/thread/"
	int camera0 = 0;
	int camera1 = 1;

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
	std::thread second (cameraOperations, camera1, CascadeClassFileXML); //spawn new thread that calls cameraOperations(camera1)
	cout << "camera threads running" << endl;

	//Makes the main thread wait for the new thread to finish execution, therefore blocks its own execution.
	first.join();
	//second.join();

	return 0;
}
