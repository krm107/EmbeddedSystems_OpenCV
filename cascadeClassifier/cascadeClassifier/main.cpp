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


using namespace cv;
using namespace std;


#define FRAME_WIDTH 320
#define FRAME_HEIGHT 240
#define BLACK cv::Scalar(0.0, 0.0, 0.0)
#define WHITE cv::Scalar(255.0, 255.0, 255.0)
#define YELLOW cv::Scalar(0.0, 255.0, 255.0)
#define GREEN cv::Scalar(0.0, 200.0, 0.0)
#define RED cv::Scalar(0.0, 0.0, 255.0)

//Tunable Parameters for object detection
int THRESHOLD = 90; //default value of 30
int scFxThous = 1100; //value on trackbar is 1000x scaled up (default 1.1 is 1100 scaled)
int MIN_NEIGHBORS = 3;
int STOP_PROGRAM = 0;

bool seeDebugFrames = true;





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
	createTrackbar( "STOP_PROGRAM", trackbarWindowName, &STOP_PROGRAM, 1, on_trackbar );

}


//attempt to find people using "Cascade Classifier" in the provides image
void findBodies(cv::Mat img, personObj localPersonObj, CascadeClassifier tempBodyCascade)
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
	
//    for ( size_t i = 0; i < bodies.size(); i++ )
//    {
	if(rectangleBodies.size() > 0) //person was detected
	{
//		//Point center( bodies[0].x + bodies[0].width/2, bodies[0].y + bodies[0].height/2 );
//		//localPersonObj = new personObj(center, bodies[0].width, bodies[0].height);
//		//ellipse( img, center, Size( bodies[0].width/2, bodies[0].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
		
		localPersonObj = personObj(rectangleBodies[0]); //assume person is the first object detected in "rectangleBodies" vector
		int thickness=1, lineType=8, shift=0;
		rectangle( img, rectangleBodies[0], GREEN, thickness, lineType, shift );
	}
//    }

}



void cameraOperations(int cameraNum, int argc, const char** argv)
{
	int tick = 0;
	int fps = 0;
	long frameCounter = 0;
	std::time_t timeBegin = std::time(0);
	cv::Mat frame1;
	cv::Mat frame2;

	cout << "Camera Thread " << cameraNum <<" Started" << endl;

	/* Using Marc's Cascade Classifier in a thread-safe manner */
	String body_cascade_name;
	CascadeClassifier body_cascade;

	
	CommandLineParser parser(argc, argv,
	"{help h||}"
	"{body_cascade|haarcascade_upperbody.xml|}");
cout << "test1234" << endl;
	parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		  "You can use Haar or LBP features.\n\n" );
	parser.printMessage();
	body_cascade_name = parser.get<String>("body_cascade");

	//-- 1. Load the cascades
	cout << "Loading Cascade Classifier" << endl;
	body_cascade.load(body_cascade_name);
	cout << "Cascade Classifier Loaded" << endl;



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

	cout << "VideoCapture Opened" << endl;


        while(true)
        {
		if(firstTimeThrough==true)
		{
			vc1 >> frame1; //get a first frame from camera to get difference of images
		}
		vc1 >> frame2; //get a second frame from camera to get difference of images

		cv::Mat imgDiff;
		cv::Mat imgThreshDiff;
		cv::Mat imgThreshNorm;

		//Take difference between images
		if(firstTimeThrough==false)
			cv::absdiff(frame1, frame2, imgDiff);
		else
			imgDiff = frame2.clone();

		//Filter image further
		cv::threshold(imgDiff, imgThreshDiff, THRESHOLD, 255.0, CV_THRESH_BINARY);
		cv::threshold(frame2, imgThreshNorm, THRESHOLD, 255.0, CV_THRESH_BINARY);

		//give video window a title to separate different cameras when called on different threads
		string windowName = "cam" + to_string(cameraNum) + ": windowName1:DifferenceAndThresh";
		if(seeDebugFrames)
			cv::imshow(windowName, imgDiff);


//Using Marc's method of "Cascade Classifier"

		//Detect and draw bodies using "Cascade Classifier"
		findBodies(imgThreshDiff, personFound, body_cascade);
		findBodies(imgThreshNorm, personFound, body_cascade);

		//Show detected objects
		string windowName1 = "cam" + to_string(cameraNum) + ": windowName2:imgThreshDiff";
		if(seeDebugFrames)
			cv::imshow(windowName1, imgThreshDiff);
		string windowName2 = "cam" + to_string(cameraNum) + ": windowName3:imgThreshNorm";
		if(seeDebugFrames)
			cv::imshow(windowName2, imgThreshNorm);
		

	        frameCounter++; //update frame count value
		std::time_t timeNow = std::time(0) - timeBegin;

		if (timeNow - tick >= 1)
		{
		    tick++;
		    fps = frameCounter;
		    frameCounter = 0;
		}


		if(temp1 > 20)
		{
			//convey which thread is being run at a time, but only every 1000ms
			cout << "camNum: " << cameraNum << "fps: " << fps << endl;
			temp1 = 0;
		}


		firstTimeThrough = false;
		temp1++;

		cv::waitKey(1); //delay for 1 milliseconds to keep "imshow()" from locking up

		if(STOP_PROGRAM > 0) //if user selects to terminate/stop the program, stop all threads
			std::terminate();
        }
}




int main( int argc, const char** argv )
{
	//run camera operations in separate threads
	//Thread info from "http://www.cplusplus.com/reference/thread/thread/"
	int camera0 = 0;
	int camera1 = 1;

	//create slider bars for object filtering
	createTrackbars();

	std::thread first (cameraOperations, camera0, argc, argv); //spawn new thread that calls cameraOperations(camera0)
	//std::thread second (cameraOperations, camera1, argc, argv); //spawn new thread that calls cameraOperations(camera1)
	cout << "camera threads running" << endl;
		
	//Makes the main thread wait for the new thread to finish execution, therefore blocks its own execution.
	first.join();
	//second.join();

	return 0;
}

