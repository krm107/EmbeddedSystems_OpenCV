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

using namespace cv;
using namespace std;


//#define MAX_NUM_OBJECTS 15 // Program will only track 30 objects at a time (this is just in case noise becomes a problem)
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480





static void detectAndDraw(const HOGDescriptor &hog, Mat &img)
{
    vector<Rect> found, found_filtered;
    double t = (double) getTickCount();
    // Run the detector with default parameters. to get a higher hit-rate
    // (and more false alarms, respectively), decrease the hitThreshold and
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
    hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
    t = (double) getTickCount() - t;
    cout << "detection time = " << (t*1000./cv::getTickFrequency()) << " ms" << endl;

    for(size_t i = 0; i < found.size(); i++ )
    {
        Rect r = found[i];

        size_t j;
        // Do not add small detections inside a bigger detection.
        for ( j = 0; j < found.size(); j++ )
            if ( j != i && (r & found[j]) == r )
                break;

        if ( j == found.size() )
            found_filtered.push_back(r);
    }

    for (size_t i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];

        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
    }
}




void cameraOperations(int cameraNum)
{
	int camera_id = -1;
	int tick = 0;
	int fps = 0;
	long frameCounter = 0;
	std::time_t timeBegin = std::time(0);
	Mat frame1;


	//create "Histogram of Oriented Gradients"
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	
	//Set source of video: "camera 0" is the builtin laptop webcam, "camera 1" is usb webcam
	VideoCapture vc1(cameraNum); //Open the Default Camera

	if (!vc1.isOpened() ) 
		exit(EXIT_FAILURE); //Check if we succeeded in receiving images from camera. If not, quit program.

	vc1.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); //Set height and width of capture frame
	vc1.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);


	//give video window a title
	string windowName = "people detector" + to_string(cameraNum);


        while(true)
        {
		vc1 >> frame1; //get a new frame from camera
		if ( frame1.empty() )
		{
			cout << "Camera " << cameraNum << " " << "could not be opened" << endl;
			break;
		}

		detectAndDraw(hog, frame1);

		//display frame count on video window
		cv::resize(frame1, frame1, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
		cv::rectangle(frame1, cv::Rect(0, 0, 900, 40), cv::Scalar(0, 0, 0), -1);
		cv::putText(frame1, cv::format("Frames per second: %d - %d", fps, frameCounter), 				cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));


		imshow(windowName, frame1);


		//delay for 1 milliseconds
		waitKey(1);


		//update frame count value
	        frameCounter++;
		std::time_t timeNow = std::time(0) - timeBegin;

		if (timeNow - tick >= 1)
		{
		    tick++;
		    fps = frameCounter;
		    frameCounter = 0;
		}


		//convey which thread is being run at a time
		cout << cameraNum << endl;
        }
}




int main(void)
{
	//run camera operations in separate threads
	//Thread info from "http://www.cplusplus.com/reference/thread/thread/"
	int camera0 = 0;
	int camera1 = 1;

	std::thread first (cameraOperations, camera0); //spawn new thread that calls cameraOperations(camera0)
	std::thread second (cameraOperations, camera1); //spawn new thread that calls cameraOperations(camera1)
	std::cout << "main, foo and bar now execute concurrently...\n";


	while(true)
	{
		//delay for 1 milliseconds
		//waitKey(1);
	}


	return 0;
}

