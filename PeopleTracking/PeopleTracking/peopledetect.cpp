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
//#define FRAME_WIDTH 640
//#define FRAME_HEIGHT 480
#define FRAME_WIDTH 320
#define FRAME_HEIGHT 240


#define BLACK cv::Scalar(0.0, 0.0, 0.0)
#define WHITE cv::Scalar(255.0, 255.0, 255.0)
#define YELLOW cv::Scalar(0.0, 255.0, 255.0)
#define GREEN cv::Scalar(0.0, 200.0, 0.0)
#define RED cv::Scalar(0.0, 0.0, 255.0)



//https://github.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp/blob/master/main.cpp



/*
void CascadeClassifier::detectMultiScale(
	const Mat& image, vector<Rect>& objects, 
	double scaleFactor=1.1, int minNeighbors=3, 
	int flags=0, Size minSize=Size(), 
	Size maxSize=Size() )


https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
scaleFactor – Parameter specifying how much the image size is reduced at each image scale.

Basically the scale factor is used to create your scale pyramid. More explanation can be found here. In short, as described here, your model has a fixed size defined during training, which is visible in the xml. This means that this size of face is detected in the image if present. However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm.

1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce size by 5%, you increase the chance of a matching size with the model for detection is found. This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.

minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.

This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.

minSize – Minimum possible object size. Objects smaller than that are ignored.

This parameter determine how small size you want to detect. You decide it! Usually, [30, 30] is a good start for face detection.

maxSize – Maximum possible object size. Objects bigger than this are ignored.

This parameter determine how big size you want to detect. Again, you decide it! Usually, you don't need to set it manually, the default value assumes you want to detect without an upper limit on the size of the face.
*/


static void detectAndDraw(const HOGDescriptor &hog, Mat &img)
{
    vector<Rect> found, found_filtered;
    double t = (double) getTickCount();
    // Run the detector with default parameters. to get a higher hit-rate
    // (and more false alarms, respectively), decrease the hitThreshold and
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
    //hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
    hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 1);
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
	cv::Mat frame1;
	cv::Mat frame2;


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

	//flag to only capture 2 frames at runtime, each loop after will only get 1 frame
	bool firstTimeThrough = true;


        while(true)
        {
		if(firstTimeThrough==true)
		{
			vc1 >> frame1; //get a first frame from camera to get difference of images
			firstTimeThrough = false;
		}
		vc1 >> frame2; //get a second frame from camera to get difference of images

		cv::Mat frame1Copy = frame1.clone();
		cv::Mat frame2Copy = frame2.clone();

		cv::Mat imgDifference;
		cv::Mat imgThresh;

		//Get images into grayscale and make images softer with GaussianBlur()
		cv::cvtColor(frame1Copy, frame1Copy, CV_BGR2GRAY);
		cv::cvtColor(frame2Copy, frame2Copy, CV_BGR2GRAY);

		cv::GaussianBlur(frame1Copy, frame1Copy, cv::Size(5, 5), 0);
		cv::GaussianBlur(frame2Copy, frame2Copy, cv::Size(5, 5), 0);

		//Take difference between images
		cv::absdiff(frame1Copy, frame2Copy, imgDifference);
		//Filter image further
		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);
		cv::imshow("imgThresh", imgThresh);

		//filter out noise and exaggerate the lines that remain using dilate()
		cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

		for (unsigned int i = 0; i < 2; i++) {
		    cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		    cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		    cv::erode(imgThresh, imgThresh, structuringElement5x5);
		}




		cv::Mat threshCopy = imgThresh.clone();

		std::vector<std::vector<cv::Point> > contours;

		cv::findContours(threshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat contourImg(threshCopy.size(), CV_8UC3, BLACK);
		cv::drawContours(contourImg, contours, -1, WHITE, -1);
		cv::imshow("threshold contours", contourImg);


		std::vector<std::vector<cv::Point> > convexHullsVector(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
		    cv::convexHull(contours[i], convexHullsVector[i]);
		}

		
		cv::Mat convexHullImg(threshCopy.size(), CV_8UC3, BLACK);
		cv::drawContours(convexHullImg, convexHullsVector, -1, WHITE, -1);
		cv::imshow("convexHulls", convexHullImg);


		// move frame 2 up to where frame 1 for the next image subtraction
		frame1 = frame2.clone();


		cv::imshow("frame1", frame1);
		//display frame count on video window
		cv::resize(frame1, frame1, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
		cv::rectangle(frame1, cv::Rect(0, 0, 900, 40), cv::Scalar(0, 0, 0), -1);
		cv::putText(frame1, cv::format("Fps: %d - %d", fps, frameCounter),
			cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));








/*
		detectAndDraw(hog, frame1);

		//display frame count on video window
		cv::resize(frame1, frame1, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
		cv::rectangle(frame1, cv::Rect(0, 0, 900, 40), cv::Scalar(0, 0, 0), -1);
		cv::putText(frame1, cv::format("Fps: %d - %d", fps, frameCounter),
			cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));


		imshow(windowName, frame1);
*/




		//delay for 1 milliseconds to keep "imshow()" from locking up
		cv::waitKey(1);


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
	std::cout << "camera objects now execute concurrently...\n";


	while(true)
	{
		//delay for 1 milliseconds
		//waitKey(1);
	}


	return 0;
}

