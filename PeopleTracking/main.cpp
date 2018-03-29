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
#include "main.h" //also includes "personObj.h" inside "main.h"


using namespace cv;
using namespace std;


//#define MAX_NUM_OBJECTS 15 // Program will only track 30 objects at a time (this is just in case noise becomes a problem)
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
//#define FRAME_WIDTH 320
//#define FRAME_HEIGHT 240


#define BLACK cv::Scalar(0.0, 0.0, 0.0)
#define WHITE cv::Scalar(255.0, 255.0, 255.0)
#define YELLOW cv::Scalar(0.0, 255.0, 255.0)
#define GREEN cv::Scalar(0.0, 200.0, 0.0)
#define RED cv::Scalar(0.0, 0.0, 255.0)



//https://github.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp/blob/master/main.cpp
//https://github.com/MicrocontrollersAndMore/OpenCV_3_Multiple_Object_Tracking_by_Image_Subtraction_Cpp


bool seeDebugFrames = false;



//used to draw "contoursVector" on the display
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, BLACK);

    cv::drawContours(image, contours, -1, WHITE, -1);

    cv::imshow(strImageName, image);
}


//used to draw "peopleObjsVector" on the display
void drawAndShowContours(cv::Size imageSize, std::vector<personObj> peopleObjs, std::string strImageName) {

    cv::Mat image(imageSize, CV_8UC3, BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto &personObj : peopleObjs) {
        if (personObj.blnStillBeingTracked == true) {
            contours.push_back(personObj.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, WHITE, -1);

    cv::imshow(strImageName, image);
}


//used to find last frame object closeness to this frame (pythagorean)
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {
    
    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}



void matchCurrentFrameBlobsToExistingBlobs(std::vector<personObj> &existingPeopleObjs, std::vector<personObj> &currentFramePeopleObjs) {

    for (auto &existingPersonObj : existingPeopleObjs) {

        existingPersonObj.blnCurrentMatchFoundOrNewBlob = false;

        existingPersonObj.predictNextPosition();
    }

    for (auto &thisFramePersonObj : currentFramePeopleObjs) {

        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0;

        for (unsigned int i = 0; i < existingPeopleObjs.size(); i++) {
            if (existingPeopleObjs[i].blnStillBeingTracked == true) {
                double dblDistance = distanceBetweenPoints(thisFramePersonObj.centerPositions.back(), existingPeopleObjs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
        }

	//object in last frame is the closest object in this frame
        if (dblLeastDistance < thisFramePersonObj.dblCurrentDiagonalSize * 1.15) {
		existingPeopleObjs[intIndexOfLeastDistance].currentContour = thisFramePersonObj.currentContour;
		existingPeopleObjs[intIndexOfLeastDistance].currentBoundingRect = thisFramePersonObj.currentBoundingRect;
		existingPeopleObjs[intIndexOfLeastDistance].centerPositions.push_back(thisFramePersonObj.centerPositions.back());
		existingPeopleObjs[intIndexOfLeastDistance].dblCurrentDiagonalSize = thisFramePersonObj.dblCurrentDiagonalSize;
		existingPeopleObjs[intIndexOfLeastDistance].dblCurrentAspectRatio = thisFramePersonObj.dblCurrentAspectRatio;

		existingPeopleObjs[intIndexOfLeastDistance].blnStillBeingTracked = true;
		existingPeopleObjs[intIndexOfLeastDistance].blnCurrentMatchFoundOrNewBlob = true;
        }
        else {
                thisFramePersonObj.blnCurrentMatchFoundOrNewBlob = true;
    		existingPeopleObjs.push_back(thisFramePersonObj);
        }

    }

    for (auto &existingPersonObj : existingPeopleObjs) {

        if (existingPersonObj.blnCurrentMatchFoundOrNewBlob == false) {
            	existingPersonObj.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingPersonObj.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            	existingPersonObj.blnStillBeingTracked = false;
        }

    }

}


void drawPersonObjInfoOnImage(std::vector<personObj> &personObjsVector, cv::Mat &frame) {

    for (unsigned int i = 0; i < personObjsVector.size(); i++) {

        if (personObjsVector[i].blnStillBeingTracked == true) {
            cv::rectangle(frame, personObjsVector[i].currentBoundingRect, RED, 2);

            int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
            double dblFontScale = personObjsVector[i].dblCurrentDiagonalSize / 60.0;
            int intFontThickness = (int)std::round(dblFontScale * 1.0);

            cv::putText(frame, std::to_string(i), personObjsVector[i].centerPositions.back(), intFontFace, dblFontScale, GREEN, intFontThickness);
        }
    }
}




void cameraOperations(int cameraNum)
{
	int tick = 0;
	int fps = 0;
	long frameCounter = 0;
	std::time_t timeBegin = std::time(0);
	cv::Mat frame1;
	cv::Mat frame2;

	
	//Set source of video: "camera 0" is the builtin laptop webcam, "camera 1" is usb webcam
	VideoCapture vc1(cameraNum); //Open the Default Camera

	if (!vc1.isOpened() ) 
		exit(EXIT_FAILURE); //Check if we succeeded in receiving images from camera. If not, quit program.

	vc1.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); //Set height and width of capture frame
	vc1.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

	//flag to only capture 2 frames at runtime, each loop after will only get 1 frame
	bool firstTimeThrough = true;

	//keep track of number of detected people in the camera frames
	std::vector<personObj> peopleObjsVector;

	//Only "cout" current thread every so often
	unsigned long temp1 = 0;


        while(true)
        {
		if(firstTimeThrough==true)
		{
			vc1 >> frame1; //get a first frame from camera to get difference of images
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
		//give video window a title
		string windowName = "cam" + to_string(cameraNum) + ": windowName1:DifferenceAndThresh";
		if(seeDebugFrames)
			cv::imshow(windowName, imgThresh);

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



		//find contours in image after the filtering above is finished
		cv::Mat threshCopy = imgThresh.clone();
		std::vector<std::vector<cv::Point> > contoursVector;
		cv::findContours(threshCopy, contoursVector, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		//display contours
		cv::Mat contourImg(threshCopy.size(), CV_8UC3, BLACK);
		cv::drawContours(contourImg, contoursVector, -1, WHITE, -1);
		//give video window a title
		string windowName2 = "cam" + to_string(cameraNum) + ": 2:contours";
		if(seeDebugFrames)
			cv::imshow(windowName2, contourImg);

		//find convexhulls in image after the contour outlines are found
			//this will allow us to make the detected contours "solid" instead of "hollow"
		std::vector<std::vector<cv::Point> > convexHullsVector(contoursVector.size());

		for (unsigned int i = 0; i < contoursVector.size(); i++) {
		    cv::convexHull(contoursVector[i], convexHullsVector[i]);
		}

		cv::Mat convexHullImg(threshCopy.size(), CV_8UC3, BLACK);
		cv::drawContours(convexHullImg, convexHullsVector, -1, WHITE, -1);
		//give video window a title
		string windowName3 = "cam" + to_string(cameraNum) + ": 3:convexHulls";
		if(seeDebugFrames)
			cv::imshow(windowName3, convexHullImg);


//////////////////////////////////////////////////////////////////////
//NEW CODE: 2018/03/28
//////////////////////////////////////////////////////////////////////
		//filter out smaller objects that are not possibly people
		std::vector<personObj> peopleObjsCurrentFrameVector;

		for (auto &convexHull : convexHullsVector) {
		    personObj possiblePerson(convexHull);

			//Tune these parameters for the correct object size
		    if (possiblePerson.currentBoundingRect.area() > 100 &&
		        possiblePerson.dblCurrentAspectRatio >= 0.2 &&
		        possiblePerson.dblCurrentAspectRatio <= 1.25 &&
		        possiblePerson.currentBoundingRect.width > 20 &&
		        possiblePerson.currentBoundingRect.height > 20 &&
		        possiblePerson.dblCurrentDiagonalSize > 30.0 &&
		        (cv::contourArea(possiblePerson.currentContour) / (double)possiblePerson.currentBoundingRect.area()) > 0.40) {
		        peopleObjsCurrentFrameVector.push_back(possiblePerson);
		    }
		}


		if (firstTimeThrough == true) {
		    for (auto &currentPersonObj : peopleObjsCurrentFrameVector) {
		        peopleObjsVector.push_back(currentPersonObj);
		    }
		}
		else {
		    matchCurrentFrameBlobsToExistingBlobs(peopleObjsVector, peopleObjsCurrentFrameVector);
		}

		//give video window a title
		string windowName5 = "cam" + to_string(cameraNum) + ": 5:peopleObjsVector";
		drawAndShowContours(imgThresh.size(), peopleObjsVector, windowName5);

		frame2Copy = frame2.clone(); //get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

		drawPersonObjInfoOnImage(peopleObjsVector, frame2Copy);

		//give video window a title
		string windowName6 = "cam" + to_string(cameraNum) + ": FinalOutput";
		cv::imshow(windowName6, frame2Copy);


		//update frame count value
	        frameCounter++;
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

		//delay for 1 milliseconds to keep "imshow()" from locking up
		cv::waitKey(1);
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

