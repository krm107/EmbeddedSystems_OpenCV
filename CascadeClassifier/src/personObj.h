#ifndef SRC_PERSONOBJ_H_
#define SRC_PERSONOBJ_H_

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#define rollAvgSize 20

///////////////////////////////////////////////////////////////////////////////////////////////////
class personObj {


private:
	double rollAvgArr[rollAvgSize]; //PreviousMeasurementsArray
	int rollAvgIndex = 0;
	double rollAvgSum = 0;
	int rollAvgCount = 0;





public:
// member variables ///////////////////////////////////////////////////////////////////////////
	cv::Rect boundingRect;
	cv::Point centerPosition;

	double dblCurrentDiagonalSize = 0.0;
	double dblCurrentAspectRatio = 0.0;
	double dblCurrentArea = 0.0;

	bool stillBeingTracked = false;
	int numConsecutiveFramesWithoutAMatch = 0;



// function prototypes ////////////////////////////////////////////////////////////////////////
	personObj(cv::Rect rectangleOfPerson);
	void predictNextPosition(void);
	double rollingAverageCalc(double newValue);

};

#endif /* SRC_PERSONOBJ_H_ */
