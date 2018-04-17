/*
 * personObj.cpp
 *
 *  Created on: Apr 16, 2018
 *      Author: km
 */

#include "personObj.h"



personObj::personObj(cv::Rect rectangleOfPerson) {

	boundingRect = rectangleOfPerson;

	centerPosition.x = (rectangleOfPerson.x + rectangleOfPerson.x + rectangleOfPerson.width) / 2;
	centerPosition.y = (rectangleOfPerson.y + rectangleOfPerson.y + rectangleOfPerson.height) / 2;

	dblCurrentDiagonalSize = sqrt(pow(rectangleOfPerson.width, 2) + pow(rectangleOfPerson.height, 2));

	dblCurrentAspectRatio = (float)rectangleOfPerson.width / (float)rectangleOfPerson.height;

	//calculate area of bounded rectangle
	if(rectangleOfPerson.width > 0 && rectangleOfPerson.height > 0)
		dblCurrentArea = rectangleOfPerson.width * rectangleOfPerson.height;

	stillBeingTracked = false;

	int numConsecutiveFramesWithoutAMatch = 0;
}



//http://playground.arduino.cc/Main/RunningAverage
double personObj::rollingAverageCalc(double newValue) {

	// keep sum updated to improve speed.
	rollAvgSum -= rollAvgArr[rollAvgIndex];
	//add new value to rolling average at present index
	rollAvgArr[rollAvgIndex] = newValue;
	//Add new value to present sum
	rollAvgSum += rollAvgArr[rollAvgIndex];
	rollAvgIndex++;
	rollAvgIndex = rollAvgIndex % rollAvgSize;

	if (rollAvgCount < rollAvgSize)
		rollAvgCount++;

	//average out all values by dividing by the number of elements in the array
	return rollAvgSum / (double)rollAvgCount;
}
