#include "personObj.h"



personObj::personObj(cv::Rect rectangleOfPerson) {

    boundingRect = rectangleOfPerson;

    centerPosition.x = (rectangleOfPerson.x + rectangleOfPerson.x + rectangleOfPerson.width) / 2;
    centerPosition.y = (rectangleOfPerson.y + rectangleOfPerson.y + rectangleOfPerson.height) / 2;

    dblCurrentDiagonalSize = sqrt(pow(rectangleOfPerson.width, 2) + pow(rectangleOfPerson.height, 2));

    dblCurrentAspectRatio = (float)rectangleOfPerson.width / (float)rectangleOfPerson.height;

    blnStillBeingTracked = true;

    intNumOfConsecutiveFramesWithoutAMatch = 0;
}


