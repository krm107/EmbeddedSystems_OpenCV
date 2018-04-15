#ifndef Person_Obj
#define Person_Obj

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////
class personObj {


private:



public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    cv::Rect boundingRect;
    cv::Point centerPosition;

    double dblCurrentDiagonalSize;
    double dblCurrentAspectRatio;

    bool blnStillBeingTracked;
    int intNumOfConsecutiveFramesWithoutAMatch;



    // function prototypes ////////////////////////////////////////////////////////////////////////
    personObj(cv::Rect rectangleOfPerson);
    void predictNextPosition(void);

};

#endif    // Person_Obj
