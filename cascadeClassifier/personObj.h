#ifndef Person_Obj
#define Person_Obj

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////
class personObj {
public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> currentContour;

    cv::Rect currentBoundingRect;

    std::vector<cv::Point> centerPositions;

    double dblCurrentDiagonalSize;
    double dblCurrentAspectRatio;

    bool blnCurrentMatchFoundOrNewBlob;

    bool blnStillBeingTracked;

    int intNumOfConsecutiveFramesWithoutAMatch;

    cv::Point predictedNextPosition;
    
    // function prototypes ////////////////////////////////////////////////////////////////////////
    personObj(std::vector<cv::Point> _contour);
    void predictNextPosition(void);

};

#endif    // Person_Obj
