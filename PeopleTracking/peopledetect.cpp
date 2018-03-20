#include <iostream>
#include <stdexcept>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;


#define CAMERA_NUMBER 0 //flag to set source of video: "camera 0" is the builtin laptop webcam, "camera 1" is usb webcam
#define MAX_NUM_OBJECTS 15 // Program will only track 30 objects at a time (this is just in case noise becomes a problem)
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480


const char* keys =
{
    "{ help h      |                     | print help message }"
    "{ image i     |                     | specify input image}"
    "{ camera c    |                     | enable camera capturing }"
    "{ video v     | /vtest.avi   	| use video as input }"
    "{ directory d |                     | images directory}"
};

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

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);

	if (parser.has("help"))
	{
		cout << "\nThis program demonstrates the use of the HoG descriptor using\n"
		    " HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n";
		parser.printMessage();
		cout << "During execution:\n\tHit q or ESC key to quit.\n"
		    "\tUsing OpenCV version " << CV_VERSION << "\n"
		    "Note: camera device number must be different from -1.\n" << endl;
		return 0;
	}

	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	namedWindow("people detector", 1);

	string pattern_glob = "";
	int camera_id = -1;



//from ROOBockey project:
	Mat frame;
	Mat ColorThresholded_Img0, ColorThresholded_Img, outputImg0, outputImg, src, HSV_Input;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	VideoCapture vc(CAMERA_NUMBER); //Open the Default Camera
	if (!vc.isOpened()) exit(EXIT_FAILURE); //Check if we succeeded in receiving images from camera. If not, quit program.
	vc.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); //Set height and width of capture frame
	vc.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

	vc >> frame; //get a new frame from camera

	if (!vc.isOpened())
	{
		stringstream msg;
		msg << "can't open camera: " << camera_id;
		throw runtime_error(msg.str());
	}


        while(true)
        {
		vc >> frame;

		if (frame.empty())
			break;

		detectAndDraw(hog, frame);

		imshow("people detector", frame);
		int c = waitKey( vc.isOpened() ? 30 : 0 ) & 255;
		if ( c == 'q' || c == 'Q' || c == 27)
			break;
        }

    return 0;
}
