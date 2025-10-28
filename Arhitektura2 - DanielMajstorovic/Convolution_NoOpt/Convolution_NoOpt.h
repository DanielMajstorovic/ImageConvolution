#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

class Convolution_NoOpt
{
    char* inputFilePath;
    char* outputFilePath;
    Mat convolutionKernel;
    Mat inputImage;

public:
    Convolution_NoOpt(int argc, char* argv[]);
    void readArguments(int argc, char* argv[]);
    void saveImage(Mat image);
    Mat performConvolution();
    Mat performParallelConvolution();
    String test();
};
