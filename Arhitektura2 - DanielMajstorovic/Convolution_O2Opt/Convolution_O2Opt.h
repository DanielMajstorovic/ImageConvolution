#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

class Convolution_O2Opt
{
    char* inputFilePath;
    char* outputFilePath;
    Mat convolutionKernel;
    Mat inputImage;

public:
    Convolution_O2Opt(int argc, char* argv[]);
    void readArguments(int argc, char* argv[]);
    void saveImage(Mat image);
    Mat performConvolution();
    Mat performParallelConvolution();
    String test();
};
