#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

class ConvolutionUsingIntrinsicFunctions
{
    char* inputFilePath;
    char* outputFilePath;
    Mat convolutionKernel;
    Mat inputImage;

public:
    ConvolutionUsingIntrinsicFunctions(int argc, char* argv[]);
    void readArguments(int argc, char* argv[]);
    void saveImage(Mat image);
    Mat performConvolution();
    Mat performParallelConvolution();
    String test();
};
