#include <iostream>
#include "ConvolutionUsingIntrinsicFunctions.h"

int main(int argc, char* argv[]) {
    
    ConvolutionUsingIntrinsicFunctions conv(argc, argv);


    std::cout << 22;
    conv.saveImage(conv.performConvolution());
    conv.saveImage(conv.performParallelConvolution());

    return 0;
}