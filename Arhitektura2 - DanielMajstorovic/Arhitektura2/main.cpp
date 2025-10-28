#include <iostream>
#include <fstream>
#include <sstream>
#include "Convolution_NoOpt.h"
#include "Convolution_O1Opt.h"
#include "Convolution_O2Opt.h"
#include "Convolution_OXOpt.h"
#include "ConvolutionUsingIntrinsicFunctions.h"

std::string modifyFileName(const std::string& originalPath, const std::string& suffix);
std::string removeFirstTwoLines(const std::string& input);

int main(int argc, char* argv[]) {

    std::ofstream outFile("rezultati.txt");

    if (!outFile.is_open()) {
        std::cerr << "Fajl nije otvoren!" << std::endl;
        return 1;
    }

    Convolution_NoOpt cNoOpt(argc, argv);
    std::string noOptTestResult = cNoOpt.test();
    std::cout << noOptTestResult << std::endl;
    imwrite(modifyFileName(argv[2], "NoOptSeq"), cNoOpt.performConvolution());
    imwrite(modifyFileName(argv[2], "NoOptPar"), cNoOpt.performParallelConvolution());
    outFile << noOptTestResult;

    Convolution_O1Opt cO1Opt(argc, argv);
    std::string o1OptTestResult = cO1Opt.test();
    std::cout << o1OptTestResult << std::endl;
    imwrite(modifyFileName(argv[2], "O1OptSeq"), cO1Opt.performConvolution());
    imwrite(modifyFileName(argv[2], "O1OptPar"), cO1Opt.performParallelConvolution());
    outFile << removeFirstTwoLines(o1OptTestResult);

    Convolution_O2Opt cO2Opt(argc, argv);
    std::string o2OptTestResult = cO2Opt.test();
    std::cout << o2OptTestResult << std::endl;
    imwrite(modifyFileName(argv[2], "O2OptSeq"), cO2Opt.performConvolution());
    imwrite(modifyFileName(argv[2], "O2OptPar"), cO2Opt.performParallelConvolution());
    outFile << removeFirstTwoLines(o2OptTestResult);

    Convolution_OXOpt cOXOpt(argc, argv);
    std::string oXOptTestResult = cOXOpt.test();
    std::cout << oXOptTestResult << std::endl;
    imwrite(modifyFileName(argv[2], "OXOptSeq"), cOXOpt.performConvolution());
    imwrite(modifyFileName(argv[2], "OXOptPar"), cOXOpt.performParallelConvolution());
    outFile << removeFirstTwoLines(oXOptTestResult);

    ConvolutionUsingIntrinsicFunctions cUIF(argc, argv);
    std::string uifTestResult = cUIF.test();
    std::cout << uifTestResult << std::endl;
    imwrite(modifyFileName(argv[2], "IntrinsicsSeq"), cUIF.performConvolution());
    imwrite(modifyFileName(argv[2], "IntrinsicsPar"), cUIF.performParallelConvolution());
    outFile << removeFirstTwoLines(uifTestResult);

    outFile.close();

    return 0;
}

std::string modifyFileName(const std::string& originalPath, const std::string& suffix) {
    size_t dotPosition = originalPath.find_last_of(".");

    if (dotPosition != std::string::npos) {
        std::string fileName = originalPath.substr(0, dotPosition);
        std::string fileExtension = originalPath.substr(dotPosition);
        return fileName + "_" + suffix + fileExtension;
    }
    else {
        return originalPath + "_" + suffix;
    }
}

std::string removeFirstTwoLines(const std::string& input) {
    std::istringstream stream(input);
    std::string line;
    std::string result;
    int lineCount = 0;

    while (std::getline(stream, line)) {
        if (lineCount >= 2) {
            result += line + "\n";
        }
        lineCount++;
    }

    return result;
}