#include "Convolution_O2Opt.h"
#include <stdexcept>
#include <vector>
#include <cmath>

Convolution_O2Opt::Convolution_O2Opt(int argc, char* argv[])
{
	readArguments(argc, argv);
}

void Convolution_O2Opt::readArguments(int argc, char* argv[])
{

	if (argc < 3) {
		throw invalid_argument("Unesite dovoljan broj argumenata!");
	}
	inputFilePath = argv[1];
	outputFilePath = argv[2];
	inputImage = imread(inputFilePath);

	if (argc == 3) {
		// Podrazumijevano detekcija horizontalnih ivica
		double defaultKernel[9] = {
			-1, -1, -1,
			2, 2, 2,
			-1, -1, -1
		};
		convolutionKernel = Mat(3, 3, CV_64F, defaultKernel).clone(); // clone jer se defaultKernel dealocira
	}
	else
	{
		int size = argc - 3;
		double sqrtSize = sqrt(size);

		// Provjera da li je kernel kvadratnog oblika i neparne dimenzije
		if (size % 2 == 0 || sqrtSize != floor(sqrtSize)) {
			throw invalid_argument("Dimenzija kernela nije odgovarajuca");
		}

		// Ucitavanje kernela iz argumenata komandne linije
		vector<double> kernelArray(size);
		for (int i = 3; i < argc; i++) {
			kernelArray[i - 3] = atof(argv[i]);
		}

		// Kreiranje kernela koristeci ucitane vrijednosti
		convolutionKernel = Mat((int)sqrtSize, (int)sqrtSize, CV_64F, kernelArray.data()).clone();
	}
}

void Convolution_O2Opt::saveImage(Mat image)
{
	imwrite(outputFilePath, image);
}

Mat Convolution_O2Opt::performConvolution()
{
	int kernelRowsSizeHalf = convolutionKernel.rows / 2;
	int kernelColsSizeHalf = convolutionKernel.cols / 2;

	inputImage.convertTo(inputImage, CV_64FC3);
	// expandedImage je prosirena originalna slika (da bi centar kernela kretao od pocetka originalnog sadrzaja)
	Mat expandedImage(inputImage.rows + convolutionKernel.rows - kernelRowsSizeHalf, inputImage.cols + convolutionKernel.cols - kernelColsSizeHalf, CV_64FC3, Scalar(0, 0, 0));
	for (int x = 0; x < inputImage.rows; x++) {
		for (int y = 0; y < inputImage.cols; y++) {
			expandedImage.at<Vec3d>(x + kernelRowsSizeHalf, y + kernelColsSizeHalf) = inputImage.at<Vec3d>(x, y);
		}
	}
	// Izracunavanje piksela rezultujuce slike
	Mat resultImage(inputImage.rows, inputImage.cols, CV_64FC3, Scalar(0, 0, 0));
	// Centar kernela se krece po originalnom sadrzaju koji se nalazi u expandedImage
	for (int x = kernelRowsSizeHalf; x < expandedImage.rows - kernelRowsSizeHalf; x++) {
		for (int y = kernelColsSizeHalf; y < expandedImage.cols - kernelColsSizeHalf; y++) {
			// Racunanje piksela rezultujuce slike
			double r = 0, g = 0, b = 0;
			for (int u = -kernelRowsSizeHalf; u <= kernelRowsSizeHalf; u++) {
				for (int v = -kernelColsSizeHalf; v <= kernelColsSizeHalf; v++) {
					r += expandedImage.at<Vec3d>(x + u, y + v)[0] * convolutionKernel.at<double>(u + kernelRowsSizeHalf, v + kernelColsSizeHalf);
					g += expandedImage.at<Vec3d>(x + u, y + v)[1] * convolutionKernel.at<double>(u + kernelRowsSizeHalf, v + kernelColsSizeHalf);
					b += expandedImage.at<Vec3d>(x + u, y + v)[2] * convolutionKernel.at<double>(u + kernelRowsSizeHalf, v + kernelColsSizeHalf);
				}
			}
			resultImage.at<Vec3d>(x - kernelRowsSizeHalf, y - kernelColsSizeHalf) = Vec3d(r, g, b);
		}
	}
	resultImage.convertTo(resultImage, CV_8UC3);
	return resultImage;
}

Mat Convolution_O2Opt::performParallelConvolution()
{
	int kernelRowsSizeHalf = convolutionKernel.rows / 2;
	int kernelColsSizeHalf = convolutionKernel.cols / 2;

	inputImage.convertTo(inputImage, CV_64FC3);
	// expandedImage je prosirena originalna slika (da bi centar kernela kretao od pocetka originalnog sadrzaja)
	Mat expandedImage(inputImage.rows + convolutionKernel.rows - kernelRowsSizeHalf, inputImage.cols + convolutionKernel.cols - kernelColsSizeHalf, CV_64FC3, Scalar(0, 0, 0));
#pragma omp parallel for schedule(static, 2)
	for (int x = 0; x < inputImage.rows; x++) {
		for (int y = 0; y < inputImage.cols; y++) {
			expandedImage.at<Vec3d>(x + kernelRowsSizeHalf, y + kernelColsSizeHalf) = inputImage.at<Vec3d>(x, y);
		}
	}
	// Izracunavanje piksela rezultujuce slike
	Mat resultImage(inputImage.rows, inputImage.cols, CV_64FC3, Scalar(0, 0, 0));
	// Centar kernela se krece po originalnom sadrzaju koji se nalazi u expandedImage
#pragma omp parallel for schedule(static, 2)
	for (int x = kernelRowsSizeHalf; x < expandedImage.rows - kernelRowsSizeHalf; x++) {
		for (int y = kernelColsSizeHalf; y < expandedImage.cols - kernelColsSizeHalf; y++) {
			// Racunanje piksela rezultujuce slike
			double r = 0, g = 0, b = 0;
#pragma omp parallel for reduction (+:r,g,b) 
			for (int u = -kernelRowsSizeHalf; u <= kernelRowsSizeHalf; u++) {
				for (int v = -kernelColsSizeHalf; v <= kernelColsSizeHalf; v++) {
					r += expandedImage.at<Vec3d>(x + u, y + v)[0] * convolutionKernel.at<double>(u + kernelRowsSizeHalf, v + kernelColsSizeHalf);
					g += expandedImage.at<Vec3d>(x + u, y + v)[1] * convolutionKernel.at<double>(u + kernelRowsSizeHalf, v + kernelColsSizeHalf);
					b += expandedImage.at<Vec3d>(x + u, y + v)[2] * convolutionKernel.at<double>(u + kernelRowsSizeHalf, v + kernelColsSizeHalf);
				}
			}
			resultImage.at<Vec3d>(x - kernelRowsSizeHalf, y - kernelColsSizeHalf) = Vec3d(r, g, b);
		}
	}
	resultImage.convertTo(resultImage, CV_8UC3);
	return resultImage;
}

String Convolution_O2Opt::test()
{
	int testIterations = 3;
	int warmUpIterations = 3;
	String log = "Dimenzija slike: ";
	log += to_string(inputImage.cols) + " x " + to_string(inputImage.rows);
	log += "\nSlika na putanji: ";
	log += inputFilePath;
	log += "\nO2 optimizacija, sekvencijalno izvrsavanje: Srednje vrijeme: ";

	// Prethodno pokretanje (zagrijavanje)
	for (int i = 0; i < warmUpIterations; i++) {
		performConvolution();
	}

	// Vremena za serijsko izvrsavanje
	std::vector<double> times(testIterations);

	double totalTime = 0;
	for (int i = 0; i < testIterations; i++) {
		double start = omp_get_wtime();
		Mat img = performConvolution();
		double end = omp_get_wtime();
		times[i] = end - start;
		totalTime += times[i];
	}

	// Izracunavanje srednje vrednosti
	double avgTime = totalTime / testIterations;

	// Izracunavanje varijanse
	double tmpSum = 0;
	for (int i = 0; i < testIterations; i++) {
		double diff = avgTime - times[i];
		tmpSum += (diff * diff);
	}
	double varianse = tmpSum / (testIterations - 1);

	log += std::to_string(avgTime);
	log += " Varijansa: ";
	log += std::to_string(varianse);

	log += "\nO2 optimizacija, paralelno izvrsavanje: Srednje vrijeme: ";

	// Prethodno pokretanje za paralelno izvrsavanje
	for (int i = 0; i < warmUpIterations; i++) {
		performParallelConvolution();
	}

	// Vremena za paralelno izvrsavanje
	totalTime = 0;
	for (int i = 0; i < testIterations; i++) {
		double start = omp_get_wtime();
		Mat img = performParallelConvolution();
		double end = omp_get_wtime();
		times[i] = end - start;
		totalTime += times[i];
	}

	// Ponovno izracunavanje srednje vrednosti
	avgTime = totalTime / testIterations;

	// Ponovno izracunavanje varijanse
	tmpSum = 0;
	for (int i = 0; i < testIterations; i++) {
		double diff = avgTime - times[i];
		tmpSum += (diff * diff);
	}
	varianse = tmpSum / (testIterations - 1);

	log += std::to_string(avgTime);
	log += " Varijansa: ";
	log += std::to_string(varianse);

	return log;
}


