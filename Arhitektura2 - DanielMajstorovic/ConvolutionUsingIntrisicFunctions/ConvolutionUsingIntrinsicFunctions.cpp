#include "ConvolutionUsingIntrinsicFunctions.h"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <immintrin.h>

ConvolutionUsingIntrinsicFunctions::ConvolutionUsingIntrinsicFunctions(int argc, char* argv[])
{
	readArguments(argc, argv);
}

void ConvolutionUsingIntrinsicFunctions::readArguments(int argc, char* argv[])
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

void ConvolutionUsingIntrinsicFunctions::saveImage(Mat image)
{
	imwrite(outputFilePath, image);
}

Mat ConvolutionUsingIntrinsicFunctions::performConvolution()
{
	int kernelRowsSizeHalf = convolutionKernel.rows / 2;
	int kernelColsSizeHalf = convolutionKernel.cols / 2;

	inputImage.convertTo(inputImage, CV_64FC3);

	// Prosirena originalna slika
	Mat expandedImage(inputImage.rows + convolutionKernel.rows - kernelRowsSizeHalf, inputImage.cols + convolutionKernel.cols - kernelColsSizeHalf, CV_64FC3, Scalar(0, 0, 0));
	for (int x = 0; x < inputImage.rows; x++) {
		for (int y = 0; y < inputImage.cols; y++) {
			expandedImage.at<Vec3d>(x + kernelRowsSizeHalf, y + kernelColsSizeHalf) = inputImage.at<Vec3d>(x, y);
		}
	}

	// Rezultujuca slika
	Mat resultImage(inputImage.rows, inputImage.cols, CV_64FC3, Scalar(0, 0, 0));

	// Inicijalizuj AVX registre
	__m256d rgb_vec, kernel_vec, result_vec, temp_vec;

	for (int x = kernelRowsSizeHalf; x < expandedImage.rows - kernelRowsSizeHalf; x++) {
		for (int y = kernelColsSizeHalf; y < expandedImage.cols - kernelColsSizeHalf; y++) {
			// Postavi result_vec na 0
			result_vec = _mm256_setzero_pd();

			for (int u = -kernelRowsSizeHalf; u <= kernelRowsSizeHalf; u++) {
				for (int v = -kernelColsSizeHalf; v <= kernelColsSizeHalf; v++) {

					// Učitaj R, G, B i postavi četvrtu vrednost na 0
					rgb_vec = _mm256_set_pd(0.0,
						expandedImage.at<Vec3d>(x + u, y + v)[2], // B komponenta
						expandedImage.at<Vec3d>(x + u, y + v)[1], // G komponenta
						expandedImage.at<Vec3d>(x + u, y + v)[0]); // R komponenta

					// Učitaj kernel vrednost i dupliraj je u AVX registar
					double kernel_val = convolutionKernel.at<double>(u + kernelRowsSizeHalf, v + kernelColsSizeHalf);
					kernel_vec = _mm256_set1_pd(kernel_val);

					// Pomnoži RGB vrednosti sa kernel vrednostima
					temp_vec = _mm256_mul_pd(rgb_vec, kernel_vec);

					// Saberi rezultate
					result_vec = _mm256_add_pd(result_vec, temp_vec);
				}
			}

			// Ekstrakcija rezultata (samo prva tri elementa)
			double r = ((double*)&result_vec)[0]; // R komponenta
			double g = ((double*)&result_vec)[1]; // G komponenta
			double b = ((double*)&result_vec)[2]; // B komponenta

			// Sačuvaj rezultate u rezultujuću sliku
			resultImage.at<Vec3d>(x - kernelRowsSizeHalf, y - kernelColsSizeHalf) = Vec3d(r, g, b);
		}
	}



	resultImage.convertTo(resultImage, CV_8UC3); // Konvertovanje natrag u 8-bitni format
	return resultImage;
}

Mat ConvolutionUsingIntrinsicFunctions::performParallelConvolution()
{
	int kernelRowsSizeHalf = convolutionKernel.rows / 2;
	int kernelColsSizeHalf = convolutionKernel.cols / 2;

	inputImage.convertTo(inputImage, CV_64FC3);

	// Prosirena originalna slika
	Mat expandedImage(inputImage.rows + convolutionKernel.rows - kernelRowsSizeHalf, inputImage.cols + convolutionKernel.cols - kernelColsSizeHalf, CV_64FC3, Scalar(0, 0, 0));
	for (int x = 0; x < inputImage.rows; x++) {
		for (int y = 0; y < inputImage.cols; y++) {
			expandedImage.at<Vec3d>(x + kernelRowsSizeHalf, y + kernelColsSizeHalf) = inputImage.at<Vec3d>(x, y);
		}
	}

	// Rezultujuca slika
	Mat resultImage(inputImage.rows, inputImage.cols, CV_64FC3, Scalar(0, 0, 0));

	// Paralelizacija spoljašnjih petlji
#pragma omp parallel for
	for (int x = kernelRowsSizeHalf; x < expandedImage.rows - kernelRowsSizeHalf; x++) {
		for (int y = kernelColsSizeHalf; y < expandedImage.cols - kernelColsSizeHalf; y++) {
			// Inicijalizacija AVX registra
			__m256d result_vec = _mm256_setzero_pd();

			for (int u = -kernelRowsSizeHalf; u <= kernelRowsSizeHalf; u++) {
				for (int v = -kernelColsSizeHalf; v <= kernelColsSizeHalf; v++) {
					// Učitaj R, G, B i postavi četvrtu vrednost na 0
					__m256d rgb_vec = _mm256_set_pd(0.0,
						expandedImage.at<Vec3d>(x + u, y + v)[2], // B komponenta
						expandedImage.at<Vec3d>(x + u, y + v)[1], // G komponenta
						expandedImage.at<Vec3d>(x + u, y + v)[0]); // R komponenta

					// Učitaj kernel vrednost
					double kernel_val = convolutionKernel.at<double>(u + kernelRowsSizeHalf, v + kernelColsSizeHalf);
					__m256d kernel_vec = _mm256_set1_pd(kernel_val);

					// Pomnoži RGB vrednosti sa kernel vrednostima
					__m256d temp_vec = _mm256_mul_pd(rgb_vec, kernel_vec);

					// Saberi rezultate
					result_vec = _mm256_add_pd(result_vec, temp_vec);
				}
			}

			// Ekstrakcija rezultata (samo prva tri elementa)
			double r = ((double*)&result_vec)[0]; // R komponenta
			double g = ((double*)&result_vec)[1]; // G komponenta
			double b = ((double*)&result_vec)[2]; // B komponenta

			// Sačuvaj rezultate u rezultujuću sliku
			resultImage.at<Vec3d>(x - kernelRowsSizeHalf, y - kernelColsSizeHalf) = Vec3d(r, g, b);
		}
	}

	return resultImage;
}


String ConvolutionUsingIntrinsicFunctions::test()
{
	int testIterations = 3;
	int warmUpIterations = 3;
	String log = "Dimenzija slike: ";
	log += to_string(inputImage.cols) + " x " + to_string(inputImage.rows);
	log += "\nSlika na putanji: ";
	log += inputFilePath;
	log += "\nIntrinzicne funkcije: Srednje vrijeme: ";

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

	log += "\nIntrinzicne funkcije (paralelno): Srednje vrijeme: ";

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


