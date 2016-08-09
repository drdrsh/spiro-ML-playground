
#include <vector>
#include <thread>
#include <iostream>
#include <sstream>
#include <itkImage.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkImageFileWriter.h>
#include <itkShrinkImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkAffineTransform.h>
#include <itkPointSet.h>
#include <itkThinPlateSplineKernelTransform.h>
#include <itkImageRandomConstIteratorWithIndex.h>
#include <itkAdditiveGaussianNoiseImageFilter.h>



#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>


#ifdef _WIN32
#include <direct.h>
#elif defined __linux__
#include <sys/stat.h>
#endif

typedef signed short    PixelType;
typedef   double        CoordinateRepType;

const unsigned int      Dimension = 3;

typedef itk::Image< PixelType, Dimension > ImageType;
typedef itk::ImageSeriesReader< ImageType > SeriesReaderType;
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::ImageFileWriter< ImageType > WriterType;

typedef itk::GDCMSeriesFileNames NamesGeneratorType;
typedef itk::IdentityTransform<CoordinateRepType, Dimension> IdentityTransformType;
typedef itk::AffineTransform<CoordinateRepType, Dimension>  AffineTransformType;
typedef itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType> NoiseFilterType;

typedef itk::ThinPlateSplineKernelTransform< CoordinateRepType, Dimension> TPSTransformType;
typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
typedef itk::LinearInterpolateImageFunction<ImageType, ImageType> InterpolatorType;

typedef   itk::Point< CoordinateRepType, Dimension>        PointType;
typedef   TPSTransformType::PointSetType				   PointSetType;
typedef   PointSetType::PointIdentifier                    PointIdType;

typedef std::vector<int> IntVector;
typedef std::vector<std::thread> ThreadVector;

bool file_exists(std::string filename) {
    return boost::filesystem::exists(filename) && boost::filesystem::file_size(filename) != 0;
}

void save_image(ImageType::Pointer image, std::string outputFilename) {
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(outputFilename);
	writer->UseCompressionOn();
	writer->SetInput(image);
	std::cout << std::endl << "Writing: " << outputFilename << std::endl;
	writer->Update();
}

ResampleImageFilterType::Pointer get_default_resampler(ImageType::Pointer inputImage) {

	ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
	resample->SetInput(inputImage);
	resample->SetSize(inputImage->GetLargestPossibleRegion().GetSize());
	resample->SetOutputSpacing(inputImage->GetSpacing());
	resample->SetOutputDirection(inputImage->GetDirection());
	resample->SetOutputOrigin(inputImage->GetOrigin());
	resample->SetDefaultPixelValue(0);
	return resample;

}

void deform_rotate(ImageType::Pointer inputImage, std::string outputFilename) {

    if(file_exists(outputFilename)) {
        std::cout << std::endl <<  outputFilename << " already exists, exiting " <<  std::endl;
        return;
    }

	float degRange = 5.0f;
	boost::random::mt19937 rng;
	boost::random::uniform_int_distribution<> idd(-50, 50);
	float angles[3] = {
		degRange * ((float)idd(rng) / 50.0f),
		degRange * ((float)idd(rng) / 50.0f),
		degRange * ((float)idd(rng) / 50.0f),
	};

	AffineTransformType::Pointer transform = AffineTransformType::New();

	const ImageType::SpacingType & spacing = inputImage->GetSpacing();
	const ImageType::PointType & origin = inputImage->GetOrigin();
	ImageType::SizeType size = inputImage->GetLargestPossibleRegion().GetSize();

	double center[3];
	for (int i = 0; i < Dimension; i++) {
		center[i] = origin[i] + spacing[i] * size[i] / 2.0;
	}

	AffineTransformType::OutputVectorType translation1;
	for (int i = 0; i < Dimension; i++) {
		translation1[i] = -center[i];
	}
	transform->Translate(translation1);

	const double degreesToRadians = std::atan(1.0) / 45.0;
	for (int i = 0; i < Dimension; i++) {
		float angleInDegrees = angles[i];
		double angle = angleInDegrees * degreesToRadians;
		AffineTransformType::OutputVectorType axis;
		axis[i] = 1;
		transform->Rotate3D(axis, angle, false);
	}

	AffineTransformType::OutputVectorType translation2;
	for (int i = 0; i < Dimension; i++) {
		translation2[i] = center[i];
	}
	transform->Translate(translation2);

	ResampleImageFilterType::Pointer resampler = get_default_resampler(inputImage);
	resampler->SetTransform(transform);
	save_image(resampler->GetOutput(), outputFilename);

}

void deform_noise(ImageType::Pointer inputImage, std::string outputFilename) {

    if(file_exists(outputFilename)) {
        std::cout << std::endl <<  outputFilename << " already exists, exiting " <<  std::endl;
        return;
    }


    float std = 100.0f;
	NoiseFilterType::Pointer filter = NoiseFilterType::New();

	filter->SetStandardDeviation(std);
	filter->SetInput(inputImage);
	save_image(filter->GetOutput(), outputFilename);
}

void deform_histogram(ImageType::Pointer inputImage, std::string outputFilename) {

}

void deform_tps(ImageType::Pointer inputImage, std::string outputFilename) {

    if(file_exists(outputFilename)) {
        std::cout << std::endl <<  outputFilename << " already exists, exiting " <<  std::endl;
        return;
    }

    float deform_factor = 0.01f;
	unsigned int deform_landmarks = 200;

	boost::random::mt19937 rng;
	boost::random::uniform_int_distribution<> iid(-100, 100);


	PointSetType::Pointer sourceLandMarks = PointSetType::New();
	PointSetType::Pointer targetLandMarks = PointSetType::New();
	PointType p1;     PointType p2;
	PointSetType::PointsContainer::Pointer sourceLandMarkContainer =
		sourceLandMarks->GetPoints();
	PointSetType::PointsContainer::Pointer targetLandMarkContainer =
		targetLandMarks->GetPoints();

	PointIdType id = itk::NumericTraits< PointIdType >::ZeroValue();

	itk::ImageRandomConstIteratorWithIndex<ImageType> it(inputImage, inputImage->GetLargestPossibleRegion());
	it.SetNumberOfSamples(deform_landmarks);
	it.GoToBegin();
	while (!it.IsAtEnd()) {
		PixelType t = it.Get();
		ImageType::IndexType idx = it.GetIndex();
		ImageType::PointType pnt;
		inputImage->TransformIndexToPhysicalPoint(idx, pnt);
		p1 = pnt;
		p2 = pnt;

		p2[0] += deform_factor * iid(rng);
		p2[1] += deform_factor * iid(rng);
		p2[2] += deform_factor * iid(rng);

		sourceLandMarkContainer->InsertElement(id, p1);
		targetLandMarkContainer->InsertElement(id++, p2);

		++it;
	}

	TPSTransformType::Pointer tps = TPSTransformType::New();
	tps->SetSourceLandmarks(sourceLandMarks);
	tps->SetTargetLandmarks(targetLandMarks);
	tps->ComputeWMatrix();

	ResampleImageFilterType::Pointer resampler = get_default_resampler(inputImage);
	resampler->SetTransform(tps);
	save_image(resampler->GetOutput(), outputFilename);
}

int main(int argc, const char* argv[]) {
	
	try {

		namespace po = boost::program_options;
		boost::program_options::options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("input,i", po::value<std::string>()->default_value(""), "Input filename.")
            ("output,o", po::value<std::string>()->default_value(""), "Output directory.")
            ("count,c", po::value<unsigned int>()->default_value(50), "Number of augmentation for the image.");

		boost::program_options::variables_map vm;
		boost::program_options::store(po::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);

		if (vm.count("help")) {
			std::cout << desc << '\n';
			return EXIT_SUCCESS;
		}

		if (!vm.count("input")) {
			std::cerr << "No input file was specified" << std::endl;
			return EXIT_FAILURE;
		}

        if (!vm.count("output")) {
            std::cerr << "No output directory was specified" << std::endl;
            return EXIT_FAILURE;
        }



        const std::string inputFilename = vm["input"].as<std::string>();
        const std::string outputDirName = vm["output"].as<std::string>();
        const unsigned int augment_count = vm["count"].as<unsigned int>();

		ReaderType::Pointer reader = ReaderType::New();
		reader->SetFileName(inputFilename);
		reader->Update();
		
		ImageType::Pointer inputImage = reader->GetOutput();

		boost::filesystem::path filenameParts(inputFilename);
		std::string basename = filenameParts.stem().generic_string();

		boost::random::mt19937 rng;
		boost::random::uniform_int_distribution<> operationDice(0, 2);

		enum operation {NOISE, TPS, ROTATE};

		ThreadVector threads;
		for (unsigned int i = 0; i < augment_count; i++) {

			int op = operationDice(rng);
			std::string outputFilename = boost::filesystem::path(outputDirName + "/" + basename + "_" + std::to_string(i) + ".nrrd").generic_string();
			switch (op) {
				case NOISE:
					threads.push_back(std::thread(deform_noise, inputImage, outputFilename));
				break;
				case TPS:
					threads.push_back(std::thread(deform_tps, inputImage, outputFilename));
					break;
				case ROTATE:
					threads.push_back(std::thread(deform_rotate, inputImage, outputFilename));
					break;
			}
		}

		for (ThreadVector::iterator it = threads.begin(); it != threads.end(); ++it) {
			/*
			* A quick and dirty way of waiting for all threads to finish
			* thread that hasn't started execution is "non-joinable" so if, for any reason, threads created in the
			* previous loop don't start execution right away, the main thread will go through this loop blazing fast
			* and exit the program killing all other threads.
			*/
			if ((*it).joinable()) {
				(*it).join();
			}
		}


	} catch (itk::ExceptionObject &ex) {
        std::cout << ex << std::endl;
        return EXIT_FAILURE;
    }
}