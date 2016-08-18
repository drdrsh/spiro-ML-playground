
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
#include <itkImageDuplicator.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>



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

ImageType::Pointer duplicate_image(ImageType::Pointer inputImage) {
    typedef itk::ImageDuplicator< ImageType > DuplicatorType;
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(inputImage);
    duplicator->Update();
    return duplicator->GetOutput();
}

void save_image(ImageType::Pointer image, std::string outputFilename) {
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(outputFilename);
	writer->UseCompressionOn();
	writer->SetInput(image);
	std::cout << std::endl << "Writing: " << outputFilename << std::endl;
	writer->Update();
}

ImageType::RegionType get_central_region(ImageType::Pointer inputImage) {

    ImageType::SizeType imageSize = inputImage->GetLargestPossibleRegion().GetSize();

    ImageType::IndexType regionStart;
    ImageType::SizeType regionSize;

    regionStart[0] = (imageSize[0] / 10);
    regionStart[1] = (imageSize[0] / 10);
    regionStart[2] = (imageSize[0] / 15);

    regionSize[0] = (imageSize[0] - (regionStart[0] * 2));
    regionSize[1] = (imageSize[1] - (regionStart[1] * 2));
    regionSize[2] = (imageSize[2] - (regionStart[2] * 2));

    ImageType::RegionType region;
    region.SetIndex(regionStart);
    region.SetSize(regionSize);

    return region;

}

ImageType::Pointer get_central_image(ImageType::Pointer image) {

    ImageType::Pointer inputImage = duplicate_image(image);
    ImageType::SizeType imageSize = inputImage->GetLargestPossibleRegion().GetSize();
    inputImage->SetRequestedRegion(get_central_region(inputImage));
    inputImage->Update();
    return inputImage;

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

void deform_shear(ImageType::Pointer inputImage, std::string outputFilename) {

    float low_end  = .1f;
    float high_end = .3f;

    boost::random::mt19937 rng(time(0));

    boost::random::uniform_int_distribution<> idd(-50, 50);

    float coeff = (high_end - low_end) * ((float)idd(rng) / 50.0f);
    coeff += low_end;

    boost::random::uniform_int_distribution<> axis_idd(0, 2);
    int axis1 = axis_idd(rng);
    int axis2 = -1;

    switch(axis1) {
        case 0:
            axis2 = 1;
            break;
        case 1:
            axis2 = 2;
            break;
        case 2:
            axis2 = 0;
            break;
    }

    AffineTransformType::Pointer transform = AffineTransformType::New();
    /*

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
   */
    //const double degreesToRadians = std::atan(1.0) / 45.0;
    //double angle = coeff * degreesToRadians;
    transform->Shear(axis1, axis2,  coeff, false);
    /*
    AffineTransformType::OutputVectorType translation2;
    for (int i = 0; i < Dimension; i++) {
        translation2[i] = center[i];
    }
    transform->Translate(translation2);
    */

    ResampleImageFilterType::Pointer resampler = get_default_resampler(inputImage);
    resampler->SetTransform(transform);
    save_image(resampler->GetOutput(), outputFilename);

}

void deform_rotate(ImageType::Pointer inputImage, std::string outputFilename) {


	float degRange = 5.0f;
	boost::random::mt19937 rng(time(0));
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

    float std = 100.0f;
	NoiseFilterType::Pointer filter = NoiseFilterType::New();

	filter->SetStandardDeviation(std);
	filter->SetInput(inputImage);
    filter->GetOutput()->SetRequestedRegion(get_central_region(inputImage));

	save_image(filter->GetOutput(), outputFilename);
}

void deform_histogram(ImageType::Pointer inputImage, std::string outputFilename) {

}


void deform_tps(ImageType::Pointer inputImage, std::string outputFilename) {



	boost::random::mt19937 rng(time(0));
	boost::random::uniform_int_distribution<> iid(-100, 100);

    float rnd = (float)iid(rng);
	float deform_factor = 0.3f * (float)( (rnd + 1) / 100.0f);
	unsigned int deform_landmarks = 500;

    ImageType::SizeType imageSize = inputImage->GetLargestPossibleRegion().GetSize();


    PointSetType::Pointer sourceLandMarks = PointSetType::New();
	PointSetType::Pointer targetLandMarks = PointSetType::New();
	PointSetType::PointsContainer::Pointer sourceLandMarkContainer = sourceLandMarks->GetPoints();
	PointSetType::PointsContainer::Pointer targetLandMarkContainer = targetLandMarks->GetPoints();

	PointIdType id = itk::NumericTraits< PointIdType >::ZeroValue();

    ImageType::SpacingType sp = inputImage->GetSpacing();
	itk::ImageRandomConstIteratorWithIndex<ImageType> it(inputImage, get_central_region(inputImage) );

    it.SetNumberOfSamples(deform_landmarks);
    it.GoToBegin();
	while (!it.IsAtEnd()) {

        PointType p1;
        PointType p2;

		ImageType::IndexType idx = it.GetIndex();
		PointType pnt;
		inputImage->TransformIndexToPhysicalPoint(idx, pnt);
        p1 = pnt;
        p2 = pnt;
        double randF = deform_factor * iid(rng);
        for(int i=0;i <Dimension; i++) {
            p2[i] += randF;
            ImageType::IndexType midx;
            inputImage->TransformPhysicalPointToIndex(p2, midx);
            if(midx[i] > imageSize[i]) {
                p2[i] -= (randF * 2);
            }
        }

        //std::cout << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
        //std::cout << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;

        sourceLandMarkContainer->InsertElement(id,   p1);
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

		boost::random::mt19937 rng(time(0));

        double op_prob[] = {0.0, 0.5, 0.3, 0.2};
        boost::random::discrete_distribution<unsigned int> operationDice(op_prob);
		enum operation {NOISE, TPS, ROTATE, SHEAR};

		ThreadVector threads;
		for (unsigned int i = 0; i < augment_count; i++) {

			int op = operationDice(rng);
			std::string outputFilename = boost::filesystem::path(outputDirName + "/" + basename + "_" + std::to_string(i) + ".nrrd").generic_string();
            std::cout << "Performing opetration " << op << " on image " << i << std::endl;
            if(file_exists(outputFilename)) {
                std::cout << std::endl <<  outputFilename << " already exists, exiting " <<  std::endl;
                continue;
            }
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
                case SHEAR:
                    threads.push_back(std::thread(deform_shear, inputImage, outputFilename));
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