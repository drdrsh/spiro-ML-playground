
#include <fstream>
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
#include <boost/thread/thread.hpp>
#include <boost/lockfree/queue.hpp>

#include <mutex>

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

typedef void(*OperationFunction)(ImageType::Pointer, std::string, std::vector<float>);

typedef std::vector<int> IntVector;
typedef std::vector<std::thread> ThreadVector;

std::mutex log_mutex;
std::mutex cout_mutex;

std::ofstream log_file;

//Thread-safe logging macro to allow simple std::cout style logging
#define LOG_LINE(X) log_mutex.lock();  log_file << X << std::endl;   log_mutex.unlock();

#define COUT_LINE(X) cout_mutex.lock();  std::cout << X << std::endl;   cout_mutex.unlock();


ImageType::Pointer globalInputImage;

class ImageJob {

public:
    char outputFilename[2048];
    OperationFunction operation;
	float params[20];

    ImageJob() {}

    ImageJob(const ImageJob& rhs) {
        strncpy(outputFilename, rhs.outputFilename, 2048);
		memcpy(params, rhs.params, 20 * sizeof(float));
        operation = rhs.operation;
    }


    ImageJob( OperationFunction p, std::string o, std::vector<float> f) {
        operation = p;
		std::size_t length = o.copy(outputFilename, 2048, 0);
		outputFilename[length] = '\0';
		unsigned int it_max = 20;
		if (f.size() < it_max) {
			it_max = f.size();
		}
		for (unsigned int i = 0; i < it_max; ++i) {
			params[i] = f[i];
		}
    }
};

boost::lockfree::queue<ImageJob> queue(128);

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
	COUT_LINE("Writing: " << outputFilename);
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

void deform_shear(ImageType::Pointer inputImage, std::string outputFilename, std::vector<float> params) {

    float low_end  = params[0];
    float high_end = params[1];

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

	LOG_LINE("Performing Shear with coeff " << coeff << " on axes " << axis1 << ", " << axis2);

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

void deform_rotate(ImageType::Pointer inputImage, std::string outputFilename, std::vector<float> params) {

	float min_deg = params[0];
	float max_deg = params[1];

	boost::random::mt19937 rng(time(0));
	boost::random::uniform_int_distribution<> idd(-500, 500);

    float x = (float)idd(rng);
    float y = (float)idd(rng);
    float z = (float)idd(rng);
	float range_deg = max_deg - min_deg;
	float angles[3] = {
		min_deg + range_deg * (x / 500.0f),
		min_deg + range_deg * (y / 500.0f),
		min_deg + range_deg * (z / 500.0f)
	};

	LOG_LINE("Performing Rotation with angles " << angles[0] << ", " << angles[1] << ", " << angles[2]);

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

void deform_noise(ImageType::Pointer inputImage, std::string outputFilename, std::vector<float> params) {

	float std = params[0];
	NoiseFilterType::Pointer filter = NoiseFilterType::New();

	LOG_LINE("Performing Noise with params " << params[0]);

	filter->SetStandardDeviation(std);
	filter->SetInput(inputImage);
    filter->GetOutput()->SetRequestedRegion(get_central_region(inputImage));

	save_image(filter->GetOutput(), outputFilename);
}


void deform_histogram(ImageType::Pointer inputImage, std::string outputFilenam, std::vector<float> paramse) {

}


void deform_tps(ImageType::Pointer inputImage, std::string outputFilename, std::vector<float> params) {



	boost::random::mt19937 rng(time(0));
	boost::random::uniform_int_distribution<> iid(-100, 100);

    float rnd = (float)iid(rng);
	unsigned int deform_landmarks = (unsigned int)params[0];
	float deform_factor = params[1] * (rnd / 100.0f);

	LOG_LINE("Performing TPS with params " << params[0] << ", " << deform_factor);

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




void process_entry(void) {

    ImageJob job;

    while(queue.pop(job)) {
        std::string o = job.outputFilename;
		std::vector<float> p(std::begin(job.params), std::end(job.params));
        job.operation(globalInputImage, o, p);
    }

}

int main(int argc, const char* argv[]) {
	
	try {
		namespace po = boost::program_options;
		boost::program_options::options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("input,i", po::value<std::string>()->default_value(""), "Input filename.")
            ("output,o", po::value<std::string>()->default_value(""), "Output directory.")
            ("count,c", po::value<unsigned int>()->default_value(50), "Number of augmentation for the image.")
            ("threads,t", po::value<unsigned int>()->default_value(8), "Number of threads to use.")
            ("tps-count,a", po::value<unsigned int>()->default_value(200), "Thin plate spline point count.")
            ("tps-factor,b", po::value<float>()->default_value(0.1f), "Thin plate spline deform factor.")
            ("tps-prob,T", po::value<float>()->default_value(0.5f), "Percent of TPS deformation in output images.")
            ("rotate-min-deg,d", po::value<float>()->default_value(0.0f), "Minimum rotation angle in degrees.")
            ("rotate-max-deg,e", po::value<float>()->default_value(5.0f), "Maximum rotation angle in degrees.")
            ("rotate-prob,R", po::value<float>()->default_value(0.5f), "Percent of rotated images in output images")
            ("shear-min-fac,f", po::value<float>()->default_value(0.1f), "Minimum shear factor.")
            ("shear-max-fac,g", po::value<float>()->default_value(0.3f), "Maximum shear factor.")
            ("shear-prob,S", po::value<float>()->default_value(0.0f), "Percent of sheared images in output images.")
            ("noise-std,j", po::value<float>()->default_value(100.0f), "Standard deviation for gaussian noise.")
            ("noise-prob,N", po::value<float>()->default_value(0.0f), "Percent of gaussian noise in output images.");

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



		

        std::string inputFilename = vm["input"].as<std::string>();
        std::string outputDirName = vm["output"].as<std::string>();
        unsigned int augment_count = vm["count"].as<unsigned int>();
        unsigned int number_of_threads = vm["threads"].as<unsigned int>();
		unsigned int tps_count = vm["tps-count"].as<unsigned int>();
		float tps_factor = vm["tps-factor"].as<float>();
		float tps_prob = vm["tps-prob"].as<float>();
		float rot_min_deg = vm["rotate-min-deg"].as<float>();
		float rot_max_deg = vm["rotate-max-deg"].as<float>();
		float rot_prob = vm["rotate-prob"].as<float>();
		float shear_min_fac = vm["shear-min-fac"].as<float>();
		float shear_max_fac = vm["shear-min-fac"].as<float>();
		float shear_prob = vm["shear-prob"].as<float>();
		float noise_std = vm["noise-std"].as<float>();
		float noise_prob = vm["noise-prob"].as<float>();


		float prob_max = noise_prob + shear_prob + tps_prob + rot_prob;
		noise_prob /= prob_max;
		shear_prob /= prob_max;
		tps_prob /= prob_max;
		rot_prob /= prob_max;

		if (number_of_threads <= 0) {
			std::cerr << "Invalid number of threads." << std::endl;
			return EXIT_FAILURE;
		}

		if (augment_count <= 0) {
			std::cerr << "Invalid number of replications." << std::endl;
			return EXIT_FAILURE;
		}

		if (prob_max <= 0) {
			std::cerr << "Invalid filter probability values." << std::endl;
			return EXIT_FAILURE;
		}

		if (shear_min_fac > shear_max_fac) {
			std::cerr << "Shear minimum factor has to be less than or equal to shear maximum factor." << std::endl;
			return EXIT_FAILURE;
		}

		if (rot_min_deg > rot_max_deg) {
			std::cerr << "Minimum rotation degrees has to be less than or equal to maximum rotation degrees." << std::endl;
			return EXIT_FAILURE;
		}

		if (noise_std < 0) {
			std::cerr << "Noise standard deviation cannot be less than 0." << std::endl;
			return EXIT_FAILURE;
		}

		if (tps_count <= 0) {
			std::cerr << "Thin plate spline point count has to be an integer greater than 0." << std::endl;
			return EXIT_FAILURE;
		}


		ReaderType::Pointer reader = ReaderType::New();
		reader->SetFileName(inputFilename);
		reader->Update();
		
		ImageType::Pointer inputImage = reader->GetOutput();
        globalInputImage = inputImage;

        boost::filesystem::path filenameParts(inputFilename);
		std::string basename = filenameParts.stem().generic_string();

		boost::random::mt19937 rng(time(0));

		std::vector<std::string> op_names;
		op_names.push_back("Noise");
		op_names.push_back("Thin plate spline deformation");
		op_names.push_back("Rotate");
		op_names.push_back("Shear");

        double op_prob[] = {noise_prob, tps_prob, rot_prob, shear_prob};
        boost::random::discrete_distribution<unsigned int> operationDice(op_prob);
		enum operation {NOISE, TPS, ROTATE, SHEAR};

		std::string outputFilename = boost::filesystem::path(outputDirName + "/" + basename + "_" + std::to_string(0) + ".nrrd").generic_string();
		std::string log_filename = boost::filesystem::path(outputDirName + "/" + basename + ".log").generic_string();
		log_file = std::ofstream(log_filename, std::ios_base::out | std::ios_base::trunc);

        WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(outputFilename);
        writer->UseCompressionOn();
        writer->SetInput(inputImage);
		LOG_LINE("Will be writing " << augment_count << " replicas for this file");
        std::cout << "Writing unmodified image: " << outputFilename << std::endl;
        writer->Update();
        boost::thread_group workers;
        for (unsigned int i = 1; i < augment_count; i++) {

			int op = operationDice(rng);
			outputFilename = boost::filesystem::path(outputDirName + "/" + basename + "_" + std::to_string(i) + ".nrrd").generic_string();

			std::cout << "Performing opetration " << op_names[op] << " on image " << i << std::endl;

            if(file_exists(outputFilename)) {
                std::cout << std::endl <<  outputFilename << " already exists, exiting " <<  std::endl;
                continue;
            }
			std::vector<float> params;
			switch (op) {
				case NOISE:
					params.push_back(noise_std);
                    queue.push(ImageJob(deform_noise, outputFilename.c_str(), params));
				break;
				case TPS:
					params.push_back((float)tps_count);
					params.push_back(tps_factor);
					queue.push(ImageJob(deform_tps, outputFilename.c_str(), params));
					break;
                case ROTATE:
					params.push_back(rot_min_deg);
					params.push_back(rot_max_deg);
					queue.push(ImageJob(deform_rotate,outputFilename.c_str(), params));
                    break;
                case SHEAR:
					params.push_back(shear_min_fac);
					params.push_back(shear_max_fac);
					queue.push(ImageJob(deform_shear, outputFilename.c_str(), params));
                    break;
			}
		}

        for (int i = 0; i != number_of_threads; ++i) {
            workers.create_thread(process_entry);
        }

        workers.join_all();

	} catch (itk::ExceptionObject &ex) {
        std::cout << ex << std::endl;
        return EXIT_FAILURE;
    }
}