/*
 * Modification on an ITK example by Dženan Zukić <dzenan.zukic@kitware.com>
 * Modified to allow changing output file name without specifying series name
 */

#include <vector>
#include <thread>
#include <iostream>
#include <sstream>
#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkShrinkImageFilter.h"


#ifdef _WIN32
    #include <direct.h>
#elif defined __linux__
    #include <sys/stat.h>
#endif

typedef signed short    PixelType;
const unsigned int      Dimension = 3;
typedef itk::Image< PixelType, Dimension >         ImageType;
typedef itk::ImageSeriesReader< ImageType >        ReaderType;
typedef itk::ShrinkImageFilter< ImageType, ImageType> FilterType;

typedef itk::ImageSeriesReader< ImageType >        ReaderType;
typedef itk::GDCMSeriesFileNames NamesGeneratorType;

typedef std::vector<unsigned int> UIntVector;
typedef std::vector<std::thread> ThreadVector;


#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

bool file_exists(std::string filename) {
    return boost::filesystem::exists(filename) && boost::filesystem::file_size(filename) != 0;
}



void shrinkAndWriteImage(int factor, ReaderType::Pointer input, std::string outputFilename) {
	FilterType::Pointer filter = FilterType::New();
	filter->SetInput(input->GetOutput());
	filter->SetShrinkFactor(0, factor);
	filter->SetShrinkFactor(1, factor);
	filter->SetShrinkFactor(2, factor);

	typedef itk::ImageFileWriter< ImageType > WriterType;
	WriterType::Pointer writer = WriterType::New();

	writer->SetFileName(outputFilename);
	writer->UseCompressionOn();
	writer->SetInput(filter->GetOutput());
	std::cout << "Writing: " << outputFilename << std::endl;
	try {
		writer->Update();
	}
	catch (itk::ExceptionObject &ex) {
		std::cout << ex << std::endl;
	}
}

int main(int argc, char* argv[]) {

    try {

	    namespace po = boost::program_options;
		boost::program_options::options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("dicom,d", po::value<std::string>()->default_value(""), "DICOM directory.")
            ("output,o", po::value<std::string>()->default_value(""), "Output directory.")
            ("factors,f", po::value<UIntVector>()->multitoken(), "Shrink factors");

		boost::program_options::variables_map vm;
		boost::program_options::store(po::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);

		if (vm.count("help")) {
			std::cout << desc << '\n';
			return EXIT_SUCCESS;
		}

		if (!vm.count("dicom")) {
			std::cerr << "DICOM directory not specified" << std::endl;
			return EXIT_FAILURE;
		}

        if (!vm.count("output")) {
            std::cerr << "No output directory was specified" << std::endl;
            return EXIT_FAILURE;
        }

        if(!vm.count("factors")) {
            std::cerr << "At least one shrink factor has to be specified" << std::endl;
            return EXIT_FAILURE;
        }



        const std::string dicomDirName = vm["dicom"].as<std::string>();
        const std::string outputDirName = vm["output"].as<std::string>();
        const UIntVector shrinkFactors = vm["factors"].as<UIntVector>();


        NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

        nameGenerator->SetUseSeriesDetails(true);
        nameGenerator->AddSeriesRestriction("0008|0021");
        nameGenerator->SetGlobalWarningDisplay(false);
        nameGenerator->SetDirectory(dicomDirName);



        typedef std::vector< std::string >    SeriesIdContainer;
        const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
        SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
        SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();

        if (seriesItr != seriesEnd)  {
            std::cout << "The directory: ";
            std::cout << dicomDirName << std::endl;
            std::cout << "Contains the following DICOM Series: ";
            std::cout << std::endl;
        }  else  {
            std::cout << "No DICOMs in: " << dicomDirName << std::endl;
            return EXIT_SUCCESS;
        }

        while (seriesItr != seriesEnd) {
            std::cout << seriesItr->c_str() << std::endl;
            ++seriesItr;
        }

        seriesItr = seriesUID.begin();
        while (seriesItr != seriesUID.end()) {
            std::string seriesIdentifier;
            seriesIdentifier = seriesItr->c_str();
            seriesItr++;

            std::cout << "\nReading: ";
            std::cout << seriesIdentifier << std::endl;
            typedef std::vector< std::string >   FileNamesContainer;
            FileNamesContainer fileNames;
            fileNames = nameGenerator->GetFileNames(seriesIdentifier);

            ReaderType::Pointer reader = ReaderType::New();
            typedef itk::GDCMImageIO       ImageIOType;
            ImageIOType::Pointer dicomIO = ImageIOType::New();
            reader->SetImageIO(dicomIO);
            reader->SetFileNames(fileNames);
			reader->Update();

			ThreadVector threads;
			for (UIntVector::iterator it = shrinkFactors.begin(); it != shrinkFactors.end(); ++it) {
				std::stringstream ssd;
				ssd << outputDirName << "/" << *it << "/";

                boost::filesystem::path dir(ssd.str());
                boost::filesystem::create_directory(dir);

                boost::filesystem::path filenameParts(dicomDirName);
                std::string basename = filenameParts.stem().generic_string();

				std::stringstream ssf;
				ssf << outputDirName << "/" << *it << "/" << basename << ".nrrd";
				threads.push_back(std::thread(shrinkAndWriteImage, *it, reader, ssf.str()));
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

        }
    }
    catch (itk::ExceptionObject &ex)
    {
        std::cout << ex << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}