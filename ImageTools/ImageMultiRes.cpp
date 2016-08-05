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

typedef std::vector<int> IntVector;
typedef std::vector<std::thread> ThreadVector;

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

int main(int argc, char* argv[])
{
	IntVector shrinkFactors;
	
	shrinkFactors.push_back(2);
	shrinkFactors.push_back(3);
	shrinkFactors.push_back(5);
	shrinkFactors.push_back(7);

	if (argc < 4)
    {
        std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << " DicomDirectory OutputDirectory Basename" << std::endl;
		return EXIT_FAILURE;
    }

    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

	std::string dicomDirName = argv[1];
	std::string outputDirName= argv[2];
	std::string basename = argv[3];

	nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0008|0021");
    nameGenerator->SetGlobalWarningDisplay(false);
    nameGenerator->SetDirectory(dicomDirName);


    try
    {
        typedef std::vector< std::string >    SeriesIdContainer;
        const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
        SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
        SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();

        if (seriesItr != seriesEnd)
        {
            std::cout << "The directory: ";
            std::cout << dicomDirName << std::endl;
            std::cout << "Contains the following DICOM Series: ";
            std::cout << std::endl;
        }
        else
        {
            std::cout << "No DICOMs in: " << dicomDirName << std::endl;
            return EXIT_SUCCESS;
        }

        while (seriesItr != seriesEnd)
        {
            std::cout << seriesItr->c_str() << std::endl;
            ++seriesItr;
        }

        seriesItr = seriesUID.begin();
        while (seriesItr != seriesUID.end())
        {
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
			for (IntVector::iterator it = shrinkFactors.begin(); it != shrinkFactors.end(); ++it) {
				std::stringstream ssd;
				ssd << outputDirName << "/" << *it << "/";
#ifdef _WIN32
				mkdir(ssd.str().c_str());
#elif defined __linux__
                mkdir(ssd.str().c_str(), 0755 );
#endif
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