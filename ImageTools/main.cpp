/*
 * Modification on an ITK example by Dženan Zukić <dzenan.zukic@kitware.com>
 * Modified to allow changing output file name without specifying series name
 */

#include <vector>
#include <iostream>
#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " [DicomDirectory outputFileName]";
        std::cerr << "\nIf DicomDirectory is not specified, current directory is used\n";
    }

    typedef signed short    PixelType;
    const unsigned int      Dimension = 3;
    typedef itk::Image< PixelType, Dimension >         ImageType;
    typedef itk::ImageSeriesReader< ImageType >        ReaderType;
    std::string dirName = "."; //current directory by default
    //dirName = "/home/mostafa/SummerProject/Data/CU100010";
    if (argc > 1)
    {
        dirName = argv[1];
    }

    typedef itk::ImageSeriesReader< ImageType >        ReaderType;
    typedef itk::GDCMSeriesFileNames NamesGeneratorType;
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0008|0021");
    nameGenerator->SetGlobalWarningDisplay(false);
    nameGenerator->SetDirectory(dirName);

    try
    {
        typedef std::vector< std::string >    SeriesIdContainer;
        const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
        SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
        SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();

        if (seriesItr != seriesEnd)
        {
            std::cout << "The directory: ";
            std::cout << dirName << std::endl;
            std::cout << "Contains the following DICOM Series: ";
            std::cout << std::endl;
        }
        else
        {
            std::cout << "No DICOMs in: " << dirName << std::endl;
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

            typedef itk::ImageFileWriter< ImageType > WriterType;
            WriterType::Pointer writer = WriterType::New();
            std::string outFileName;
            if (argc > 2)
            {
                outFileName = argv[2];
            }
            else
            {
                outFileName = dirName + std::string("/") + seriesIdentifier + ".nrrd";
            }
            writer->SetFileName(outFileName);
            writer->UseCompressionOn();
            writer->SetInput(reader->GetOutput());
            std::cout << "Writing: " << outFileName << std::endl;
            try
            {
                writer->Update();
            }
            catch (itk::ExceptionObject &ex)
            {
                std::cout << ex << std::endl;
                continue;
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