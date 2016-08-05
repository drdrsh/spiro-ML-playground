/*
 * Modification on an ITK example
 * https://itk.org/Wiki/ITK/Examples/ImageProcessing/ResampleImageFilter
 */

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

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#ifdef _WIN32
    #include <direct.h>
#elif defined __linux__
    #include <sys/stat.h>
#endif

typedef signed short    PixelType;
const unsigned int      Dimension = 3;

typedef itk::Image< PixelType, Dimension > ImageType;
typedef itk::ImageSeriesReader< ImageType > ReaderType;
typedef itk::GDCMSeriesFileNames NamesGeneratorType;
typedef itk::IdentityTransform<double, Dimension> TransformType;
typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

typedef std::vector<int> IntVector;
typedef std::vector<std::thread> ThreadVector;

int main(int argc, const char* argv[]) {
    try {
        namespace po = boost::program_options;
        boost::program_options::options_description desc{"Options"};
        desc.add_options()
                ("help,h", "Help screen")
                ("dicom,i", po::value<std::string>()->default_value(""), "Input DICOM series directory.")
                ("output,o", po::value<std::string>()->default_value(""), "Output directory.")
                ("spacing-x,a", po::value<float>()->default_value(0.0f), "(Optional) Output image X spacing.")
                ("spacing-y,b", po::value<float>()->default_value(0.0f), "(Optional) Output image Y spacing.")
                ("spacing-z,c", po::value<float>()->default_value(0.0f), "(Optional) Output image Z spacing.")
                ("size-x,x", po::value<unsigned int>()->default_value(0), "(Optional) Output image X size.")
                ("size-y,y", po::value<unsigned int>()->default_value(0), "(Optional) Output image Y size.")
                ("size-z,z", po::value<unsigned int>()->default_value(0), "(Optional) Output image Z size.");

        boost::program_options::variables_map vm;
        boost::program_options::store(po::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << '\n';
        } else if (vm.count("size-x")) {
            std::cout << "Size-X: " << vm["size-x"].as<unsigned int>() << '\n';
        }

    } catch (std::exception ex) {
        std::cerr << ex.what() << '\n';
    }
}