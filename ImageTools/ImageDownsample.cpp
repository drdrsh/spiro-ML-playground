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

#include "OptionParser.h"

typedef signed short    PixelType;
const unsigned int      Dimension = 3;

typedef itk::Image< PixelType, Dimension > ImageType;
typedef itk::ImageSeriesReader< ImageType > ReaderType;
typedef itk::GDCMSeriesFileNames NamesGeneratorType;
typedef itk::IdentityTransform<double, Dimension> TransformType;
typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

typedef std::vector<int> IntVector;
typedef std::vector<std::thread> ThreadVector;

enum  optionIndex {
    UNKNOWN,
    HELP,
    DICOM_DIRECTORY,
    OUTPUT_DIRECTORY,
    OUTPUT_IMAGE_SIZE_X, OUTPUT_IMAGE_SIZE_Y, OUTPUT_IMAGE_SIZE_Z,
    OUTPUT_IMAGE_SPC_X, OUTPUT_IMAGE_SPC_Y, OUTPUT_IMAGE_SPC_Z
};
const option::Descriptor usage[] =  {
        {UNKNOWN,                0,"" , ""    ,option::Arg::None, "USAGE: ImageDownsample [options]\n\nOptions:" },
        {HELP,                   0,"h" , "help",     option::Arg::None, "  --help -h  \tPrint usage and exit." },
        {DICOM_DIRECTORY,        0,"i" , "dicom",    option::Arg::None, "  --dicom -i  \tInput DICOM series directory." },
        {OUTPUT_DIRECTORY,       0,"o" , "output",   option::Arg::None, "  --output -o \tOutput directory." },
        {OUTPUT_IMAGE_SIZE_X,    0,"x", "size-x",    option::Arg::Optional, "  --size-x, -x  \t(Optional) Output Image X Size." },
        {OUTPUT_IMAGE_SIZE_Y,    0,"y", "size-y",    option::Arg::Optional, "  --size-y, -y  \t(Optional) Output Image Y Size." },
        {OUTPUT_IMAGE_SIZE_Z,    0,"z", "size-z",    option::Arg::Optional, "  --size-z, -z  \t(Optional) Output Image Z Size." },
        {OUTPUT_IMAGE_SPC_X,     0,"a", "spacing-x", option::Arg::Optional, "  --spacing-x, -a  \t(Optional) Output Image X Spacing." },
        {OUTPUT_IMAGE_SPC_Y,     0,"b", "spacing-y", option::Arg::Optional, "  --spacing-y, -b  \t(Optional) Output Image Y Spacing." },
        {OUTPUT_IMAGE_SPC_Z,     0,"c", "spacing-z", option::Arg::Optional, "  --spacing-z, -c  \t(Optional) Output Image Z Spacing." },
        {0,0,0,0,0,0}
};


int main(int argc, char* argv[]) {

    argc -= (argc > 0);
    argv += (argc > 0); // skip program name argv[0] if present
    option::Stats stats(usage, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(usage, argc, argv, options, buffer);

    if (parse.error()) {
        return 1;
    }

    parse.parse(usage, argc, argv, options, buffer);

    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, usage);
        return 0;
    }

    option::Option *opt = options[OUTPUT_IMAGE_SIZE_X];
    std::cout << "Unknown option: " << parse.nonOption(opt->index()) << "\n";

    /*
    for (int i = 0; i < parse.nonOptionsCount(); ++i) {
        std::cout << "Non-option #" << i << ": " << parse.nonOption(i) << "\n";
    }

    for (int i = 0; i < parse.optionsCount(); ++i) {
        std::cout << "option #" << i << ": " << parse.nonOption(i) << "\n";
    }
    */
}