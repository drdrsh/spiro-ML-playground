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
#include <itkPointSet.h>
#include <itkThinPlateSplineKernelTransform.h>
#include <itkImageRandomConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>



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
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::ImageFileWriter< ImageType > WriterType;

int main(int argc, const char* argv[]) {

	const unsigned int gridSize = 2;
	const unsigned int imageSize = 100;

	try {

		ImageType::Pointer image = ImageType::New();
	
		ImageType::RegionType region;
		ImageType::IndexType start;
		start[0] = 0;
		start[1] = 0;
		start[2] = 0;

		ImageType::SizeType size;
		size[0] = imageSize;
		size[1] = imageSize;
		size[2] = imageSize;

		region.SetSize(size);
		region.SetIndex(start);

		image->SetRegions(region);
		image->Allocate();
		
		itk::ImageRegionIterator<ImageType> it(image, image->GetLargestPossibleRegion());
		for (it = it.Begin(); !it.IsAtEnd(); ++it) {
			ImageType::IndexType idx = it.GetIndex();
			if (idx[0] % gridSize == 0 && idx[1] % gridSize == 0 /*&& idx[2] % gridSize == 0*/) {
				it.Set(1);
			}
		}
   
		std::string outputFilename = "grid.nrrd";
		WriterType::Pointer writer = WriterType::New();
		writer->SetFileName(outputFilename);
		writer->UseCompressionOn();
		writer->SetInput(image);
		std::cout << "Writing: " << outputFilename << std::endl;
		writer->Update();
    } catch (std::exception ex) {
        std::cerr << ex.what() << '\n';
    }
}