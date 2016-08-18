#include <itkMacro.h>
#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImage.h>
#include <itkMetaDataDictionary.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkNeighborhoodConnectedImageFilter.h>
#include <itkMaskNegatedImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkVotingBinaryIterativeHoleFillingImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkImageLinearIteratorWithIndex.h>

#define IMAGE_DIMENSIONS   3
#define INPUT_IMAGE_DATA   int

typedef itk::Image< INPUT_IMAGE_DATA, IMAGE_DIMENSIONS > InputImageType;

typedef itk::ImageFileReader< InputImageType > ReaderType;
typedef itk::ImageFileWriter< InputImageType > WriterType;

typedef itk::BinaryThresholdImageFilter < InputImageType, InputImageType >  ThresholdFilterType;
typedef itk::NeighborhoodConnectedImageFilter < InputImageType, InputImageType > ConnectedFilterType;
typedef itk::BinaryBallStructuringElement < INPUT_IMAGE_DATA, IMAGE_DIMENSIONS > StructuringElementType;
typedef itk::BinaryErodeImageFilter < InputImageType, InputImageType, StructuringElementType> ErodeFilterType;
typedef itk::BinaryDilateImageFilter < InputImageType, InputImageType, StructuringElementType> DilateFilterType;
typedef itk::MaskImageFilter< InputImageType, InputImageType, InputImageType> MaskFilterType;
typedef itk::ImageLinearConstIteratorWithIndex< InputImageType > LinearIteratorType;

//typedef itk::Statistics::ScalarImageTextureCalculator<ImageTypeInt> TextureCalculator;
#define LUNG_THRESHOLD -700
#define COUNT_THRESHOLD 0.1
#define SLICE_DIRECTION 0
#define RIGHT_LUNG		0
#define LEFT_LUNG		1
InputImageType::IndexType get_seed(InputImageType::Pointer image, int lung) {

	InputImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
	InputImageType::IndexType regionStart;
	InputImageType::SizeType  regionSize;
	for (int i = 0; i < IMAGE_DIMENSIONS; i++) {
		if (i == SLICE_DIRECTION) {
			if(lung == RIGHT_LUNG) {
				regionStart[i] = imageSize[i] / 2 - 1;
			}
			else {
				regionStart[i] = 0;
			}
			regionSize[i] = (int)(imageSize[i] / 2) - 1;
			continue;
		}
		regionStart[i] = (int)(imageSize[i] / 2);
		regionSize[i] = 1;
	}
	InputImageType::RegionType targetRegion(regionStart, regionSize);
	bool is_found = false;
	int number_of_lung_voxels = 0;
	int target_lung_voxels = imageSize[SLICE_DIRECTION] * COUNT_THRESHOLD;
	LinearIteratorType it(image, targetRegion);
	it.SetDirection(SLICE_DIRECTION);

	it.GoToBegin();
	while (!it.IsAtEnd()) {
		while (!it.IsAtEndOfLine()) {
			int value = it.Get();
			if (value < LUNG_THRESHOLD) {
				number_of_lung_voxels++;
				if (number_of_lung_voxels >= target_lung_voxels) {
					//it.Set(2500);
					return it.GetIndex();
				}
			}
			++it;
		}
		it.NextLine();
	}
	return InputImageType::IndexType();
}

int main (int argc, char *argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " InputImage OutputImage";
    }

    std::string inputImageName = argv[1];
    std::string outputImageName = argv[2];

    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(inputImageName);
    InputImageType::Pointer inputImage = reader->GetOutput();
    reader->Update();

    InputImageType::SizeType  imageSize = inputImage->GetLargestPossibleRegion().GetSize();


    ThresholdFilterType::Pointer thresholdFilter = ThresholdFilterType::New();

    thresholdFilter->SetInput(inputImage);
    thresholdFilter->SetInsideValue(1);
    thresholdFilter->SetOutsideValue(0);
    thresholdFilter->SetLowerThreshold(-1000.0);
    thresholdFilter->SetUpperThreshold(-400.0);
    thresholdFilter->SetNumberOfThreads(8);


    ConnectedFilterType::Pointer neighborhoodConnected = ConnectedFilterType::New();
    neighborhoodConnected->SetInput( thresholdFilter->GetOutput() );
    neighborhoodConnected->SetLower(1);
    neighborhoodConnected->SetUpper(1);
    neighborhoodConnected->SetReplaceValue(1);//TODO: magic number
    neighborhoodConnected->SetNumberOfThreads(8);

    InputImageType::SizeType radius;
    radius[0] = 1;
    radius[1] = 1;
    radius[2] = 1;
    neighborhoodConnected->SetRadius(radius);

	InputImageType::IndexType leftSeed = get_seed(inputImage, LEFT_LUNG);
    InputImageType::IndexType rightSeed = get_seed(inputImage, RIGHT_LUNG);

	/*
    int leftX  = (imageSize[0] / 4);
    int rightX = (imageSize[0] - imageSize[0] / 4);
    int halfY  = (imageSize[1] / 2);
    int halfZ  = (imageSize[2] / 2);

    leftSeed[0]  = leftX;
    leftSeed[1]  = halfY;
    leftSeed[2]  = halfZ;

    rightSeed[0] = rightX;
    rightSeed[1] = halfY;
    rightSeed[2] = halfZ;
	*/

    neighborhoodConnected->AddSeed(leftSeed);
    neighborhoodConnected->AddSeed(rightSeed);
    /*
    HoleFillingType::Pointer holeFillingFilter = HoleFillingType::New();
    radius[0] = 2;
    radius[1] = 2;
    radius[2] = 2;
    holeFillingFilter->SetRadius(radius);
    holeFillingFilter->SetInput(neighborhoodConnected->GetOutput());
    holeFillingFilter->SetBackgroundValue( 1 );
    holeFillingFilter->SetForegroundValue( 0 );
    holeFillingFilter->SetMajorityThreshold( 2 );
    holeFillingFilter->SetMaximumNumberOfIterations( 40 );
    */

    DilateFilterType::Pointer binaryDilateFilter = DilateFilterType::New();
    ErodeFilterType::Pointer binaryErodeFilter = ErodeFilterType::New();

    StructuringElementType structuringElement;
    structuringElement.SetRadius( 4 );
    structuringElement.CreateStructuringElement();

    binaryDilateFilter->SetKernel( structuringElement );
    binaryErodeFilter->SetKernel( structuringElement );

    binaryDilateFilter->SetBackgroundValue( 0 );
    binaryDilateFilter->SetForegroundValue( 1 );
    binaryErodeFilter->SetBackgroundValue( 0 );
    binaryErodeFilter->SetForegroundValue( 1 );

    binaryDilateFilter->SetInput( neighborhoodConnected->GetOutput() );
    binaryErodeFilter->SetInput( binaryDilateFilter->GetOutput() );

    binaryDilateFilter->SetNumberOfThreads(8);
    binaryErodeFilter->SetNumberOfThreads(8);


    MaskFilterType::Pointer maskImageFilter = MaskFilterType::New();
    maskImageFilter->SetInput(inputImage);
    maskImageFilter->SetMaskImage(binaryErodeFilter->GetOutput());
    maskImageFilter->SetNumberOfThreads(8);
    

    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( outputImageName );
    writer->SetInput(maskImageFilter->GetOutput());
    writer->Update();

    return 0;
}
