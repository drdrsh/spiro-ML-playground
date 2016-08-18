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
#define LUNG_THRESHOLD -750
#define LUNG_MEDIAN    -820
#define SLICE_DIRECTION 0
#define RIGHT_LUNG		0
#define LEFT_LUNG		1

struct Voxel{
    Voxel(InputImageType::IndexType i, InputImageType::PixelType p) {
        idx = i;
        val = p;
    }
    InputImageType::IndexType idx;
    InputImageType::PixelType val;
};

bool sortVoxels(const Voxel &lhs, const Voxel &rhs) { return lhs.val < rhs.val; }

std::vector<InputImageType::IndexType> get_seed(InputImageType::Pointer image, int lung) {

    std::vector<InputImageType::IndexType> indices;
    std::vector<std::vector<Voxel>> sequences;


	InputImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
	InputImageType::IndexType regionStart;
	InputImageType::SizeType  regionSize;
	for (unsigned int j = 0; j < IMAGE_DIMENSIONS; j++) {
		if (j == SLICE_DIRECTION) {
			if(lung == RIGHT_LUNG) {
				regionStart[j] = imageSize[j] / 2 - 1;
			}
			else {
				regionStart[j] = 0;
			}
			regionSize[j] = (int)(imageSize[j] / 2) - 1;
			continue;
		}
		regionStart[j] = (int)(imageSize[j] / 2) - 2;
		regionSize[j] = 5;
	}
	InputImageType::RegionType targetRegion(regionStart, regionSize);
	bool is_found = false;
	int number_of_lung_voxels = 0;

    LinearIteratorType it(image, targetRegion);
	it.SetDirection(SLICE_DIRECTION);

	it.GoToBegin();
	while (!it.IsAtEnd()) {
        std::vector<Voxel> voxels;
        for(;!it.IsAtEndOfLine(); ++it) {
			int value = it.Get();
			if (value < LUNG_THRESHOLD) {
				number_of_lung_voxels++;
                Voxel v = Voxel(it.GetIndex(), value);
                voxels.push_back(v);
                continue;
			}

            if(!voxels.empty()) {
                sequences.push_back(voxels);
            }
            voxels.clear();
		}
		it.NextLine();
	}

    std::vector<Voxel> longest_sequence;
    for (std::vector<std::vector<Voxel>>::iterator it = sequences.begin() ; it != sequences.end(); ++it) {
        std::vector<Voxel> x = *it;
        if(x.size() > longest_sequence.size()) {
            longest_sequence = x;
        }
    }

    for (std::vector<Voxel>::iterator it = longest_sequence.begin() ; it != longest_sequence.end(); ++it) {
        if(it == longest_sequence.begin() || it == longest_sequence.end()) {
            continue;
        }
        Voxel x = *it;
        indices.push_back(x.idx);
    }



    //std::sort(longest_sequence.begin(), longest_sequence.end(), sortVoxels);
    int size = longest_sequence.size();
    if(size % 2 == 0) {
        //indices.push_back(longest_sequence[size/2].idx);
    } else {
        //indices.push_back(longest_sequence[(int)(size/2+1)].idx);
    }

    return indices;
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
    thresholdFilter->SetLowerThreshold(-1500);
    thresholdFilter->SetUpperThreshold(-400);
    thresholdFilter->SetNumberOfThreads(8);


    ConnectedFilterType::Pointer neighborhoodConnected = ConnectedFilterType::New();
    neighborhoodConnected->SetInput( thresholdFilter->GetOutput() );
    neighborhoodConnected->SetLower(1);
    neighborhoodConnected->SetUpper(1);
    neighborhoodConnected->SetReplaceValue(1);
    neighborhoodConnected->SetNumberOfThreads(8);

    InputImageType::SizeType radius;
    radius[0] = 1;
    radius[1] = 1;
    radius[2] = 1;
    neighborhoodConnected->SetRadius(radius);

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
    std::vector<InputImageType::IndexType> seeds;
    std::vector<InputImageType::IndexType> s;

    s = get_seed(inputImage, LEFT_LUNG);
    seeds.insert(seeds.end(), s.begin(), s.end());

    s = get_seed(inputImage, RIGHT_LUNG);
    seeds.insert(seeds.end(), s.begin(), s.end());

    for(std::vector<InputImageType::IndexType>::iterator it = seeds.begin() ; it != seeds.end(); ++it) {
        neighborhoodConnected->AddSeed(*it);
    }

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
