#include<stdio.h>
#include<iostream>
#include<fstream>

#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

#include"bpvo/bitplanes_descriptor.h"

#include "bpvo/timer.h"
#include"bpvo/vo.h"
#include"bpvo/utils.h"
#include"utils/stereo_algorithm.h"
#include"bpvo/config_file.h"
#include"bpvo/trajectory.h"
#include"utils/viz.h"

void colorizeDisparity(const cv::Mat& src, cv::Mat& dst, double min_d, double num_d)
{
  THROW_ERROR_IF( src.type() != cv::DataType<float>::type, "disparity must be float" );

  double scale = 0.0;
  if(num_d > 0) {
    scale = 255.0 / num_d;
  } else {
    double max_val = 0;
    cv::minMaxLoc(src, nullptr, &max_val);
    scale = 255.0 / max_val;
  }

  src.convertTo(dst, CV_8U, scale);
  cv::applyColorMap(dst, dst, cv::COLORMAP_JET);

  for(int y = 0; y < src.rows; ++y)
    for(int x = 0; x < src.cols; ++x)
      if(src.at<float>(y,x) <= min_d)
        dst.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
}

void overlayDisparity(const cv::Mat& I, const cv::Mat& D, cv::Mat& dst,
                      double alpha, double min_d, double num_d)
{
  cv::Mat image;
  switch( I.type() )
  {
    case CV_8UC1:
      cv::cvtColor(I, image, CV_GRAY2BGR);
      break;
    case CV_8UC3:
      image = I;
      break;
    case CV_8UC4:
      cv::cvtColor(I, image, CV_BGRA2BGR);
      break;
    default:
      THROW_ERROR("unsupported image type");
  }

  cv::addWeighted(image, alpha, bpvo::colorizeDisparity(D, min_d, num_d), 1.0-alpha, 0.0, dst);
}

/*!
@brief Reads input data file into OpenCV matrix
@return Returns the Matrix
*/
cv::Mat readDataFile(std::ifstream& inputFile, char delimiter)
{
    cv::Mat dataMat;
    std::string buffer;
    while(std::getline(inputFile,buffer))
    {
        std::stringstream ss(buffer);
        std::string elem;
        cv::Mat rowMat;
        while(std::getline(ss,elem,delimiter))
        {
            rowMat.push_back(atof(&elem[0]));
        }

        if(dataMat.empty())
        {
            dataMat.push_back(rowMat);
            dataMat = dataMat.t();
        }
        else
        {
            cv::vconcat(dataMat,rowMat.t(),dataMat);
        }
    }
    return dataMat;
}

int main()
{
    std::string configFile = "/home/sourav/workspace/bpvo/bpvo_codes/bin/cfg/oxford_stereo.cfg";
    bpvo::ConfigFile config(configFile);
    bpvo::StereoAlgorithm strAlg(config);

    std::stringstream dataPath("/media/sourav/My Passport/current/data/oxford/2014-12-10-18-10-50/stereo/");
    std::string imageNamesFileName = dataPath.str();
    std::ifstream imageNamesFile(imageNamesFileName.append("left.txt").c_str());

    std::cout << imageNamesFileName << std::endl;

    std::vector<std::string> imgNames;
    std::string buffer;
    while(std::getline(imageNamesFile,buffer))
    {
        std::stringstream ss(buffer);
        std::string elem;
        while(std::getline(ss,elem))
        {
            imgNames.push_back(elem);
        }
//        std::cout << imgNames[imgNames.size()-1] << std::endl;
    }

    double total_time = 0.0f;

    bpvo::Trajectory trajectory;
    bpvo::Matrix33 K; K << 983.044006, 0.0, 643.646973, 0.0, 983.044006, 493.378998, 0.0, 0.0, 1.0;
//    bpvo::Matrix33 K; K << 491.522003, 0.0, 321.8234865, 0.0, 491.522003, 246.689499, 0.0, 0.0, 1.0;
//    bpvo::Matrix33 K; K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;

    float b = 0.1;

    bpvo::AlgorithmParameters params;
    params.numPyramidLevels = 4;
    params.maxIterations = 100;
    params.parameterTolerance = 1e-6;
    params.functionTolerance = 1e-6;
    params.verbosity = bpvo::VerbosityType::kSilent;
    params.minTranslationMagToKeyFrame = 0.1;
    params.minRotationMagToKeyFrame = 2.5;
    params.maxFractionOfGoodPointsToKeyFrame = 0.7;
    params.goodPointThreshold = 0.8;

    cv::Mat left, right, dmap;
//    std::string lImgName = "/media/sourav/My Passport/current/data/oxford/2014-12-10-18-10-50/stereo/left/1418132416737115.png";
//    std::string rImgName = "/media/sourav/My Passport/current/data/oxford/2014-12-10-18-10-50/stereo/right/1418132416737115.png";
    bpvo::VisualOdometry vo(K,b, bpvo::ImageSize(960, 1280), params);

    for(int i1=0; i1<6000; i1++)
    {
        char imgNumStr[100];
        std::sprintf(imgNumStr,"%07d.png",i1);
        std::string lImgName = dataPath.str().append("left_rect/").append(imgNumStr);
        std::string rImgName = dataPath.str().append("right_rect/").append(imgNumStr);

        // Tsukuba
        char lImgNameTs[1000];
        std::sprintf(lImgNameTs,"/home/sourav/workspace/NTSD-200/fluorescent/left/frame_%d.png",i1);
        char rImgNameTs[1000];
        std::sprintf(rImgNameTs,"/home/sourav/workspace/NTSD-200/fluorescent/right/frame_%d.png",i1);
        char dImgNameTs[1000];
        std::sprintf(dImgNameTs,"/home/sourav/workspace/NTSD-200/disparity_maps/left/frame_%d.png",i1);


        left = cv::imread(lImgName,cv::IMREAD_GRAYSCALE);
        right = cv::imread(rImgName,cv::IMREAD_GRAYSCALE);
//        dmap = cv::imread(dImgNameTs,cv::IMREAD_GRAYSCALE);
//        dmap.convertTo(dmap,CV_32FC1);

//        cv::resize(left,left,cv::Size(640,480),cv::INTER_AREA);
//        cv::resize(right,right,cv::Size(640,480),cv::INTER_AREA);

        strAlg.run(left,right,dmap);


//        dmap.convertTo(dmap,CV_8UC3,1,0);


        bpvo::Timer timer;
        auto result = vo.addFrame(left.ptr<uint8_t>(), dmap.ptr<float>());
        auto tt = timer.stop().count();
        total_time += ( tt / 1000.0f);

//        cv::applyColorMap(dmap, dmap, cv::COLORMAP_JET);
//        colorizeDisparity(dmap,dmap,1.0,128.0);
        overlayDisparity(left,dmap,dmap,0.5,1.0,128.0);
        cv::imshow("dmap",dmap);
//        cv::imshow("left",left);
        cv::waitKey(1);

        trajectory.push_back( result.pose );
        std::cout << result.pose << std::endl;

        fprintf(stdout, "Frame %03d [%03d ms] %0.2f Hz\r", i1, (int) tt, i1 / total_time);
        fflush(stdout);
    }
    trajectory.writeCameraPath("poses_xyz.txt");
    trajectory.write("poses_se3.txt");
}
