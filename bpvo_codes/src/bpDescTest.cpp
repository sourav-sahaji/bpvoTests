#include<stdio.h>
#include<iostream>
#include<fstream>

#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

#include"bpvo/bitplanes_descriptor.h"

#include "bpvo/timer.h"

//using namespace std;
//using namespace cv;
//using namespace bpvo;


int main()
{
            std::cout << "This code will calculate BP descriptor for a dataset, and then probably compare it with another dataset" << std::endl;

//            // Testing code on an random image first
//            cv::Mat img = cv::imread("randomTestImage.png");
//            cv::resize(img,img,cv::Size(64,32),cv::INTER_AREA);
//            cv::imshow("img",img);
//            cv::waitKey(1);

//            bpvo::Timer timer;

//            bpvo::BitPlanesDescriptor bpDesc;
//            bpDesc.compute(img);
//            std::vector<cv::Mat> bp;
//            for(int ch=0; ch<bpDesc.numChannels(); ch++)
//                bp.push_back(bpDesc.getChannel(ch));
//            std::cout << bp[1].size() << std::endl;

//            double tt = timer.stop().count();

//            std::cout << tt << std::endl;

            // Read a dataset
            cv::VideoCapture cap1("/media/sourav/Default/Users/n9349995/Desktop/dataset/slamData/alderley/night.mpeg");
            std::ofstream descsFile("BPdescsFile.txt");

            int numFrames = cap1.get(CV_CAP_PROP_FRAME_COUNT);
            std::cout << "Num Images = " << numFrames << std::endl;

            std::vector<cv::Mat> outDescs;
            for(int i1=0; i1<numFrames; i1++)
            {
                cv::Mat img1;
                cap1 >> img1;

                if(img1.empty())
                {
                    std::cerr << "Image empty, exiting..." << i1 << std::endl;
                    break;
                }

                cv::cvtColor(img1,img1,cv::COLOR_BGR2GRAY);
                cv::resize(img1,img1,cv::Size(64,32),cv::INTER_AREA);

                bpvo::BitPlanesDescriptor bpDesc;

                bpDesc.compute(img1);

                std::vector<cv::Mat> bp;
                for(int ch=0; ch<bpDesc.numChannels(); ch++)
                {
                    bp.push_back(bpDesc.getChannel(ch));
                }

                cv::Mat outDesc=cv::Mat::zeros(img1.rows,img1.cols,CV_8UC1);
                for(int r1=0; r1<outDesc.rows; r1++)
                    for(int c1=0; c1<outDesc.cols; c1++)
                        for(int j1=0; j1<8; j1++)
                        {
                            if((int)bp[j1].at<float>(r1,c1)==1)
                                outDesc.at<uchar>(r1,c1) |= 1 << j1;
                        }

                outDescs.push_back(outDesc);

                cv::imshow("img1",img1);
                cv::imshow("img2",outDesc);

//                cv::imshow("bp0",bp[0]);
//                cv::imshow("bp1",bp[1]);
//                cv::imshow("bp2",bp[2]);
//                cv::imshow("bp3",bp[3]);
//                cv::imshow("bp4",bp[4]);
//                cv::imshow("bp5",bp[5]);
//                cv::imshow("bp6",bp[6]);
//                cv::imshow("bp7",bp[7]);

                cv::waitKey(1);

                int skipCounter = 0;
                while(skipCounter++<10)
                    cap1 >> img1;
            }

            for(int i1=0; i1<outDescs.size(); i1++)
            {
                for(int j1=0; j1<outDescs[i1].rows; j1++)
                    for(int k1=0; k1<outDescs[i1].cols; k1++)
                    {
                        descsFile << (int)outDescs[i1].at<uchar>(j1,k1) << " ";
                    }
                    descsFile << "\n";
                }
            descsFile.close();
}
