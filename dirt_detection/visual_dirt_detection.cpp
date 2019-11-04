#include <iostream>
#include <mutex>

#include <vector>
#include <memory>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ml/ml.hpp>

#include "visual_dirt_detection.h"

//#define PRINT_TIMING

using namespace std;
/*
#if CV_MAJOR_VERSION == 3
namespace EM = cv::ml;
#else
namespace EM = cv;
#endif

*/
namespace v4r{

void Visual_Dirt_Detection::compute_dirt(){

#ifdef PRINT_TIMING
    //Start timing:
    auto start = std::chrono::system_clock::now();
#endif
    //Transforming the image to float and then to the Lab space is a good (but costly) idea
    cv::Mat imageF;
    image.convertTo(imageF,CV_8UC4); //added for CLAHE
    //image.convertTo(imageF,CV_8UC3,1.0f/255.0f);
    cv::cvtColor(imageF,Lab,cv::COLOR_BGR2Lab);

    //CLAHE Algorithm implementation might improve the results of dirt detection besides slightly more calculation time is needed
    // Extract the L channel
    if(0)  //set for CLAHE implementation
    {
        std::vector<cv::Mat> lab_planes(3);
        cv::split(Lab, lab_planes);  // now we have the L image in lab_planes[0]

        // apply the CLAHE algorithm to the L channel
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        cv::Mat dst;
        clahe->apply(lab_planes[0], dst);

        // Merge the the color planes back into an Lab image
        dst.copyTo(lab_planes[0]);
        cv::merge(lab_planes, Lab);

       // convert back to RGBlab_planes[0]
       cv::Mat image_clahe;
       cv::cvtColor(Lab, image_clahe, cv::COLOR_Lab2BGR);

       cv::Mat diff;
       cv::subtract(image_clahe,imageF,diff);

       image.convertTo(image_clahe,CV_32FC3,1.0f/255.0f);
       cv::cvtColor(image_clahe,Lab,cv::COLOR_BGR2Lab);

       //Display the steps of the CLAHE Algorithm
       /*cv::namedWindow( "Display window1", 0 );
       cv::imshow("Display window1", imageF); cv::waitKey(3);
       cv::namedWindow( "Display window2", 0 );
       cv::imshow("Display window2", image_clahe); cv::waitKey(3);*/
       //cv::namedWindow( "Display window3", 0 );
       //cv::imshow("Display window3", diff); cv::waitKey(3);
       //cv::namedWindow( "Display window4", 0 );
       //cv::imshow("Display window4", lab_planes[2]); cv::waitKey(3);
    }

    //blur the image
    cv::Mat blurred;
    int size=(int)sigma*4-1;//some arbitrary rule to get a good kernel size for blurring
    cv::GaussianBlur(Lab,blurred,cv::Size(size,size),sigma);

    //calculate 2 norm of the gradient values
    cv::Mat gradx;
    cv::Mat grady;
    cv::Sobel(blurred,gradx,CV_32F,1,0); //maybe this allows to do this without splitting
    cv::Sobel(blurred,grady,CV_32F,0,1);

    cv::Mat gradxx,gradyy;
    cv::pow(gradx,2,gradxx);
    cv::pow(grady,2,gradyy);
	cv::imshow("gradxx",gradxx);
    cv::Mat magnitude;
    cv::sqrt(gradxx+gradyy,magnitude);

    //cv::namedWindow( "Display window1", 0 );
    //cv::imshow("Display window1", magnitude); cv::waitKey(3);


    //This has the assumption that the step size is half of the window size so the code might fail at this point if settings are otherwise
    int height=(image.rows-window_size_x + window_step_size_x)/window_step_size_y ;
    int width=(image.cols-window_size_y + window_step_size_y)/window_step_size_x ;
    int threshold=((float)(window_size_x*window_size_y))*inlier_ratio;



    //vector of valid patches with average and standard deviation of every channel (L, a, b)
    cv::Mat patchesL(width*height,2,CV_32FC1);
    cv::Mat patchesA(width*height,2,CV_32FC1);
    cv::Mat patchesB(width*height,2,CV_32FC1);
	cv::Mat patchesLAll(width*height,2,CV_32FC1);
	cv::Mat patchesAAll(width*height,2,CV_32FC1);
	cv::Mat patchesBAll(width*height,2,CV_32FC1);
    int validPatchCount=0;

    cv::Mat validPatch(height,width,CV_8UC1);
    validPatch = cv::Scalar(0);

    int allCount = 0;
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            //test if patch is not fully out of plane
            int validCount=0;
            for(int m=0;m<window_size_y;m++){
                for(int n=0;n<window_size_x;n++){
                   int _i=i*window_step_size_y+m;
                   int _j=j*window_step_size_x+n;
                   if(mask.at<unsigned char>(_i,_j)>0){
                       validCount++;
                   }

                }
            }


            //extract patch and anc calculate standard deviation and mean
            if(validCount>threshold){
                validPatch.at<unsigned char>(i,j)=255;
                cv::Rect region_of_interest(j*window_step_size_x,i*window_step_size_y,window_size_x,window_size_y);
                cv::Mat ROI(magnitude,region_of_interest);
                cv::Scalar mean;
                cv::Scalar stddv;
                cv::meanStdDev(ROI,mean,stddv);
                patchesL.at<float>(validPatchCount,0)=mean[0];
                patchesL.at<float>(validPatchCount,1)=stddv[0];
                patchesA.at<float>(validPatchCount,0)=mean[1];
                patchesA.at<float>(validPatchCount,1)=stddv[1];
                patchesB.at<float>(validPatchCount,0)=mean[2];
                patchesB.at<float>(validPatchCount,1)=stddv[2];
                validPatchCount++;

            }
			if(true){
				validPatch.at<unsigned char>(i,j)=255;
				cv::Rect region_of_interest(j*window_step_size_x,i*window_step_size_y,window_size_x,window_size_y);
				cv::Mat ROI(magnitude,region_of_interest);
				cv::Scalar mean;
				cv::Scalar stddv;
				cv::meanStdDev(ROI,mean,stddv);
				patchesLAll.at<float>(allCount,0)=mean[0];
				patchesLAll.at<float>(allCount,1)=stddv[0];
				patchesAAll.at<float>(allCount,0)=mean[1];
				patchesAAll.at<float>(allCount,1)=stddv[1];
				patchesBAll.at<float>(allCount,0)=mean[2];
				patchesBAll.at<float>(allCount,1)=stddv[2];
				allCount++;

			}
        }
    }

    //setup a dirtmap
    dirt = cv::Mat(image.rows,image.cols,CV_32FC1);
    dirt = cv::Scalar(0.0f);
    if(validPatchCount == 0){
        return;
    }
    //resize the patches or otherwise unset parameters would be used to train the GMM
    patchesL.rows=validPatchCount;
    patchesA.rows=validPatchCount;
    patchesB.rows=validPatchCount;

#ifdef PRINT_TIMING
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "collecting data dirt took " << elapsed.count()<< " microseconds" << std::endl;
    auto start2=end;
#endif


    //learn GMM
#if CV_MAJOR_VERSION == 2
    cv::EM gmmL(k,cv::EM::COV_MAT_DIAGONAL);//EM::COV_MAT_GENERIC ... EM::COV_MAT_DIAGONAL is standard but not fitting
    cv::EM gmmA(k,cv::EM::COV_MAT_DIAGONAL);
    cv::EM gmmB(k,cv::EM::COV_MAT_DIAGONAL);
    gmmL.train(patchesL);
    gmmA.train(patchesA);
    gmmB.train(patchesB);
    if(!gmmL.isTrained() || !gmmA.isTrained() || !gmmB.isTrained()){
        std::cout << "Could not train the GMMs" << std::endl;
        return;
    }
#else
    //cv::ml::EM::Params params(k,cv::ml::EM::COV_MAT_GENERIC);
    cv::Ptr<cv::ml::EM> gmmL = cv::ml::EM::create();
    gmmL->setClustersNumber(k);
    gmmL->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);//COV_MAT_GENERIC
    cv::Ptr<cv::ml::EM> gmmA = cv::ml::EM::create();
    gmmA->setClustersNumber(k);
    gmmA->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
    cv::Ptr<cv::ml::EM> gmmB = cv::ml::EM::create();
    gmmB->setClustersNumber(k);
    gmmB->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);


    gmmL->trainEM(patchesL);
    gmmA->trainEM(patchesA);
    gmmB->trainEM(patchesB);

    if(!gmmL->isTrained() || !gmmA->isTrained() || !gmmB->isTrained()){
        std::cout << "Could not train the GMMs" << std::endl;
        return;
    }
    cv::Mat probs_throwaway;
#endif



#ifdef PRINT_TIMING
    end = std::chrono::system_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start2);
    std::cout << "training the GMM took " << elapsed.count()<< " microseconds" << std::endl;
    auto start3=end;
#endif

    //apply the trained GMM to the  already captured elements
    cv::Mat patch(1,2,CV_32FC1);
    prob = cv::Mat(image.rows,image.cols,CV_32FC1);
    prob = 1000;
    cv::Mat experiment(height,width,CV_32FC1);
    validPatchCount=0;
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            if(validPatch.at<unsigned char>(i,j) || true){
                cv::Vec2d result;
                patch.at<float>(0)=patchesLAll.at<float>(validPatchCount,0);
                patch.at<float>(1)=patchesLAll.at<float>(validPatchCount,1);
                //result = gmmL->predict(patch);
#if CV_MAJOR_VERSION == 2
                result = gmmL.predict(patch);
#else
                result = gmmL->predict2(patch,probs_throwaway);
#endif
                float logLikelihoodL = -result[0];

                patch.at<float>(0)=patchesAAll.at<float>(validPatchCount,0);
                patch.at<float>(1)=patchesAAll.at<float>(validPatchCount,1);
                //result = gmmA->predict(patch);
#if CV_MAJOR_VERSION == 2
                result = gmmA.predict(patch);
#else
                result = gmmA->predict2(patch,probs_throwaway);
#endif
                float logLikelihoodA = -result[0];

                patch.at<float>(0)=patchesBAll.at<float>(validPatchCount,0);
                patch.at<float>(1)=patchesBAll.at<float>(validPatchCount,1);
#if CV_MAJOR_VERSION == 2
                result = gmmB.predict(patch);
#else
                result = gmmB->predict2(patch,probs_throwaway);
#endif
                float logLikelihoodB = -result[0];

                //get the combined likelihood:
                float logLikelihood = logLikelihoodL*weight_L+
                                      logLikelihoodA*weight_a+
                                      logLikelihoodB*weight_b;
				experiment.at<float>(i,j) = exp(-logLikelihood);
                for(int m=0;m<window_step_size_y;m++){
                    for(int n=0;n<window_step_size_x;n++){
                        int _i=i*window_step_size_y + (window_size_y - window_step_size_y)/2 +m;
                        int _j=j*window_step_size_x + (window_size_x - window_step_size_x)/2 + n;
                        prob.at<float>(_i,_j)=logLikelihood;
                        if(logLikelihood>prob_threshold){
                            dirt.at<float>(_i,_j)=1.0f;
                        }
                    }
                }

                validPatchCount++;

            }
        }
    }

#ifdef PRINT_TIMING
    end = std::chrono::system_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start3);
    std::cout << "marking the data took " << elapsed.count()<< " microseconds" << std::endl;

    end = std::chrono::system_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "estimating dirt took " << elapsed.count()<< " microseconds" << std::endl;
#endif
}

cv::Mat Visual_Dirt_Detection::compute_dirt(cv::Mat mask, cv::Mat image)
{
    this->mask=mask;
    this->image=image;
    compute_dirt();
    return dirt;
}


}
