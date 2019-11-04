#include <iostream>
#include <mutex>

#include <vector>
#include <memory>

#include "plane_segmentation.h"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>



using namespace Eigen;
using namespace std;


void v4r::Plane_Segmentation::setIntrinsics(float fx,float fy,float cx,float cy)
{
    this->fx=fx;
    this->fy=fy;
    this->_fx=1.0f/fx;
    this->_fy=1.0f/fy;
    this->cx=cx;
    this->cy=cy;

}


void v4r::Plane_Segmentation::setInput(cv::Mat depth){
    this->depth=depth;
    this->lastInput=2;
}
void v4r::Plane_Segmentation::setInput(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sortedCloud){
    this->_sortedCloud=sortedCloud;
    this->lastInput=1;
}

void v4r::Plane_Segmentation::setInput(pcl::PointCloud<pcl::PointXYZRGB>::Ptr sortedCloud)
{
    cols=sortedCloud->width;
    rows=sortedCloud->height;
    if(points.cols != cols+2 || points.rows != rows+2){
        points = cv::Mat(rows+2,cols+2,CV_32FC4);
        points.setTo(cv::Scalar(NAN,NAN,NAN,0));
    }
    for(unsigned int i=0;i<sortedCloud->height;i++){
        for(unsigned int j=0;j<sortedCloud->width;j++){
            pcl::PointXYZRGB p;
            p=sortedCloud->at(j,i);
            points.at<Eigen::Vector4f>(i+1,j+1)=Eigen::Vector4f(p.x,p.y,p.z,0);

        }

    }

    this->lastInput=3;
    recalculateNormals=true;
}



void v4r::Plane_Segmentation::createNormalsFromPoints()
{
    const float weights[][3] = {{1,2,1},
                                {0,0,0},
                                {-1,-2,-1}};
    for(int i=0;i<normals.rows;i++){
        for(int j=0;j<normals.cols;j++){
            Eigen::Vector4f cp=points.at<Eigen::Vector4f>(i+1,j+1);
            if(isnan(cp[2])){
                normals.at<Eigen::Vector4f>(i,j)=Eigen::Vector4f(NAN,NAN,NAN,NAN);
            }else{
                Eigen::Vector4f dpdx(0,0,0,0);
                Eigen::Vector4f dpdy(0,0,0,0);
                for(int k=-1;k<=1;k++){
                    for(int l=-1;l<=1;l++){
                        Eigen::Vector4f p=points.at<Eigen::Vector4f>(i+k+1,j+l+1);
                        if(!isnan(p[2])){
                            if(fabs(cp[2]-p[2])<maxStepSize){
                                float w=weights[k+1][l+1];
                                dpdx+=w*(cp-p);
                                w=weights[l+1][k+1];
                                dpdy+=w*(cp-p);
                            }
                        }
                    }
                }
                //if then the cross product of dpdx,dpdy returns zero, set this normal to pointing towards the camera (lonley point)
                Eigen::Vector3f N3=dpdx.head<3>().cross(dpdy.head<3>());
                Eigen::Vector4f N;
                if(N3[0]==0.0f && N3[1]==0.0f && N3[2]==0.0f){
                    //N[1]=1.0f;//make it green if it fails //having it face towards the camera would be good tough
                    N[0]=N[1]=N[2]=N[3]=NAN;
                }else{
                    //normalize N
                    N3=-N3.normalized();//this takes 11 ms.... or does it?
                    N=Eigen::Vector4f(N3[0],N3[1],N3[2],0);
                }
                normals.at<Eigen::Vector4f>(i,j)=N;
            }

        }
    }
}

void v4r::Plane_Segmentation::createPointsFromDepth(){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            float z = depth.at<float>(i,j);
            Eigen::Vector4f point((j-cx)*z*_fx,
                                  (i-cy)*z*_fy,
                                  z,
                                  0);
            points.at<Eigen::Vector4f>(i+1,j+1)=point;
        }
    }
}

void v4r::Plane_Segmentation::createPointsNormalsFromDepth()
{
    //the first line has to be done upfront
    for(int j=0;j<cols;j++){
        float z = depth.at<float>(0,j);
        if(z>0.0f && !isnan(z)){
            Eigen::Vector4f point((j-cx)*z*_fx,
                                  (0-cy)*z*_fy,
                                  z,
                                  0);
            points.at<Eigen::Vector4f>(1,j+1)=point;
        }
    }

    for(int i=0;i<rows;i++){

        if(i+1<rows){
            float z = depth.at<float>(i+1,0);
            if(z>0.0f && !isnan(z)){
                Eigen::Vector4f point((0-cx)*z*_fx,
                                      (i+1-cy)*z*_fy,
                                      z,
                                      0);
                points.at<Eigen::Vector4f>(i+1+1,1)=point;
            }
        }
        const Eigen::Vector4f nanV(NAN,NAN,NAN,NAN);
        Eigen::Vector4f window[3][3]={{nanV,points.at<Eigen::Vector4f>(i+1-1,1),nanV},//the values on the right get set at the very first
                                      {nanV,points.at<Eigen::Vector4f>(i+1+0,1),nanV},//therefore this is ok.
                                      {nanV,points.at<Eigen::Vector4f>(i+1+1,1),nanV}};
        for(int j=0;j<cols;j++){

            //for this one create new points
            if(i+1<rows && j+1<cols){
                float z = depth.at<float>(i+1,j+1);
                if(z>0.0f && !isnan(z)){
                    Eigen::Vector4f point((j+1-cx)*z*_fx,
                                          (i+1-cy)*z*_fy,
                                          z,
                                          0);
                    points.at<Eigen::Vector4f>(i+1+1,j+1+1)=point;
                }//otherwise NAN
            }
            window[2][2]=points.at<Eigen::Vector4f>(i+1+1,j+1+1);
            window[1][2]=points.at<Eigen::Vector4f>(i+1+0,j+1+1);
            window[0][2]=points.at<Eigen::Vector4f>(i+1-1,j+1+1);

            const float weights[][3] = {{1,2,1},
                                        {0,0,0},
                                        {-1,-2,-1}};

            //calculate tne normals of this window

            Vector4f cp=window[1][1];
            if(isnan(cp[2])){
                normals.at<Eigen::Vector4f>(i,j)=Eigen::Vector4f(NAN,NAN,NAN,NAN);
            }else{
                Eigen::Vector4f dpdx(0,0,0,0);
                Eigen::Vector4f dpdy(0,0,0,0);
                for(int k=-1;k<=1;k++){
                    for(int l=-1;l<=1;l++){
                        Eigen::Vector4f p=window[k+1][l+1];
                        if(!isnan(p[2])){
                            if(fabs(cp[2]-p[2])<maxStepSize){
                                float w=weights[k+1][l+1];
                                dpdx+=w*(cp-p);
                                w=weights[l+1][k+1];
                                dpdy+=w*(cp-p);
                                //std::cout << "something is fishy"<< std::endl;
                            }
                        }
                    }
                }
                //if then the cross product of dpdx,dpdy returns zero, set this normal to pointing towards the camera (lonley point)
                Eigen::Vector3f N3=dpdx.head<3>().cross(dpdy.head<3>());
                Eigen::Vector4f N;
                if(N3[0]==0.0f && N3[1]==0.0f && N3[2]==0.0f){
                    //N[1]=1.0f;//make it green if it fails //having it face towards the camera would be good tough
                    N[0]=N[1]=N[2]=N[3]=NAN;
                }else{
                    //normalize N
                    N3=-N3.normalized();//this takes 11 ms.... or does it?
                    N=Eigen::Vector4f(N3[0],N3[1],N3[2],0);

                }
                normals.at<Eigen::Vector4f>(i,j)=N;

                //TODO: store all this into the PCL point structure:
            }



            //at the end: cycle the windows
            window[0][0]=window[0][1];
            window[1][0]=window[1][1];
            window[2][0]=window[2][1];
            window[0][1]=window[0][2];
            window[1][1]=window[1][2];
            window[2][1]=window[2][2];


        }
    }
}

template <bool doNormalTest>
cv::Mat v4r::Plane_Segmentation::getDebugImage()
{
    int lastPlaneId=0;
    Eigen::Vector4f plane(0,0,0,0);
    //float _planeNorm=1;
    for(int i=0;i<rowsOfPatches;i++){
        for(int j=0;j<colsOfPatches;j++){
            float distThreshold;
            float cosThreshold;
            if(useVariableThresholds){
                //read it from the buffer
                Vector4f thresholds = thresholdsBuffer.at<Vector4f>(i,j);
                distThreshold=thresholds[0];
                cosThreshold=thresholds[1];
            }else{
                distThreshold=maxInlierBlockDist;
                cosThreshold=minCosBlockAngle;
            }
            int planeId=patchIds.at<int>(i,j);
            plane[0]=planes.at<PlaneSegment>(i,j).x;
            plane[1]=planes.at<PlaneSegment>(i,j).y;
            plane[2]=planes.at<PlaneSegment>(i,j).z;
            plane[3]=0;
            if(planeId){
                if(planeId!=lastPlaneId){
                    //plane.head<3>()=planeList[planeId].plane;
                    lastPlaneId=planeId;
                    //_planeNorm=1.0f/plane.norm();
                }
                //cout << "planeId " << planeId << endl;
                //Mark the pixel in the segmentation map for the already existing patches
                if(planes.at<PlaneSegment>(i,j).nrInliers > minAbsBlockInlier){
                    for(int k=0;k<patchDim;k++){
                        for(int l=0;l<patchDim;l++){
                            //mark the points in debug:
                            //honsetly we should still check if the point is an inlier
                            Eigen::Vector4f normal=normals.at<Eigen::Vector4f>(i*patchDim+k,j*patchDim+l);
                            Eigen::Vector4f point=points.at<Eigen::Vector4f>(i*patchDim+k+1,j*patchDim+l+1);
                            if(isInlier<doNormalTest>(point,normal,plane,
                                                      cosThreshold,distThreshold)|| false){
                                segmentation.at<int>(i*patchDim+k,j*patchDim+l)=patchIds.at<int>(i,j);
                            }

                        }
                    }
                }
            }
        }
    }

    return generateColorCodedTextureDebug();
}

template cv::Mat v4r::Plane_Segmentation::getDebugImage<true>();
template cv::Mat v4r::Plane_Segmentation::getDebugImage<false>();

template <bool doNormalTest>
cv::Mat v4r::Plane_Segmentation::getDebugImage(int channel)
{
    int lastPlaneId=0;
    Eigen::Vector4f plane(0,0,0,0);//=planeList[planeId].plane;
    cv::Mat debug2(segmentation.rows,segmentation.cols,CV_32SC1);
    //float _planeNorm=1;
    for(int i=0;i<rowsOfPatches;i++){
        for(int j=0;j<colsOfPatches;j++){
            float distThreshold;
            float cosThreshold;
            if(useVariableThresholds){
                //read it from the buffer
                Vector4f thresholds = thresholdsBuffer.at<Vector4f>(i,j);
                distThreshold=thresholds[0];
                cosThreshold=thresholds[1];
            }else{
                distThreshold=maxInlierBlockDist;
                cosThreshold=minCosBlockAngle;
            }

            int planeId=patchIds.at<int>(i,j);
            if(planeId){
                if(planeId!=lastPlaneId){
                    plane.head<3>()=planeList[planeId].plane;
                    lastPlaneId=planeId;
                    //_planeNorm=1.0f/plane.norm();
                }

                //Mark the pixel in the segmentation map for the already existing patches
                if(planes.at<PlaneSegment>(i,j).nrInliers > minAbsBlockInlier){
                    for(int k=0;k<patchDim;k++){
                        for(int l=0;l<patchDim;l++){
                            //mark the points in debug:
                            //honsetly we should still check if the point is an inlier
                            Eigen::Vector4f normal=normals.at<Eigen::Vector4f>(i*patchDim+k,j*patchDim+l);
                            Eigen::Vector4f point=points.at<Eigen::Vector4f>(i*patchDim+k+1,j*patchDim+l+1);
                            if(isInlier<doNormalTest>(point,normal,plane,
                                                      cosThreshold,distThreshold)){
                                segmentation.at<int>(i*patchDim+k,j*patchDim+l)=planeId;
                            }
                        }
                    }
                }
            }
        }
    }

    return generateColorCodedTextureDebug();
}
template cv::Mat v4r::Plane_Segmentation::getDebugImage<true>(int channel);
template cv::Mat v4r::Plane_Segmentation::getDebugImage<false>(int channel);


template <bool doNormalTest>
cv::Mat v4r::Plane_Segmentation::generateDebugTextureForPlane(Vector4f plane,int index)
{
    cv::Mat colorMap(1,64*48,CV_8UC3);
    colorMap.at<cv::Vec3b>(0)=cv::Vec3b(0,0,0);
    colorMap.at<cv::Vec3b>(1)=cv::Vec3b(0,0,200);
    colorMap.at<cv::Vec3b>(2)=cv::Vec3b(0,200,0);
    colorMap.at<cv::Vec3b>(3)=cv::Vec3b(200,0,0);
    colorMap.at<cv::Vec3b>(4)=cv::Vec3b(0,200,200);
    colorMap.at<cv::Vec3b>(5)=cv::Vec3b(250,0,0);
    colorMap.at<cv::Vec3b>(6)=cv::Vec3b(200,200,200);
    colorMap.at<cv::Vec3b>(7)=cv::Vec3b(0,0,100);
    colorMap.at<cv::Vec3b>(8)=cv::Vec3b(0,100,0);
    colorMap.at<cv::Vec3b>(9)=cv::Vec3b(100,0,0);
    colorMap.at<cv::Vec3b>(10)=cv::Vec3b(0,100,100);
    colorMap.at<cv::Vec3b>(11)=cv::Vec3b(100,100,0);
    colorMap.at<cv::Vec3b>(12)=cv::Vec3b(100,100,100);
    int cols=0;
    int rows=0;
    for (int n=13;n<colorMap.cols;n++){
        colorMap.at<cv::Vec3b>(n)=cv::Vec3b(n/10*50,((n%10)/5)*50,(n%5)*50);
    }
    cv::Mat test(normals.rows,normals.cols,CV_8UC3);
    test.setTo(cv::Scalar(0,0,0));
    for(int i=0;i<normals.rows;i++){
        for(int j=0;j<normals.cols;j++){
            Eigen::Vector4f n=normals.at<Eigen::Vector4f>(i,j);
            Eigen::Vector4f p=points.at<Eigen::Vector4f>(i+1,j+1);
            if(isInlier<doNormalTest>(p,n,plane)){
                test.at<cv::Vec3b>(i,j)=colorMap.at<cv::Vec3b>(0,index);
            }
        }
    }
    return test;
}

cv::Mat v4r::Plane_Segmentation::generateColorCodedTexture()
{
    cv::Mat colorMap(1,64*48,CV_8UC3);
    colorMap.at<cv::Vec3b>(0)=cv::Vec3b(0,0,0);
    colorMap.at<cv::Vec3b>(1)=cv::Vec3b(0,0,200);
    colorMap.at<cv::Vec3b>(2)=cv::Vec3b(0,200,0);
    colorMap.at<cv::Vec3b>(3)=cv::Vec3b(200,0,0);
    colorMap.at<cv::Vec3b>(4)=cv::Vec3b(0,200,200);
    colorMap.at<cv::Vec3b>(5)=cv::Vec3b(250,0,0);
    colorMap.at<cv::Vec3b>(6)=cv::Vec3b(200,200,200);
    colorMap.at<cv::Vec3b>(7)=cv::Vec3b(0,0,100);
    colorMap.at<cv::Vec3b>(8)=cv::Vec3b(0,100,0);
    colorMap.at<cv::Vec3b>(9)=cv::Vec3b(100,0,0);
    colorMap.at<cv::Vec3b>(10)=cv::Vec3b(0,100,100);
    colorMap.at<cv::Vec3b>(11)=cv::Vec3b(100,100,0);
    colorMap.at<cv::Vec3b>(12)=cv::Vec3b(100,100,100);
    int cols=0;
    int rows=0;
    for (int n=13;n<colorMap.cols;n++){
        colorMap.at<cv::Vec3b>(n)=cv::Vec3b(n/10*50,((n%10)/5)*50,(n%5)*50);
    }

    //TODO: take cols and rows from the segmentation Mat
    cols=segmentation.cols;
    rows=segmentation.rows;

    cv::Mat coloredImage(rows,cols,CV_8UC3);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            coloredImage.at<cv::Vec3b>(i,j)=colorMap.at<cv::Vec3b>(0,segmentation.at<int>(i,j));
        }
    }


    return coloredImage;
}
cv::Mat v4r::Plane_Segmentation::generateColorCodedTextureDebug()
{
    cv::Mat colorMap(1,64*48,CV_8UC3);
    colorMap.at<cv::Vec3b>(0)=cv::Vec3b(0,0,0);
    colorMap.at<cv::Vec3b>(1)=cv::Vec3b(0,0,200);
    colorMap.at<cv::Vec3b>(2)=cv::Vec3b(0,200,0);
    colorMap.at<cv::Vec3b>(3)=cv::Vec3b(200,0,0);
    colorMap.at<cv::Vec3b>(4)=cv::Vec3b(0,200,200);
    colorMap.at<cv::Vec3b>(5)=cv::Vec3b(250,0,0);
    colorMap.at<cv::Vec3b>(6)=cv::Vec3b(200,200,200);
    colorMap.at<cv::Vec3b>(7)=cv::Vec3b(0,0,100);
    colorMap.at<cv::Vec3b>(8)=cv::Vec3b(0,100,0);
    colorMap.at<cv::Vec3b>(9)=cv::Vec3b(100,0,0);
    colorMap.at<cv::Vec3b>(10)=cv::Vec3b(0,100,100);
    colorMap.at<cv::Vec3b>(11)=cv::Vec3b(100,100,0);
    colorMap.at<cv::Vec3b>(12)=cv::Vec3b(100,100,100);
    int cols=0;
    int rows=0;
    for (int n=13;n<colorMap.cols;n++){
        colorMap.at<cv::Vec3b>(n)=cv::Vec3b(n/10*50,((n%10)/5)*50,(n%5)*50);
    }

    //TODO: take cols and rows from the segmentation Mat
    cols=segmentation.cols;
    rows=segmentation.rows;

    cv::Mat coloredImage(rows,cols,CV_8UC3);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(segmentation.at<int>(i,j)>0){
                coloredImage.at<cv::Vec3b>(i,j)=colorMap.at<cv::Vec3b>(0,segmentation.at<int>(i,j));
            }else{
                coloredImage.at<cv::Vec3b>(i,j)=debug.at<cv::Vec3b>(i,j);
            }


        }
    }

    //return debug;
    return coloredImage;
}
