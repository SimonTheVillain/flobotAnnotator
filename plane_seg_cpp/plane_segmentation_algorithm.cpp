#include <iostream>
#include <mutex>

#include <vector>
#include <memory>

#include "plane_segmentation.h"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>



//#define DEBUG_IMAGES
//#define DEBUG_TEXT
//#define DEBUG_TIMINGS

using namespace std;
using namespace Eigen;

//TODO: get rid of the recalculation of plane.norm!!!!
template <bool doNormalTest>
inline bool v4r::Plane_Segmentation::isInlier(Eigen::Vector4f point, Eigen::Vector4f normal, Eigen::Vector4f plane,
                                              float cosThreshold, float distThreshold)
{
    float distance = fabs((point.dot(plane)-1.0f)/plane.norm());
    if(distance<distThreshold){
        if(doNormalTest){
        float cosAlpha=normal.dot(plane)/plane.norm();
            if(cosAlpha>cosThreshold){
                return true;
            }else{
                return false;
            }
        }else{
            return true;
        }
    }else{
        return false;
    }
}
template bool v4r::Plane_Segmentation::isInlier<true>(Eigen::Vector4f point, Eigen::Vector4f normal, Eigen::Vector4f plane,
float cosThreshold, float distThreshold);
template bool v4r::Plane_Segmentation::isInlier<false>(Eigen::Vector4f point, Eigen::Vector4f normal, Eigen::Vector4f plane,
float cosThreshold, float distThreshold);

template <bool doNormalTest>
inline bool v4r::Plane_Segmentation::isInlier(Vector4f point, Vector4f normal, Vector4f plane, float _planeNorm,
                                              float cosThreshold, float distThreshold)
{
    float distance = fabs((point.dot(plane)-1.0f)*_planeNorm);
    if(distance<distThreshold){
        if(doNormalTest){//this will get optimized away by the compiler!!
            float cosAlpha=normal.dot(plane)*_planeNorm;
            if(cosAlpha>cosThreshold){
                return true;
            }else{
                return false;
            }
        }else{
            return true;
        }
    }else{
        return false;
    }
}

inline float distanceToPlane(Eigen::Vector4f point,Eigen::Vector4f plane,float _planeNorm){
    return fabs((point.dot(plane)-1.0f)*_planeNorm);
}


inline bool v4r::Plane_Segmentation::isInPlane(Eigen::Vector4f plane1, Eigen::Vector4f plane2,Eigen::Vector4f centerPlane2,
                                               float cosThreshold,float distThreshold)
{
    float dot=plane1.dot(plane2);
    float cosAlpha=dot/(plane1.norm()*plane2.norm());
    //point in plane 2 = plane2*(plane2/(plane2.norm^2)
    float distance=std::abs(1.0f-plane1.dot(centerPlane2))/plane1.norm();//DEBUG!!!!!!!!why does this get zero??????
    if(cosAlpha >cosThreshold && distance<distThreshold){
        return true;
    }else{
        //std::cout << "DEBUG: distance " << distance << " centerPlane " << centerPlane2 << " plane1 " << plane1 << std::endl;
        if(distance>distThreshold){
            //std::cout << "DEBUG: not fitting due to distance" << std::endl;
        }
        if(cosAlpha <cosThreshold){
            //std::cout << "DEBUG: not fitting due to angle" << std::endl;
        }
        return false;
    }
}

inline bool v4r::Plane_Segmentation::isParallel(Vector4f plane1, Vector4f plane2,
                                                float cosThreshold)
{
    float dot=plane1.dot(plane2);
    float cosAlpha=dot/(plane1.norm()*plane2.norm());
    if(cosAlpha >cosThreshold){//maybe use plane angle for this
        return true;
    }else{

        return false;//false
    }

}

Vector4f v4r::Plane_Segmentation::calcPlaneFromMatrix(v4r::Plane_Segmentation::PlaneMatrix m)
{
    Eigen::Matrix3d mat;
    mat <<  m.xx,m.xy,m.xz,
            m.xy,m.yy,m.yz,
            m.xz,m.yz,m.zz;

    //hopefully this is fast!
    Eigen::Vector3d plane=mat.ldlt().solve(m.sum.cast<double>());//what do i know?
    return Eigen::Vector4f(plane[0],plane[1],plane[2],0.0f);
}

void v4r::Plane_Segmentation::replace(int from, int to, int maxIndex)
{
    int *pointer=(int*)patchIds.data;
    for(int i=0;i<maxIndex;i++){
        if(*pointer==from){
            *pointer=to;
        }
        pointer++;
    }
}

int v4r::Plane_Segmentation::allocateMemory()
{
    if(lastInput==2){//if the last input was a depthMap:
        cols=depth.cols;
        rows=depth.rows;
    }else{
        if(lastInput==1){//if the last input was a pointcloud with normals
            cols=_sortedCloud->width;
            rows=_sortedCloud->height;
        }else if(lastInput==3){//pointcloud without normals
            //cols and rows is set in this case
        }else{
            return 0;//no input data
        }
    }
    minAbsBlockInlier=(float)(patchDim*patchDim)*minBlockInlierRatio;
    colsOfPatches=cols/patchDim;
    rowsOfPatches=rows/patchDim;
    _fx=1.0f/fx;
    _fy=1.0f/fy;

    if(nrMatrices!=colsOfPatches*rowsOfPatches){
        if(matrices){
           delete[] matrices;
           delete[] planeList;
           delete[] planeMatrices;
           //delete[] planes;
        }
        nrMatrices=colsOfPatches*rowsOfPatches;
        matrices=new PlaneMatrix[nrMatrices];
        planeMatrices=new PlaneMatrix[nrMatrices];
        planeList=new Plane[rowsOfPatches*colsOfPatches+1];//TODO: eliminate this memory leak
        //planes = new Eigen::Vector3f[nrMatrices];

    }
    //Initializing matrices:
    for(int i=0;i<nrMatrices;i++){
        matrices[i].sum=Eigen::Vector3d(0,0,0);
        matrices[i].xx=0;
        matrices[i].xy=0;
        matrices[i].xz=0;
        matrices[i].yy=0;
        matrices[i].yz=0;
        matrices[i].zz=0;
        matrices[i].nrPoints=0;
        planeMatrices[i]=matrices[i];
    }


    if(segmentation.cols!=cols || segmentation.rows!=rows || segmentation.type()!=CV_32SC1){
        segmentation.create(rows,cols,CV_32SC1);
    }
    segmentation.setTo(cv::Scalar(0));


    if(doZTest){
        if(zBuffer.cols!=cols || zBuffer.rows!=rows || zBuffer.type()!=CV_32SC1){
            zBuffer.create(rows,cols,CV_32FC1);
        }
        zBuffer.setTo(cv::Scalar(10000000.0f));
    }
    if(useVariableThresholds){
        thresholdsBuffer.create(rowsOfPatches,colsOfPatches,CV_32FC4);
    }

    if(normals.cols!=cols || normals.rows!=rows){
        if(points.cols!=cols+2 || points.rows!=rows+2){
            points.create(rows+2,cols+2,CV_32FC4);
        }
        normals.create(rows,cols,CV_32FC4);
        planes.create(rowsOfPatches,colsOfPatches,CV_32FC4);//3 floats 1 integer
        centerPoints.create(rowsOfPatches,colsOfPatches,CV_32FC4);//3 floats 1 integer
        debug.create(rows,cols,CV_8UC3);
        debug.setTo(cv::Scalar(0,0,0));

        patchIds.create(rowsOfPatches,colsOfPatches,CV_32SC1);
    }
    if(lastInput!=3){//if the points are not already set in the setInput method
        points.setTo(cv::Scalar(NAN,NAN,NAN,0));
    }
    return 1;
}

template <bool doNormalTest>
void v4r::Plane_Segmentation::calculatePlaneSegments()
{
    //create the blockwise plane description
    for(int i=0;i<rowsOfPatches;i++){
        for(int j=0;j<colsOfPatches;j++){
            PlaneMatrix pm={};
            for(int m=0;m<patchDim;m++){
                for(int n=0;n<patchDim;n++){
                    Eigen::Vector4f point=points.at<Eigen::Vector4f>(i*patchDim+m+1,j*patchDim+n+1);
                    if(point[2]>0.0f && !isnan(point[2])){

                        pm.sum+=point.cast<double>().head<3>();
                        pm.xx+=point[0]*point[0];
                        pm.xy+=point[0]*point[1];
                        pm.xz+=point[0]*point[2];
                        pm.yy+=point[1]*point[1];
                        pm.yz+=point[1]*point[2];
                        pm.zz+=point[2]*point[2];
                        pm.nrPoints++;
                    }
                }
            }
            int index=i*colsOfPatches+j;
            matrices[index]=pm;

        }
    }


    //calculate all plane segments
    for(int i=0;i<rowsOfPatches;i++){
        for(int j=0;j<colsOfPatches;j++){

            int index=j+i*colsOfPatches;
            PlaneMatrix m=matrices[index];
            centerPoints.at<Eigen::Vector4f>(i,j).head<3>()=m.sum.cast<float>()/(float)m.nrPoints;
            centerPoints.at<Eigen::Vector4f>(i,j)[3]=0.0f;
            float cosThreshold = minCosAngle;
            float distThreshold= maxInlierDist;
            if(useVariableThresholds && m.nrPoints>0){//TODO: template this

                float z=centerPoints.at<Vector4f>(i,j)[2];
                Vector4f thresholds;
                thresholds[0]=maxInlierBlockDistFunc(z);
                thresholds[1]=minCosBlockAngleFunc(z);
                distThreshold=maxInlierDistFunc(z);
                thresholds[2]=distThreshold;
                cosThreshold=minCosAngleFunc(z);
                thresholds[3]=cosThreshold;
                thresholdsBuffer.at<Vector4f>(i,j)=thresholds;
            }else{
                thresholdsBuffer.at<Vector4f>(i,j)=Vector4f(-1,-1,-1,-1);
            }

            if(m.nrPoints > minAbsBlockInlier){
                Eigen::Vector4f plane=calcPlaneFromMatrix(m);//what do i know?
               //invert matrix and create plane estimation
                planes.at<PlaneSegment>(i,j).x=plane[0];//plane.cast<float>();
                planes.at<PlaneSegment>(i,j).y=plane[1];
                planes.at<PlaneSegment>(i,j).z=plane[2];
                planes.at<PlaneSegment>(i,j).nrInliers=m.nrPoints;


                //Calculate Thresholds here:
                //TODO!!!!!!

                Eigen::Vector4f plane4(plane[0],plane[1],plane[2],0);
                float _planeNorm=1.0f/plane4.norm();


                Eigen::Vector4f N;
                for(int k=0;k<patchDim;k++){
                    for(int l=0;l<patchDim;l++){
                        int u=l+j*patchDim;
                        int v=k+i*patchDim;
                        Eigen::Vector4f p=points.at<Eigen::Vector4f>(v+1,u+1);
                        if(doNormalTest){
                            N=normals.at<Eigen::Vector4f>(v,u);
                        }

                        //TODO: remove this isInlier.... or at least store the norm for this so it does not have to be recalculated for every pixel
                        if(isInlier<doNormalTest>(p,N,plane4,_planeNorm,
                                                  cosThreshold,distThreshold)){ //distance < inlierDistance
                            //mark the inlying points somehow
#ifdef DEBUG_IMAGES
                            debug.at<cv::Vec3b>(v,u)=cv::Vec3b(255,0,0);
#endif
                            segmentation.at<int>(v,u)=-1;//mark every valid element in the segmentation
                        }else{
                            planes.at<PlaneSegment>(i,j).nrInliers--;
                        }

                    }
                }
                if(planes.at<PlaneSegment>(i,j).nrInliers>minAbsBlockInlier){
                    for(int k=0;k<patchDim;k++){
                        for(int l=0;l<patchDim;l++){
                            int u=l+j*patchDim;
                            int v=k+i*patchDim;
                            Eigen::Vector4f p=points.at<Eigen::Vector4f>(v+1,u+1);
                            if(doNormalTest){
                                N=normals.at<Eigen::Vector4f>(v,u);
                            }

                            //TODO: remove this isInlier.... or at least store the norm for this so it does not have to be recalculated for every pixel
                            if(isInlier<doNormalTest>(p,N,plane4,_planeNorm,
                                                      cosThreshold,distThreshold)){ //distance < inlierDistance
                                //mark the inlying points somehow
#ifdef DEBUG_IMAGES
                                debug.at<cv::Vec3b>(v,u)=cv::Vec3b(0,255,0);
#endif

                            }

                        }
                    }
                }

            }else{
                planes.at<PlaneSegment>(i,j).x=NAN;//=Eigen::Vector3f(NAN,NAN,NAN);
                planes.at<PlaneSegment>(i,j).y=NAN;//
                planes.at<PlaneSegment>(i,j).z=NAN;//
                planes.at<PlaneSegment>(i,j).nrInliers=0;//
            }
        }
    }
}

void v4r::Plane_Segmentation::rawPatchClustering()
{
    //TODO: make this code more compact somehow (and readable)
    for(int i=0;i<rowsOfPatches;i++){
        PlaneMatrix currentPlaneMatrix;
        Eigen::Vector4f currentPlane(0,0,0,0);
        Eigen::Vector4f lastPatch(0,0,0,0);
        //int lastId=0;
        int currentId=0;
        for(int j=0;j<colsOfPatches;j++){
            int index=j+i*colsOfPatches;
            currentPlaneMatrix=matrices[index];
            PlaneSegment currentPlaneSeg=planes.at<PlaneSegment>(i,j);//TODO:planes should be renamed to patches
            Eigen::Vector4f currentPatch(currentPlaneSeg.x,currentPlaneSeg.y,currentPlaneSeg.z,0);
            Eigen::Vector4f currentCenter=centerPoints.at<Eigen::Vector4f>(i,j);
            //test if plane is valid
            bool gotSet=false;

            if(currentPlaneSeg.nrInliers>minAbsBlockInlier){
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
                bool alreadyTested=false;
                //test if lastID is valid
                if(currentId!=0){
                    //test if the new Plane element fits to the existing one
                    Eigen::Vector4f otherPlane;



                    otherPlane=calcPlaneFromMatrix(planeMatrices[currentId]);
                    if(     isInPlane(lastPatch,currentPatch,currentCenter,cosThreshold,distThreshold) &&
                            isParallel(currentPatch,otherPlane,cosThreshold)){
                        gotSet=true;
                        patchIds.at<int>(i,j)=currentId;

                        currentPlane=otherPlane;

                        planeMatrices[currentId]+=currentPlaneMatrix;
                        //if this here is the case, it is not necessary to test the top and top left plane segment
                        //except for the first row(column?)... there the top element should be tested
                        //Note: this does not bring much of an speed advantage (none)
                        //alreadyTested=true;
                    }else{//debug attempt
                        currentPlane=currentPatch;
                        currentId=0;//SIMON DEBUG ATTEMPT
                    }
                }else{//debug attempt
                    currentPlane=currentPatch;
                }

                lastPatch=currentPatch;
                //test if one of the 3 upper elements is already segmented and connect planes if necessary
                if(i>0){
                    Eigen::Vector4f newPlane(0,0,0,0);
                    //Eigen::Vector4f newPlane(0,0,0,0);
                    if(j>0 && !alreadyTested){
                        //do upper left
                        int newId=patchIds.at<int>(i-1,j-1);// it is only testing for blocks from the past(so the threshold is already checked
                        if(newId){
                            PlaneSegment currentPlaneSeg=planes.at<PlaneSegment>(i-1,j-1);
                            Eigen::Vector4f newPatch(currentPlaneSeg.x,currentPlaneSeg.y,currentPlaneSeg.z,0);

                            newPlane=calcPlaneFromMatrix(planeMatrices[newId]);
                            if(     isInPlane(newPatch,currentPatch,currentCenter,cosThreshold,distThreshold) &&
                                    isParallel(currentPatch,newPlane,cosThreshold)){
                                gotSet=true;
                                //test if this is the right time to connect
                                if(!currentId){//if the current patch does not have an ID yet just add this one:
                                    currentId=newId;
                                    patchIds.at<int>(i,j)=currentId;
                                    planeMatrices[currentId]+=currentPlaneMatrix;
                                }else{
                                    if(currentId!=newId){
                                        int nrCurrent=planeMatrices[currentId].nrPoints;
                                        int nrNew=planeMatrices[newId].nrPoints;


                                        if(nrCurrent>nrNew){//replace the one with fewer elements:
                                            replace(newId,currentId,j+i*colsOfPatches);
                                            planeMatrices[currentId]+=planeMatrices[newId];
                                            planeMatrices[newId].nrPoints=0;
                                        }else{
                                            replace(currentId,newId,j+i*colsOfPatches);
                                            planeMatrices[newId]+=planeMatrices[currentId];
                                            planeMatrices[currentId].nrPoints=0;
                                            currentId=newId;

                                        }
                                        patchIds.at<int>(i,j)=currentId;
                                    }
                                }
                                currentPlane.head<3>()=planeList[currentId].plane;
                            }
                        }
                    }

                    //upper
                    if(!alreadyTested || j==0){//forget alreadyTested. it does not make sense
                        int newId=patchIds.at<int>(i-1,j);
                        if(newId){
                            newPlane.head<3>() = planeList[newId].plane;
                            PlaneSegment currentPlaneSeg=planes.at<PlaneSegment>(i-1,j);
                            Eigen::Vector4f newPatch(currentPlaneSeg.x,currentPlaneSeg.y,currentPlaneSeg.z,0);

                            newPlane=calcPlaneFromMatrix(planeMatrices[newId]);
                            if(     isInPlane(newPatch,currentPatch,currentCenter,cosThreshold,distThreshold) &&
                                    isParallel(currentPatch,newPlane,cosThreshold)){
                                gotSet=true;
                                //test if this is the right time to connect
                                if(!currentId){//if the current patch does not have an ID yet just add this one:
                                    currentId=newId;
                                    //currentPlane.head<3>()=planeList[currentId].plane;//maybe not needed anymore
                                    patchIds.at<int>(i,j)=currentId;
                                    planeMatrices[currentId]+=currentPlaneMatrix;
                                }else{
                                    if(currentId!=newId){
                                        int nrCurrent=planeMatrices[currentId].nrPoints;
                                        int nrNew=planeMatrices[newId].nrPoints;

                                        if(nrCurrent>nrNew){//replace the one with fewer elements:

                                            replace(newId,currentId,j+i*colsOfPatches);

                                            planeMatrices[currentId]+=planeMatrices[newId];
                                            planeMatrices[newId].nrPoints=0;
                                        }else{

                                            replace(currentId,newId,j+i*colsOfPatches);
                                            planeMatrices[newId]+=planeMatrices[currentId];
                                            planeMatrices[currentId].nrPoints=0;
                                            currentId=newId;
                                        }
                                        patchIds.at<int>(i,j)=currentId;
                                    }
                                    currentPlane.head<3>()=planeList[currentId].plane;
                                }
                            }
                        }
                    }

                    //upper right
                    if(j+1<colsOfPatches){
                        int newId=patchIds.at<int>(i-1,j+1);
                        if(newId){
                            newPlane.head<3>() = planeList[newId].plane;
                            PlaneSegment currentPlaneSeg=planes.at<PlaneSegment>(i-1,j+1);
                            Eigen::Vector4f newPatch(currentPlaneSeg.x,currentPlaneSeg.y,currentPlaneSeg.z,0);

                            newPlane=calcPlaneFromMatrix(planeMatrices[newId]);
                            if(     isInPlane(newPatch,currentPatch,currentCenter,cosThreshold,distThreshold) &&
                                    isParallel(currentPatch,newPlane,cosThreshold)){
                                gotSet=true;
                                //test if this is the right time to connect
                                if(!currentId){//if the current patch does not have an ID yet just add this one:
                                    currentId=newId;

                                    //currentPlane.head<3>()=planeList[currentId].plane;
                                    patchIds.at<int>(i,j)=currentId;
                                    planeMatrices[currentId]+=currentPlaneMatrix;
                                }else{
                                    if(currentId!=newId){
                                        int nrCurrent=planeMatrices[currentId].nrPoints;
                                        int nrNew=planeMatrices[newId].nrPoints;

                                        if(nrCurrent>nrNew){//replace the one with fewer elements:

                                            replace(newId,currentId,j+i*colsOfPatches);
                                            planeMatrices[currentId]+=planeMatrices[newId];
                                            planeMatrices[newId].nrPoints=0;
                                        }else{

                                            replace(currentId,newId,j+i*colsOfPatches);
                                            planeMatrices[newId]+=planeMatrices[currentId];
                                            planeMatrices[currentId].nrPoints=0;
                                            currentId=newId;
                                        }
                                        //howOften++;
                                        patchIds.at<int>(i,j)=currentId;
                                    }
                                }
                                currentPlane.head<3>()=planeList[currentId].plane;
                            }
                        }

                    }

                }

                if(!gotSet){
                    currentId=0;
                }
                //in case the current ID could not be connected to an already existing plane:
                //create a new one
                if(currentId==0){
                    //create a new id
                    currentId=++maxId;
                    patchIds.at<int>(i,j)=currentId;

                    planeMatrices[currentId]=currentPlaneMatrix;
                    currentPlane=currentPatch;
                }

            }else{//if the current patch does not contain enought members
                currentId=0;
            }
        }
    }
}

template <bool doNormalTest,bool reverse,bool zTest>
inline void v4r::Plane_Segmentation::postProcessing1Direction(const int offsets[][2]){
    //TODO: add a distance buffer to store which patch is the better fit for a segment
    for(int i=reverse ? segmentation.rows-1 : 0 ;reverse ? i>0 : i<segmentation.rows; reverse ? i-- : i++){
        int oldId=0;
        Eigen::Vector4f oldPlane;
        float _oldPlaneNorm=0;
        int oldPlaneTTL=0;
        float cosThreshold=minCosAngle;
        float distThreshold=maxInlierDist;
        for(int j=reverse ? segmentation.cols-1 : 0 ;reverse ? j>0 : j<segmentation.cols; reverse ? j-- : j++){
            int currentId=segmentation.at<int>(i,j);
            if(currentId>0){
                // First step is to check for the surrounding patches if there are elements of the current segment ID

                if(currentId!=oldId || !oldPlaneTTL){
                    oldId=currentId;
                    bool found=false;
                    const int surrounding[9][2]={{0,0},{-1,-1},{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1}};
                    //search the plane in one of the surrounding patches
                    for(int k=0;k<9;k++){
                        int _i=i/patchDim+surrounding[k][0];
                        int _j=j/patchDim+surrounding[k][0];
                        if(_i>=0 && _j>=0 && _i<rowsOfPatches && j<colsOfPatches){
                            //TODO: do the readout for the fitting thresholds here!!!!
                            ///TODOOOOOOOOO!!!!
                            if(oldId==patchIds.at<int>(_i,_j)){
                                //read according patch segment
                                PlaneSegment p=planes.at<PlaneSegment>(_i,_j);
                                oldPlane = Eigen::Vector4f(p.x,p.y,p.z,0);
                                _oldPlaneNorm = 1.0f/oldPlane.norm();
                                found=true;


                                //is this really the best place for reading out the thresholds?
                                if(useVariableThresholds){
                                    Vector4f thresholds=thresholdsBuffer.at<Vector4f>(_i,_j);
                                    distThreshold=thresholds[2];
                                    cosThreshold=thresholds[3];
                                }

                                break;//end this loop

                            }
                        }
                    }
                    if(!found){
                        //if no fitting patch is found. we take the plane with this id
                        Plane p=resultingPlanes[oldId-1];
                        oldPlane=Eigen::Vector4f(p.plane[0],p.plane[1],p.plane[2],0);
                        _oldPlaneNorm = 1.0f/oldPlane.norm();


                        //the thresholds are used from the fitting patch if no other threshold is found
                        //DON'T KNOW WHY THIS DOES NOT WORK!!!
                        //TODO: FIND THIS BUG!!!!!
                        ///BUG:
                        if(useVariableThresholds){
                            Vector4f thresholds=thresholdsBuffer.at<Vector4f>(i/patchDim,j/patchDim);
                            distThreshold=thresholds[2];
                            cosThreshold=thresholds[3];
                        }

                    }
                    oldPlaneTTL=patchDim;
                }else{
                    oldPlaneTTL--;
                }
                //const int offsets[4][2]={{0,1},{1,-1},{1,0},{1,1}};
                if(oldId){
                    for(int k=0;k<4;k++){
                        int _i=i+offsets[k][0];
                        int _j=j+offsets[k][1];
                        if(_i>=0 && _j>=0 && _i<segmentation.rows && _j<segmentation.cols){
                            int otherId=segmentation.at<int>(_i,_j);

                            if(otherId<=0 || zTest){//only do this if pixel is not yet set
                                //test if the pixel is inside of oldPlane and set the pixel accordingly
                                Eigen::Vector4f otherPoint=points.at<Eigen::Vector4f>(_i+1,_j+1);
                                float newDist = distanceToPlane(otherPoint,oldPlane,_oldPlaneNorm);
                                float oldDist=0;
                                if(zTest){
                                    oldDist = zBuffer.at<float>(_i,_j);
                                }
                                if(newDist<oldDist || !zTest){
                                    Eigen::Vector4f otherNormal;
                                    if(doNormalTest){
                                        otherNormal=normals.at<Eigen::Vector4f>(_i,_j);
                                    }
                                    if(isInlier<doNormalTest>(otherPoint,otherNormal,oldPlane,_oldPlaneNorm,
                                                              cosThreshold,distThreshold)){
                                        segmentation.at<int>(_i,_j)=oldId;
                                        if(zTest){
                                            zBuffer.at<float>(_i,_j)=newDist;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if(false){
                        std::cout << "DEBUG: currentId " << currentId << std::endl;
                        cv::imshow("current",generateColorCodedTexture());
                        cv::waitKey(1);
                    }
                }
            }
        }
    }
    //cv::imshow("debug2",debug2);
    //cv::waitKey();
}

template <bool doNormalTest,bool zTest>
void v4r::Plane_Segmentation::postProcessing(){
    //TODO: find out why this does not deliver the same results as the code below
    const int offsets1[4][2]={{0,1},{1,-1},{1,0},{1,1}};
    postProcessing1Direction<doNormalTest,false,zTest>(offsets1);

    const int offsets2[4][2]={{0,-1},{-1,-1},{-1,0},{-1,1}};
    postProcessing1Direction<doNormalTest,true,zTest>(offsets2);
}


v4r::Plane_Segmentation::Plane_Segmentation()
{

    maxStepSize=0.1f;//max step size has 5cm (maybe make it dependant on distance)
    minBlockInlierRatio=0.9f;
    maxAngle=M_PI/180.0f*10.0f;//10° max angle(maybe make it dependant on distance)
    maxInlierDist=0.01f;//1cm inlier distance(maybe make it dependant on distance)
    maxInlierBlockDist=0.005f;
    //minCosAngle=cos(maxAngle);

    maxBlockAngle=M_PI/180.0f*10.0f;
    minCosBlockAngle=cos(maxBlockAngle);


    matrices=NULL;
    nrMatrices=0;

    lastInput=0;

    recalculateNormals=false;
    cols=0;
    rows=0;


    //define the functions for thresholds:

    maxInlierBlockDistFunc=[] (float z) -> float {
        //return 0.005f;//5mm
        float zmin=0.4f;
        float zmax=4.0f;
        float thresholdmin=0.015f;//15mm
        float thresholdmax=0.07f;//5cm
        return thresholdmin+(z-zmin)*(z-zmin)/(zmax*zmax)*(thresholdmax-thresholdmin);
    };

    minCosBlockAngleFunc=[] (float z) -> float {
        //return cos(maxBlockAngle_local);
        float zmin=0.4f;//distance measurement starts at 0.4m
        float zmax=4.0f;
        float alphamin=15.0f;
        float alphamax=60.0f;//40
        float maxAngle_local=M_PI/180.0f*min(alphamin+(z-zmin)*(z-zmin)/(zmax*zmax)*(alphamax-alphamin),90.0f);
        return cos(maxAngle_local);
    };

    minCosAngleFunc = [] (float z) -> float {
        float zmin=0.4f;//distance measurement starts at 0.4m
        float zmax=4.0f;
        float alphamin=40.0f;
        float alphamax=90.0f;
        float maxAngle_local=M_PI/180.0f*min(alphamin+(z-zmin)*(z-zmin)/(zmax*zmax)*(alphamax-alphamin),90.0f);
        return cos(maxAngle_local);
    };


    maxInlierDistFunc = [] (float z) -> float {
        //return 0.01f;
        float zmin=0.4f;
        float zmax=4.0f;
        float thresholdmin=0.005f;//5mm
        float thresholdmax=0.05f;//10cm
        return thresholdmin+(z-zmin)*(z-zmin)/(zmax*zmax)*(thresholdmax-thresholdmin);
    };


    //TODOOOOO
    //maxInlierDistFunc;
            //maxInlierBockDistFunc=
            //minCosAngleFunc
            //minCosBlockAngleFunc

}

v4r::Plane_Segmentation::~Plane_Segmentation()
{
    //right now there is nothing to free
    //TODO: check if the matrices member should be freed
}


void v4r::Plane_Segmentation::setMaxAngle(float angle)
{
    maxAngle=angle;
    minCosAngle=cos(maxAngle);
}void v4r::Plane_Segmentation::setMaxBlockAngle(float angle)
{
    maxBlockAngle=angle;
    minCosBlockAngle=cos(maxBlockAngle);
}

int v4r::Plane_Segmentation::segment()
{
#ifdef DEBUG_TIMINGS
    auto start = std::chrono::system_clock::now();//timing
#endif


    if(!allocateMemory()){
        return 0; //no input data
    }


    if(lastInput==2){//if the input only was a depthmap

        if(pointwiseNormalCheck){
            createPointsNormalsFromDepth();
            cout << "we did normal calculation" << endl;

        }else{
            //createPointsNormalsFromDepth();
            //normals.setTo(cv::Scalar(0));//TODO: why do we need normals
            createPointsFromDepth();
        }

    }else{//if the input is from a pointcloud
        //get the points from the input pointcloud:
        if(lastInput==3){//if we only got points
            //the points Mat is already set
        }else{//if we got points + normals
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    pcl::PointXYZRGBNormal p = _sortedCloud->at(j,i);
                    points.at<Eigen::Vector4f>(i+1,j+1)=Eigen::Vector4f(p.x,p.y,p.z,0);
                    normals.at<Eigen::Vector4f>(i,j)=Eigen::Vector4f(p.normal_x,p.normal_y,p.normal_z,0);
                }
            }
        }
        if(recalculateNormals){
            if(pointwiseNormalCheck){
                createNormalsFromPoints();
            }
            //std::cout << "Calculating normals from points" << std::endl;
        }
    }


#ifdef DEBUG_TIMINGS
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start=end;
    //present
    std::cout << "initialization: " << elapsed.count() << "microseconds" << std::endl;
#endif


#ifdef DEBUG_IMAGES
    cv::imshow("normals",normals);
    cv::imshow("points",points);
#endif

    //calculating a patchwise plane description
    if(pointwiseNormalCheck){
        calculatePlaneSegments<true>();
    }else{
        calculatePlaneSegments<false>();
    }
#ifdef DEBUG_IMAGES
    cv::imshow("debug",debug);
#endif


    //setting the list of planes to zero
    //TODO: this list of planes is not needed anyway
    /*for(int i=0;i<heightBlocks*widthBlocks+1;i++){
        planeList[i].plane=Eigen::Vector3f(0,0,0);
        planeList[i].nrElements=0;
    }*/
    //cv::Mat patchIds(heightBlocks,widthBlocks,CV_32SC1);//TODO: init this later
    patchIds.setTo(cv::Scalar(0));//test if this is necessary (but i think so)
    maxId=0;//everytime a new id is created, it is done by this number

    rawPatchClustering();


#ifdef DEBUG_TIMINGS
    end = std::chrono::system_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start=end;
    //present
    std::cout << "rawPatchClustering: " << elapsed.count() << "microseconds" << std::endl;

#endif
    //create vector of planes
    resultingPlanes.resize(0);//clear the vector of resulting planes
    int* newPlaneIds=new int[maxId+1];
    int newId=1;
    for(int i=0;i<=maxId;i++){
        PlaneMatrix pm = planeMatrices[i];
        if(pm.nrPoints>minNrPatches*patchDim*patchDim){
            newPlaneIds[i]=newId++;
            Plane pl;
            Eigen::Vector4f p = calcPlaneFromMatrix(pm);
            pl.plane=Eigen::Vector3f(p[0],p[1],p[2]);
            pl.nrElements=planeMatrices[i].nrPoints;
            resultingPlanes.push_back(pl);
        }else{
            newPlaneIds[i]=0;
        }
    }


    for(int i=0;i<rowsOfPatches;i++){
        for(int j=0;j<colsOfPatches;j++){
            int planeId=patchIds.at<int>(i,j);
            int newPlaneId=newPlaneIds[planeId];
            patchIds.at<int>(i,j)=newPlaneId;

            PlaneSegment p=planes.at<PlaneSegment>(i,j);
            Vector4f plane = Eigen::Vector4f(p.x,p.y,p.z,0);
            float _planeNorm = 1.0f/plane.norm();
            if(newPlaneId){
                //Mark the pixel in the segmentation map for the already existing patches
                if(planes.at<PlaneSegment>(i,j).nrInliers > minAbsBlockInlier){
                    for(int k=0;k<patchDim;k++){
                        for(int l=0;l<patchDim;l++){
                            if(segmentation.at<int>(i*patchDim+k,j*patchDim+l)==-1){
                                //we already touched these points and found out if they are in there.(they are marked with -1)
                                segmentation.at<int>(i*patchDim+k,j*patchDim+l)=newPlaneId;

                                Vector4f point = points.at<Vector4f>(i*patchDim+k+1,j*patchDim+l+1);
                                float distance = distanceToPlane(point,plane,_planeNorm);
                                if(doZTest){
                                    //setting the zBuffer to zero effectively sets these patches to be fixed
                                    zBuffer.at<float>(i*patchDim+k,j*patchDim+l)=distance*0.0f;
                                }
                            }
                        }
                    }
                }else{
                    for(int k=0;k<patchDim;k++){
                        for(int l=0;l<patchDim;l++){
                            if(segmentation.at<int>(i*patchDim+k,j*patchDim+l)==-1){
                                segmentation.at<int>(i*patchDim+k,j*patchDim+l)=0;
                                //segmentation is set to zero at every other points
                            }
                        }
                    }
                }

            }
        }
    }
    delete[] newPlaneIds;// do not create memory leaks

#ifdef DEBUG_TIMINGS
    end = std::chrono::system_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start=end;
    //present
    std::cout << "prePostProcessing: " << elapsed.count() << "microseconds" << std::endl;
#endif

    //postProcessing(newId);
/*
    std::cout << "number of resulting planes" << resultingPlanes.size() << std::endl;
    cv::imshow("beforePost",generateColorCodedTexture());
    for(int i=0;i<resultingPlanes.size();i++){
        Plane pl=resultingPlanes[i];
        Eigen::Vector4f p(pl.plane[0],pl.plane[1],pl.plane[2],0);
        cv::imshow("planeDebug",generateDebugTextureForPlane(p,i+1));
        cv::waitKey();
    }*/
#ifdef DEBUG_IMAGES
    cv::imshow("beforePost",generateColorCodedTexture());
#endif
    if(pointwiseNormalCheck){
        if(doZTest){
            postProcessing<true,true>();
        }else{
            postProcessing<true,false>();
        }

    }else{
        if(doZTest){
            postProcessing<false,true>();
        }else{
            postProcessing<false,false>();
        }

    }

#ifdef DEBUG_IMAGES
    cv::imshow("afterPost",generateColorCodedTexture());
    cv::waitKey(1);
#endif

#ifdef DEBUG_TIMINGS
    end = std::chrono::system_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start=end;
    //present
    std::cout << "postProcessing: " << elapsed.count() << "microseconds" << std::endl;
#endif


    return 1;
}
void v4r::Plane_Segmentation::dubiousLinewisePostProcessing(){
    float distThreshold=0;
    for(int i=0;i<segmentation.rows;i++){
        distThreshold=0;
        for(int j=1;j<segmentation.cols;j++){
            if(segmentation.at<int>(i,j)==0 && !isnan(points.at<Eigen::Vector4f>(i+1,j+1)[0])){
                if(segmentation.at<int>(i,j-1)!=0){
                    if(distThreshold==0){
                        float z=points.at<Eigen::Vector4f>(i+1,j)[2];
                        distThreshold=maxInlierDistFunc(z)*1.5f;//read out threshold for this thingy
                        distThreshold=0.05f;
                    }
                    Eigen::Vector3f plane3=resultingPlanes[segmentation.at<int>(i,j-1)-1].plane;//read out plane equation for current plane
                    Eigen::Vector4f plane=Eigen::Vector4f(plane3[0],plane3[1],plane3[2],0);
                    //calc distance to plane
                    Eigen::Vector4f point=points.at<Eigen::Vector4f>(i+1,j+1);
                    float dist=distanceToPlane(point,plane,plane.norm());
                    if(dist<distThreshold){
                        segmentation.at<int>(i,j)=segmentation.at<int>(i,j-1);
                    }
                }
            }

        }
    }
}



