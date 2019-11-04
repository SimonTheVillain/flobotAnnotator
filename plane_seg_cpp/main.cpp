#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>//for the optical flow


#include "plane_segmentation.h"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cb(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input){

	pcl::PCLPointCloud2 pcl_pc2;
	pcl_conversions::toPCL(*input,pcl_pc2);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
	//do stuff with temp_cloud here
	return temp_cloud;
}


int main(int argc, char *argv[])
{

	//plane prior to ground plane
	Eigen::Vector3f groundPlaneTheory;
	float maxAngleToGroundPlaneTheory;
	groundPlaneTheory = Eigen::Vector3f(-0.123342, 0.883647, 1.11059);//.normalize();
	groundPlaneTheory = groundPlaneTheory/groundPlaneTheory.norm();
	maxAngleToGroundPlaneTheory = M_PI/180.0f*40.0f;


	pcl::PointCloud<pcl::PointXYZRGB> groundCloud;

	v4r::Plane_Segmentation segmentation;



	//std::string in_path = "/home/simon/datasets/flobot/carugate/freezer_section_dirt/2018-04-18-10-20-53.bag";

	//std::string path_out = "/home/simon/datasets/flobot/carugate_annotated/dirty/mask/";
	std::string in_path = "/home/simon/datasets/flobot/lyon/2018-06-13-15-46-35_cola_spots.bag";

	std::string path_out = "/home/simon/datasets/flobot/lyon_annotated/dirty/mask/";

	bool additional_mask_enabled = true;
	std::string additional_mask_path = "/home/simon/datasets/flobot/lyon_annotated/dirty/laser_mask.png";


	cv::Mat additional_mask;
	if(additional_mask_enabled){
		additional_mask = cv::imread(additional_mask_path,cv::IMREAD_GRAYSCALE);
	}

	rosbag::Bag bag;
	bag.open(in_path,rosbag::bagmode::Read);  // BagMode is Read by default


    std::vector<std::string> topics;
    topics.push_back(std::string("/camera/depth_registered/points"));

    rosbag::View view(bag, rosbag::TopicQuery(topics));
	int image_count = 0;
	for(rosbag::MessageInstance const m: view)//rosbag::View(bag))
	{
		sensor_msgs::PointCloud2::ConstPtr pc = m.instantiate<sensor_msgs::PointCloud2>();
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud = cloud_cb(pc);

		segmentation.setInput(temp_cloud);

		segmentation.segment();
		segmentation.dubiousLinewisePostProcessing();

		int groundPlane = 0;
		float maxCosAngle = -1;
		int maxNrInliers = 0;
		Eigen::Vector3f plane;


		for(int i=0;i<segmentation.resultingPlanes.size();i++){
			//std::cout << "Plane with " << segmentation.resultingPlanes[i].nrElements << "points:" << std::endl <<
			//             segmentation.resultingPlanes[i].plane << std::endl;

			Eigen::Vector3f p=segmentation.resultingPlanes[i].plane;
			float cosAngle = p.dot(groundPlaneTheory)/p.norm();
			if(cosAngle>cos(maxAngleToGroundPlaneTheory)){//&&cosAngle>maxCosAngle){
				if(segmentation.resultingPlanes[i].nrElements>maxNrInliers){
					maxCosAngle = cosAngle;
					maxNrInliers = segmentation.resultingPlanes[i].nrElements;
					groundPlane = i+1;
					plane = p;
				}

			}
			//for debugging purposes
#ifdef DEBUG_VIEW
			cv::imshow("planes",segmentation.generateColorCodedTexture());
        cv::waitKey(1);
#endif
		}
		if(groundPlane==0){
			std::cerr << "No ground plane detected" << std::endl;
		}else {
			pcl::PointXYZRGB nanPoint;
			nanPoint.x = nanPoint.y = nanPoint.z = NAN;

			static cv::Mat mask;
			static cv::Mat dilatedMask;
			static cv::Mat dilationElement;
			//separate the plane and release it
			if (groundCloud.width != temp_cloud->width || groundCloud.height != temp_cloud->height) {
				groundCloud.width = temp_cloud->width;
				groundCloud.height = temp_cloud->height;
				groundCloud.is_dense = true;
				groundCloud.points.resize(groundCloud.width * groundCloud.height);
				mask.create(temp_cloud->height, temp_cloud->width, CV_8UC1);
				int dilation_size = 2;
				dilationElement = cv::getStructuringElement(cv::MORPH_RECT,
															cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
															cv::Point(dilation_size, dilation_size));
			}

			for (int i = 0; i < temp_cloud->height; i++) {
				for (int j = 0; j < temp_cloud->width; j++) {
					if (segmentation.segmentation.at<int>(i, j) == groundPlane &&
						!additional_mask.at<unsigned char>(i,j) ) {
						mask.at<unsigned char>(i, j) = 255;
					} else {
						mask.at<unsigned char>(i, j) = 0;
					}
				}
			}
			//Doing a slight dilation to extend the floor for about 2 pixel
			cv::dilate(mask, dilatedMask, dilationElement);
			cv::imshow("mask",mask);
			cv::waitKey(1);
			imwrite(path_out + std::to_string(image_count) +".png",mask);
			image_count++;
			/*
			cv::Mat roi(mask.rows,mask.cols,CV_8UC1);
			roi.setTo(0);
			for(int i=0;i<temp_cloud->height;i++){
				for(int j=0;j<temp_cloud->width;j++){
					pcl::PointXYZRGB point = temp_cloud->at(j,i);

					//image.at<cv::Vec3b>(i,j)=cv::Vec3b(point.b,point.g,point.r);
					if(fabs(point.x) < detectionVolumeWidth*0.5f &&
					   point.z < detectionVolumeDepth &&
					   !isnan(point.x)){// &&
						//mask.at<unsigned char>(i,j)==0){

						roi.at<unsigned char>(i,j)=200;
						Eigen::Vector3f p(point.x,point.y,point.z);


						//now calculate the distance to the ground plane
						float dist = fabs((p.dot(plane) - 1.0f)/plane.norm());
						if(dist>detectionHeight){
							roi.at<unsigned char>(i,j)=250;
							//maybe at this point we should sort of flood fill
						}
					}

				}
			}

			//cv::imshow("poi",roi);
			cv::Mat msk(roi.rows+2,roi.cols+2,CV_8UC1);
			//something wrong with the mask;
			msk.setTo(0);
			//https://stackoverflow.com/questions/16705721/opencv-floodfill-with-mask

			//now do a second run with flood filling
			bool obstaclesDetected=false;
			cv::Mat image(mask.rows,mask.cols,CV_8UC3);
			for(int i=0;i<temp_cloud->height;i++){
				for(int j=0;j<temp_cloud->width;j++){
					if(roi.at<unsigned char>(i,j)==250){
						cv::Point seed(j,i);
						int result = cv::floodFill(roi,msk,seed,251);
						if(result > detectionPixelThreshold){

							msk.setTo(0);
							result = cv::floodFill(roi,msk,seed,255);
							obstaclesDetected = true;
							//cout << "new fill to mark obstacle " << result << endl;
						}
						//cout << result << endl;
					}

					pcl::PointXYZRGB point = temp_cloud->at(j,i);
					if(roi.at<unsigned char>(i,j)>0){
						if(roi.at<unsigned char>(i,j)==255){
							image.at<cv::Vec3b>(i,j)=cv::Vec3b(point.b,point.g,255);
						}else{
							image.at<cv::Vec3b>(i,j)=cv::Vec3b(point.b,point.g,point.r);
						}
					}else{
						image.at<cv::Vec3b>(i,j)=cv::Vec3b(point.b,point.g,point.r)*0.7f;
					}
				}
			}

			std_msgs::Bool stopMsg;
			stopMsg.data = obstaclesDetected;

			obstacleStopPub.publish(stopMsg);
			if(obstaclesDetected ){

				if(imageStoreLoaction.size()!=0){
					std::thread thread(storeObstacleImage,image.clone(),imageStoreLoaction);
					thread.detach();
				}
			}else{
			}
			 */
#ifdef DEBUG_VIEW
			cv::imshow("mask",msk);
        cv::imshow("poiAfter",roi);
        cv::imshow("image",image);
#endif

        /*
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			for(int i=0;i<temp_cloud->height;i++){
				for(int j=0;j<temp_cloud->width;j++){
					pcl::PointXYZRGB point=temp_cloud->at(j,i);
					nanPoint.r=point.r;
					nanPoint.g=point.g;
					nanPoint.b=point.b;
					if(dilatedMask.at<unsigned char>(i,j)){
						groundCloud.at(j,i)=point;
					}else{
						groundCloud.at(j,i)=nanPoint;

						//the pointcloud for obstacles is sparse.
						obstacle_cloud->points.push_back(point);
					}
				}
			}
         */
			//sensor_msgs::PointCloud2 output;






		}



		//std_msgs::Int32::ConstPtr i = m.instantiate<std_msgs::Int32>();
		/*
		if (pc != nullptr){

		}
		 */
	}

	bag.close();
	return 0;
}
