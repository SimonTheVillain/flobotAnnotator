#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>//for the optical flow

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "visual_dirt_detection.h"

using namespace std;
namespace po = boost::program_options;

std::vector<std::string> get_file_list(const std::string& path)
{
	std::vector<std::string> m_file_list;
	if (!path.empty())
	{
		namespace fs = boost::filesystem;

		fs::path apk_path(path);
		fs::recursive_directory_iterator end;

		for (fs::recursive_directory_iterator i(apk_path); i != end; ++i)
		{
			const fs::path cp = (*i);
			string s = cp.string();
			s = s.substr(s.find_last_of("/") + 1);
			m_file_list.push_back(s);
		}
	}
	return m_file_list;
}


int main(int argc, char *argv[])
{

	string in_folder;
	string mask_folder;
	string out_folder;
	bool generate_mask = false;
	bool numbered_files = false;
	po::options_description desc("Allowed options");
	desc.add_options()
			("help", "produce help message")
			("in_folder", po::value(&in_folder), "Folder of input RGB levels")
			("mask_folder", po::value(&mask_folder), "Folder for masks (non floor pixel can be discarded)-")
			("out_folder", po::value(&out_folder), "resulting probabilities.")
			("generate_mask", po::value(&generate_mask), "assume black pixel are invalid.")
			("numbered_files", po::value(&numbered_files), "Ale files numbered (1.png 2.png) and so on?")

			;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	v4r::Visual_Dirt_Detection dirtDetection;

	vector<string> files = get_file_list(in_folder);

	int image_count = 0;
	for(int i=0;i<files.size();i++){
		string in_file = in_folder +"/" + files[i];

		if(!boost::filesystem::exists(in_file)){
			break;
		}
		cv::Mat rgb = cv::imread(in_file);
		cv::Mat mask;
		if(generate_mask){
			mask = cv::Mat(rgb.rows,rgb.cols,CV_8UC1);
			mask = 255;
			for(int i=0;i<rgb.rows;i++){
				for(int j=0;j<rgb.cols;j++){
					if(rgb.at<cv::Vec3b>(i,j)[0] < 1 &&
						rgb.at<cv::Vec3b>(i,j)[1] < 1 &&
						rgb.at<cv::Vec3b>(i,j)[2] < 1){
						mask.at<unsigned char>(i,j) = 0;
					}
				}
			}
		}else{
			mask = cv::imread(mask_folder  + files[i]);
		}
		cv::imshow("rgb",rgb);
		cv::imshow("mask",mask);
		//cv::waitKey(1);
		dirtDetection.compute_dirt(mask,rgb);

		cv::Mat expLL;
		cv::Mat expLL16;
		//cv::exp(-dirtDetection.prob,expLL);
		expLL = dirtDetection.prob;//*100
		expLL.convertTo(expLL16,CV_16UC1);//,255*255);
		cv::imwrite(out_folder + "/"  + files[i],expLL16);
		cv::imshow("prob",expLL16);
		cv::waitKey(1);
		image_count ++;
	}



	return 0;
}
