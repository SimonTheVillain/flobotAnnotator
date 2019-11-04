#ifndef _VISUAL_DIRT_DETECTION_
#define _VISUAL_DIRT_DETECTION_


#include <eigen3/Eigen/Eigen>
#include <opencv2/core/core.hpp>


namespace v4r{

/**
 * @brief The Visual_Dirt_Detection class
 *
 */
class Visual_Dirt_Detection{
private:
    //have matrices stored so  we don't need to initialize them every frame
    cv::Mat Lab;
    cv::Mat grad;
    cv::Mat gradL;
    cv::Mat gradA;
    cv::Mat gradB;


public:
    /**
     * @brief mask
     * Mask of bytes.255 means the pixel is valid and should be tested.
     * 0 means that the pixel is invalid.
     */
    cv::Mat mask;

    /**
     * @brief image
     * The image from which the dirt gets computed
     */
    cv::Mat image;

    /**
     * @brief window_size
     * The size of windows in
     */
    int window_size_x=16;
    int window_size_y=16;

    /**
     * @brief window_step_size
     * The step size between the windows can be smaller than the window size itself
     * This allows windows to overlap.
     */
    int window_step_size_x=8;
    int window_step_size_y=8;

    /**
     * @brief inlier_ratio
     * Only windows, that have enough valid points (given by mask) will get added to the statistical model.
     */
    float inlier_ratio=0.97f;

    /**
     * @brief weight_L
     * How much each of the Lab color channels weighs for the
     */
    float weight_L=0.0f;
    float weight_a=1.5f;
    float weight_b=1.5f;

    /**
     * @brief prob_threshold
     * The threshold above which the log likleyhood is considered dirt
     */
    float prob_threshold = 42.0f; //isn't everything here a bunch of magic numbers?


    /**
     * @brief sigma
     * To allow statistical features to overlap from one window to the next we smooth the absolute gradient values.
     */
    float sigma=3.0f;

    /**
     * @brief k
     * The sampled data will be statistically fitted with a mixture (sum) of k individual gauss curves.
     * Increasing k increases the cost dramatically. Additionally we might fit dirt patterns into our floor model.
     */
    int k=1;



    void compute_dirt();


    cv::Mat compute_dirt(cv::Mat mask,cv::Mat image);


    cv::Mat dirt;

    cv::Mat prob;
    cv::Mat probLL;
};

}
#endif
