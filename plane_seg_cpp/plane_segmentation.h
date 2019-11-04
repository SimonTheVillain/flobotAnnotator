#ifndef _PLANE_SEGMENTATION_
#define _PLANE_SEGMENTATION_


#include <eigen3/Eigen/Eigen>
#include <opencv2/core/core.hpp>


#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

namespace v4r{

/**
 * @brief The Plane_Segmentation class
 */
class Plane_Segmentation{
private:


    /**
     * @brief The PlaneMatrix struct
     * structure to store a symmetrical 3 by 3 matrix
     */
    struct PlaneMatrix{
        Eigen::Vector3d sum;
        double xx;
        double xy;
        double xz;
        double yy;
        double yz;
        double zz;
        int nrPoints;
        inline PlaneMatrix operator+(PlaneMatrix b){
            PlaneMatrix a;
            a.sum=sum+b.sum;
            a.xx=xx+b.xx;
            a.xy=xy+b.xy;
            a.xz=xz+b.xz;
            a.yy=yy+b.yy;
            a.yz=yz+b.yz;
            a.zz=zz+b.zz;
            a.nrPoints=nrPoints+b.nrPoints;
            return a;
        }
        inline void operator+=(PlaneMatrix b){
            sum=sum+b.sum;
            xx=xx+b.xx;
            xy=xy+b.xy;
            xz=xz+b.xz;
            yy=yy+b.yy;
            yz=yz+b.yz;
            zz=zz+b.zz;
            nrPoints=nrPoints+b.nrPoints;

        }
    };

    /**
     * @brief The PlaneSegment struct
     * plane segment
     */
    struct PlaneSegment{
        float x;
        float y;
        float z;
        int nrInliers;
    };

    //maybe supply this with a list of additional
    struct Plane{
        Eigen::Vector3f plane;
        int nrElements;
    };

    PlaneMatrix* matrices;
    Plane* planeList;
    PlaneMatrix* planeMatrices;
    cv::Mat planes;
    cv::Mat centerPoints;
    cv::Mat patchIds;
    //cv::Mat planeList;
    //Eigen::Vector3f* planes;
    int nrMatrices;

    //big todo for speed: switch to Vector4f elements
    //http://eigen.tuxfamily.org/index.php?title=FAQ#Vectorization

    //two versions of is inlier one is also regarding the normal
    template <bool doNormalTest>
    bool isInlier(Eigen::Vector4f point,Eigen::Vector4f normal,Eigen::Vector4f plane,
                  float cosThreshold, float distThreshold);

    template <bool doNormalTest>
    bool isInlier(Eigen::Vector4f point,Eigen::Vector4f normal,Eigen::Vector4f plane,float _planeNorm,
                  float cosThreshold, float distThreshold);


    bool isInPlane(Eigen::Vector4f plane1, Eigen::Vector4f plane2, Eigen::Vector4f centerPlane2,
                   float cosThreshold, float distThreshold);

    bool isParallel(Eigen::Vector4f plane1, Eigen::Vector4f plane2,
                    float cosThreshold);

    Eigen::Vector4f calcPlaneFromMatrix(PlaneMatrix mat);

    void replace(int from,int to,int maxIndex);



    template <bool doNormalTest>
    cv::Mat getDebugImage();
    template <bool doNormalTest>
    cv::Mat getDebugImage(int channel);


    //TODO: document these:
    /**
     * @brief minAbsBlockInlier
     */
    int minAbsBlockInlier;

    /**
     * @brief colsOfPatches, rowsOfPatches
     * The dimensions of the downsampled image of patches
     */
    int colsOfPatches;
    int rowsOfPatches;



    /**
     * @brief fx,fy,cx,cy
     * The intrinsics, which is only needed when the input was a depth image.
     */
    float fx;
    float fy;
    float cx;
    float cy;
    float _fx;
    float _fy;


    /**
     * @brief cols,rows
     * Dimensions of the input image
     */
    int cols;
    int rows;
    /**
     * @brief maxId
     * The highest used id of
     */
    int maxId;


    int allocateMemory();
    void createNormalsFromPoints();
    void createPointsFromDepth();
    void createPointsNormalsFromDepth();


    template <bool doNormalTest>
    void calculatePlaneSegments();

    //TODO: this uses up too much time... get rid of it
    void rawPatchClustering();

    template <bool doNormalTest,bool reverse,bool zTest>
    void postProcessing1Direction(const int offsets[][2]);

    template <bool doNormalTest,bool zTest>
    void postProcessing();


    //TODO: improve readability with ENUM
    //0 for there has not been any input 1 for the last input was a pointcloud 2 for the last parameter was a depthmap
    int lastInput;
    cv::Mat depth;

    /**
     * @brief zBuffer
     * This buffer contains the distance of a point to the assumed plane.
     * Only used when doZTest is set to true
     */
    cv::Mat zBuffer;

    /**
     * @brief thresholdsBuffer
     * Stores the thresholds for the according patches:
     * channel1 maxBlockDistance
     * channel2 minCosBlockAngle
     * channel3 maxInlierDistance
     * channel4 minCosAngle
     */
    cv::Mat thresholdsBuffer;


    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr _sortedCloud;

public:

    //TODO: get the naming straight
    /**
     * @brief minNrPatches
     * The minimum number of blocks that are allowed to spawn a plane
     */
    int minNrPatches=5;
    /**
     * @brief patchDim
     * Patches are made of pixel squares that have exactly these side length
     */
    int patchDim=10;

    /**
     * @brief minBlockInlierRatio
     * The minimum ratio of points that have to be in a patch before it would get discarded.
     */
    float minBlockInlierRatio=0.95;

    /**
     * @brief maxBlockAngle
     * The maximum angle that is allowed between two adjacent blocks to be able to connect them
     */
    float maxBlockAngle;
    /**
     * @brief minCosBlockAngle
     * The cos of this block angle
     */
    float minCosBlockAngle;

    /**
     * @brief minCosBlockAngleFunc
     */
    float (*minCosBlockAngleFunc)(float depth) = NULL;

    /**
     * @brief maxAngle
     * The angle the normal vector of a pointer is never allowed to be off of a patch befor getting discarded.
     */
    float maxAngle;

    /**
     * @brief minCosAngle
     * The cos of this angle
     */
    float minCosAngle;


    /**
     * @brief minCosAngleFunc
     */
    float (*minCosAngleFunc)(float depth) = NULL;

    /**
     * @brief useVariableThresholds
     */
    bool useVariableThresholds=true;

    /**
     * @brief maxStepSize
     * To calculate normals there is a maximum step size which allowes to correctly distinguish between
     * 2 adjacent plane segments.
     */
    float maxStepSize=0.05f;

    /**
     * @brief maxInlierDist
     * The maximum distance a point is allowed to be out of his plane
     */
    float maxInlierDist=0.01f;


    /**
     * @brief maxInlierDistFunc
     *
     */
    float (*maxInlierDistFunc)(float depth) = NULL;

    /**
     * @brief maxInlierBlockDist
     * The maximum distance two adjacent patches are allowed to be out of plane
     */
    float maxInlierBlockDist=0.005f;


    /**
     * @brief inlierBlockDistanceFunc
     */
    float (*maxInlierBlockDistFunc)(float depth) = NULL;

    /**
     * @brief maxDistance
     * A maximum distance at which no plane segmentation is possible anymore
     */
    float maxDistance=4.0f;//4 meter cutoff distance would be great

    /**
     * @brief pointwiseNormalCheck
     * Activating this allowes to reduce a lot of calculations and improve speed by a lot
     */
    bool pointwiseNormalCheck=false;

    //just in case that
    /**
     * @brief recalculateNormals
     * Just in case that new points are added which contain normals that are not trustworthy this allowes us to recalculate
     * normals.
     */
    bool recalculateNormals;


    /**
     * @brief doZTest
     * Only the closest possible points get added to a plane
     */
    bool doZTest=true;


    //Some parameters for maximum
    cv::Mat segmentation;
    std::vector<Plane> resultingPlanes;

    //two structures that are mainly used internally:
    //maybe combine them to one structure for better memory access)
    cv::Mat points;
    cv::Mat normals;

    cv::Mat debug;


    Plane_Segmentation();
    ~Plane_Segmentation();

    void setIntrinsics(float fx,float fy, float cx,float cy);
    void setInput(cv::Mat depth);
    void setInput(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sortedCloud);
    void setInput(pcl::PointCloud<pcl::PointXYZRGB>::Ptr sortedCloud);

    void setMaxAngle(float angle);
    void setMaxBlockAngle(float angle);


    int segment();

    void dubiousLinewisePostProcessing();



    /**
     * @brief generateDebugTextureForPlane
     *
     * @return
     */
    template <bool doNormalTest>
    cv::Mat generateDebugTextureForPlane(Eigen::Vector4f plane, int index);

    /**
     * @brief generateColorCodedTexture
     * @return
     * a rgb image of the segmentation result
     */
    cv::Mat generateColorCodedTexture();

    /**
     * @brief generateColorCodedTextureDebug
     * @return
     */
    cv::Mat generateColorCodedTextureDebug();




};
}
#endif
