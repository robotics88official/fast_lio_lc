// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/**
 * @file laserMapping.cpp
 * @author Yanliang wang (wyl410922@qq.com)
 * @brief Modification: 
 *  subscribe the keyframe and update the corresponding odometry of Fast LIO 
 * @version 0.1
 * @date 2022-03-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
  
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <map>
#include <unordered_map>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// gstam
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

// gnss
#include "GNSS_Processing.hpp"
#include "sensor_msgs/NavSatFix.h"

// save data in kitti format 
#include <sstream>
#include <fstream>
#include <iomanip>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   debug_mode = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, keyFrame_topic, keyFrame_id_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_reset, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    recontructKdTree = false;
bool    updateState = false;
int     updateFrequency = 100;

// visualize
bool visulize_map = false;

vector<vector<int>>  pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);  // T lidar to imu (imu = r * lidar + t)
M3D Lidar_R_wrt_IMU(Eye3d);  // R lidar to imu (imu = r * lidar + t)

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path, path_updated/*Publish updated status path*/;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose, msg_body_pose_updated;
std::string slam_map_frame = "";
std::string base_frame = "";
ros::Publisher pose_publisher;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

/*** Maintain keyframe mechanism ***/
// Idea: cache historical lidar frames and determine which frame is a key frame based on the subscribed seq
vector<PointCloudXYZI::Ptr> cloudKeyFrames;  // Store historical keyframe point clouds
queue< pair<uint32_t, PointCloudXYZI::Ptr> > cloudBuff;      // Cache some historical lidar frames to extract key frame point clouds
vector<uint32_t> idKeyFrames;           // ids of keyframes
queue<uint32_t> idKeyFramesBuff;         // ids of keyframes buffer
nav_msgs::Path pathKeyFrames;           // keyframes
uint32_t data_seq;                      // Data serial number
uint32_t lastKeyFramesId;               // The ID of the odometer corresponding to the latest key frame
geometry_msgs::Pose lastKeyFramesPose;  // The pose of the latest keyframe (world to imu)
vector<geometry_msgs::Pose> odoms;
/*** Maintain submap ***/
// pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeSurroundingKeyPoses(new pcl::KdTreeFLANN<pcl::PointXYZ>()); // 周围关键帧pose的kdtree
pcl::VoxelGrid<pcl::PointXYZ> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

/*back end*/
vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // Collection of corner points of all keyframes in history (downsampling)
vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // Plane point collection of all keyframes in history (downsampling)

pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());         // Historical keyframe pose (position)
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); // Historical keyframe poses
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());

pcl::PointCloud<PointTypePose>::Ptr fastlio_unoptimized_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  Store fastlio unoptimized poses
pcl::PointCloud<PointTypePose>::Ptr gnss_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  gnss trajectory

// voxel filter paprams
float odometrySurfLeafSize;
float mappingCornerLeafSize;
float mappingSurfLeafSize;

float z_tolerance;
float rotation_tolerance;

// CPU Params
int numberOfCores = 4;
double mappingProcessInterval;

/*loop clousre*/
bool startFlag = true;
bool loopClosureEnableFlag;
float loopClosureFrequency; //   Loopback detection frequency
int surroundingKeyframeSize;
float historyKeyframeSearchRadius;   // Loop detection radius kdtree search radius
float historyKeyframeSearchTimeDiff; //  interframe time threshold
int historyKeyframeSearchNum;        //   How many keyframes are used to form a submap during loopback
float historyKeyframeFitnessScore;   // icp match threshold
bool potentialLoopFlag = false;

ros::Publisher pubHistoryKeyFrames; //  发布 loop history keyframe submap
ros::Publisher pubIcpKeyFrames;
ros::Publisher pubRecentKeyFrames;
ros::Publisher pubRecentKeyFrame;
ros::Publisher pubCloudRegisteredRaw;
ros::Publisher pubLoopConstraintEdge;

bool aLoopIsClosed = false;
map<int, int> loopIndexContainer; // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
deque<std_msgs::Float64MultiArray> loopInfoVec;

nav_msgs::Path globalPath;

// Map point cloud constructed by local key frames, corresponding to kdtree, used for scan-to-map to find adjacent points
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>());

// Downsampling
pcl::VoxelGrid<PointType> downSizeFilterCorner;
// pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterICP;
// pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

float transformTobeMapped[6]; //  The pose of the current frame (under world system)

std::mutex mtx;
std::mutex mtxLoopInfo;

// Surrounding map
float surroundingkeyframeAddingDistThreshold;  //  Distance threshold to determine whether it is a key frame
float surroundingkeyframeAddingAngleThreshold; //  Angle threshold to determine whether it is a key frame
float surroundingKeyframeDensity;
float surroundingKeyframeSearchRadius;

// gtsam
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

ros::Publisher pubLaserCloudSurround;
ros::Publisher pubOptimizedGlobalMap ;           //   Publish the final optimized map

// bool    recontructKdTree = false;
int updateKdtreeCount = 0 ;        //  Updated every 100 times
bool visulize_IkdtreeMap = false;            //  visual iktree submap

// gnss
bool use_gps;
double last_timestamp_gnss = -1.0 ;
deque<nav_msgs::Odometry> gnss_buffer;
geometry_msgs::PoseStamped msg_gnss_pose;
string gnss_topic ;
bool useImuHeadingInitialization;   
bool useGpsElevation;             //  Whether to use high-level GPS optimization
float gpsCovThreshold;          //   Covariance threshold of gps direction angle and altitude difference
float poseCovThreshold;       //  pose covariance threshold from isam2

M3D Gnss_R_wrt_Lidar(Eye3d) ;         // External parameters of gnss and imu
V3D Gnss_T_wrt_Lidar(Zero3d);
bool gnss_inited = false ;                        //  Whether to complete gnss initialization
shared_ptr<GnssProcess> p_gnss(new GnssProcess());
GnssProcess gnss_data;
ros::Publisher pubGnssPath ;
nav_msgs::Path gps_path ;
vector<double>       extrinT_Gnss2Lidar(3, 0.0);
vector<double>       extrinR_Gnss2Lidar(9, 0.0);


// global map visualization radius
float globalMapVisualizationSearchRadius;
float globalMapVisualizationPoseDensity;
float globalMapVisualizationLeafSize;

// saveMap
// ros::ServiceServer srvSaveMap;
// ros::ServiceServer srvSavePose;
bool savePCD;               // Whether to save the map
string savePCDDirectory;    // save route

/**
 * Update odometer track
 */
void updatePath(const PointTypePose &pose_in)
{
    string odometryFrame = slam_map_frame;
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x =  pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z =  pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}

/**
 * Transform the point cloud cloudIn with transformIn, return the result point cloud, modify liosam, and take into account the representation of external parameters.
 */
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    
    // Note: The attitude in lio_sam is represented by euler, while the attitude angle stored in fastlio is a rotation vector. And pcl::getTransformation is not appropriate to convert euler_angle to rotation_matrix, comment
    // Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_b_lidar(state_point.offset_R_L_I  );       //  Get body2lidar external parameters
    T_b_lidar.pretranslate(state_point.offset_T_L_I);        

    Eigen::Affine3f T_w_b_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_w_b ;          //   world2body  
    T_w_b.matrix() = T_w_b_.matrix().cast<double>();
    Eigen::Isometry3d  T_w_lidar  =  T_w_b * T_b_lidar  ;           //  T_w_lidar  Transformation matrix

    Eigen::Isometry3d transCur = T_w_lidar;        

    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

/**
 * pose format transformation
 */
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

/**
 * pose format transformation
 */
gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

/**
 * Eigen format pose transformation
 */
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

/**
 * Eigen format pose transformation
 */
Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

/**
 * pose format transformation
 */
PointTypePose trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
}

/**
 * Publish thisCloud and return the msg format corresponding to thisCloud
 */
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

/**
 * The distance from the point to the origin of the coordinate system
 */
float pointDistance(PointType p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

/**
 * distance between two points
 */
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

/**
 * initialization
 */
void allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudOri.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }
}

//  eulerAngle 2 Quaterniond
Eigen::Quaterniond  EulerToQuat(float roll_, float pitch_, float yaw_)
{
    Eigen::Quaterniond q ;            //   Quaternions q and -q are equal
    Eigen::AngleAxisd roll(double(roll_), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(double(pitch_), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(double(yaw_), Eigen::Vector3d::UnitZ());
    q = yaw * pitch * roll ;
    q.normalize();
    return q ;
}

// Assign the updated pose to transformTobeMapped
void getCurPose(state_ikfom cur_state)
{
    // Euler angles have no group properties, so the result converted from SO3 or a general rotation matrix is ​​the same.
    Eigen::Vector3d eulerAngle = cur_state.rot.matrix().eulerAngles(2,1,0);        //  yaw pitch roll unit: radians
    // V3D eulerAngle  =  SO3ToEuler(cur_state.rot)/57.3 ;     //   fastlio comes with roll pitch yaw unit: degree, rotation order zyx

    // transformTobeMapped[0] = eulerAngle(0);                //  roll     When using the SO3ToEuler method, the order is rpy
    // transformTobeMapped[1] = eulerAngle(1);                //  pitch
    // transformTobeMapped[2] = eulerAngle(2);                //  yaw
    
    transformTobeMapped[0] = eulerAngle(2);                //  roll  When using the eulerAngles(2,1,0) method, the order is ypr
    transformTobeMapped[1] = eulerAngle(1);                //  pitch
    transformTobeMapped[2] = eulerAngle(0);                //  yaw
    transformTobeMapped[3] = cur_state.pos(0);          //  x
    transformTobeMapped[4] = cur_state.pos(1);          //   y
    transformTobeMapped[5] = cur_state.pos(2);          // z
}

/**
 * Calculate the pose transformation between the current frame and the previous frame. If the change is too small, it will not be set as a key frame. Otherwise, it will be set as a key frame.
 */
bool saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    // Previous frame pose
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    // Current frame pose
    Eigen::Affine3f transFinal = trans2Affine3f(transformTobeMapped);
    // Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
    //                                                     transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                    
    // pose transformation increment
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); //  Get the pose of the previous frame relative to the current frame

    // The rotation and translation amounts are both small, and the current frame is not set as a keyframe.
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        return false;
    return true;
}

/**
 * Add laser odometry factor
 */
void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        // Initialize a priori factors in the first frame
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) <<1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter   // indoor 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12    //  1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        // Variable node sets initial value
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
        // Add laser odometry factor
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); /// pre
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);                   // cur
        // Parameters: previous frame id, current frame id, pose transformation between the previous frame and the current frame (as an observation value), noise covariance
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        // Variable node sets initial value
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

/**
 * Add loop closing factor
 */
void addLoopFactor()
{
    if (loopIndexQueue.empty())
        return;

    // closed loop queue
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        // The closed loop edge corresponds to the index of the two frames
        int indexFrom = loopIndexQueue[i].first; //   cur
        int indexTo = loopIndexQueue[i].second;  //    pre
        // Pose transformation of closed loop edges
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}

/**
 * Add GPS factor
*/
void addGPSFactor()
{
    if (gnss_buffer.empty()) {
        return;
    }
    // If there are no keyframes, or the distance between the first and last keyframes is less than 5m, no GPS factor is added.
    if ((cloudKeyPoses3D->points.empty()) || (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 0.5)) {
        return;
    }
    // The pose covariance is very small, so there is no need to add GPS data for correction.
    if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold) {
        cout << "Pose cov very small\n";
        // return;
    }
    static PointType lastGPSPoint;      // latest gps data
    while (!gnss_buffer.empty())
    {
        // Delete the odometer 0.2s before the current frame
        if (gnss_buffer.front().header.stamp.toSec() < lidar_end_time - 0.05)
        {
            gnss_buffer.pop_front();
        }
        // Exit after 0.2s exceeding the current frame
        else if (gnss_buffer.front().header.stamp.toSec() > lidar_end_time + 0.05)
        {
            break;
        }
        else
        {
            nav_msgs::Odometry thisGPS = gnss_buffer.front();
            gnss_buffer.pop_front();
            // GPS noise covariance is too large to be used
            float noise_x = thisGPS.pose.covariance[0];         //  x direction covariance
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];      //   z direction covariance
            if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                continue;
            // GPS odometer location
            float gps_x = thisGPS.pose.pose.position.x;
            float gps_y = thisGPS.pose.pose.position.y;
            float gps_z = thisGPS.pose.pose.position.z;
            if (!useGpsElevation)           //  Whether to use gps altitude
            {
                gps_z = transformTobeMapped[5];
                noise_z = 0.01;
            }

            // (0,0,0) Invalid data
            if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                continue;
            // Add a GPS odometer every 5m
            PointType curGPSPoint;
            curGPSPoint.x = gps_x;
            curGPSPoint.y = gps_y;
            curGPSPoint.z = gps_z;
            if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                continue;
            else
                lastGPSPoint = curGPSPoint;
            // Add GPS factor
            gtsam::Vector Vector3(3);
            Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
            gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
            gtSAMgraph.add(gps_factor);
            aLoopIsClosed = true;
            ROS_INFO("GPS Factor Added");
            break;
        }
    }
}

void saveKeyFramesAndFactor()
{
    // Calculate the pose transformation between the current frame and the previous frame. 
    // If the change is too small, it will not be set as a key frame. 
    // Otherwise, it will be set as a key frame.
    if (saveFrame() == false)
        return;
    // Laser odometry factor (from fast-lio), the input is frame_relative pose inter-frame pose (under the body system)
    addOdomFactor();
    // GPS factors (UTM -> WGS84)
    addGPSFactor();
    // Loop closing factor (rs-loop-detect) detection based on Euclidean distance
    addLoopFactor();
    // Perform optimization
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if (aLoopIsClosed == true) // There is a loopback factor, update several times more
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }
    // After update, we need to clear the saved factor graph. Note: historical data will not be cleared, ISAM saves it.
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    // Optimization Results
    isamCurrentEstimate = isam->calculateBestEstimate();
    // Current frame pose result
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);

    // cloudKeyPoses3D adds current frame position
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    // index
    thisPose3D.intensity = cloudKeyPoses3D->size(); //  Use intensity as the index of the frame point cloud
    cloudKeyPoses3D->push_back(thisPose3D);         //  New keyframe frames are put into the queue

    // cloudKeyPoses6D adds the current frame pose
    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_end_time;
    cloudKeyPoses6D->push_back(thisPose6D);

    // pose covariance
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // ESKF Status and Variance Update
    state_ikfom state_updated = kf.get_x(); //  Get cur_pose (not corrected yet)
    Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
    Eigen::Quaterniond q = EulerToQuat(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());

    //  Update status
    state_updated.pos = pos;
    state_updated.rot =  q;
    state_point = state_updated; // Update state_point, used for state_point visualization
    // if(aLoopIsClosed == true )
    kf.change_x(state_updated);  // to cur_pose Make corrections after isam2 optimization

    // TODO: The modification of P needs to be investigated. According to yanliangwang's approach, if p is modified, it will run away.
    // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P_updated = kf.get_P(); // Get the covariance matrix of the current state estimate
    // P_updated.setIdentity();
    // P_updated(6, 6) = P_updated(7, 7) = P_updated(8, 8) = 0.00001;
    // P_updated(9, 9) = P_updated(10, 10) = P_updated(11, 11) = 0.00001;
    // P_updated(15, 15) = P_updated(16, 16) = P_updated(17, 17) = 0.0001;
    // P_updated(18, 18) = P_updated(19, 19) = P_updated(20, 20) = 0.001;
    // P_updated(21, 21) = P_updated(22, 22) = 0.00001;
    // kf.change_P(P_updated);

    // Current frame laser corner points, plane points, downsampling set
    // pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    // pcl::copyPointCloud(*feats_undistort,  *thisCornerKeyFrame);
    pcl::copyPointCloud(*feats_undistort, *thisSurfKeyFrame); // Store keyframes, point clouds without downsampling

    // Save feature point downsampling set
    // cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    updatePath(thisPose6D); //  Visualize the updated path
}

void recontructIKdTree(){
    if(recontructKdTree  &&  updateKdtreeCount >  0){
        /*** if path is too large, the rvis will crash ***/
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMapPoses(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree finds the set of keyframes adjacent to the keyframe of the latest frame
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMapPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMapPoses->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            subMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);     //  subMap pose collection
        // Downsampling
        pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyPoses;
        downSizeFilterSubMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterSubMapKeyPoses.setInputCloud(subMapKeyPoses);
        downSizeFilterSubMapKeyPoses.filter(*subMapKeyPosesDS);         //  subMap poses  downsample
        // Extract feature point clouds corresponding to local adjacent key frames
        for (int i = 0; i < (int)subMapKeyPosesDS->size(); ++i)
        {
            // Distance is too large
            if (pointDistance(subMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                    continue;
            int thisKeyInd = (int)subMapKeyPosesDS->points[i].intensity;
            // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *subMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
        }
        // downsampling, release
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(subMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*subMapKeyFramesDS);

        std::cout << "subMapKeyFramesDS sizes  =  "   << subMapKeyFramesDS->points.size()  << std::endl;
        
        ikdtree.reconstruct(subMapKeyFramesDS->points);
        updateKdtreeCount = 0;
        ROS_INFO("Reconstructed  ikdtree ");
        int featsFromMapNum = ikdtree.validnum();
        kdtree_size_st = ikdtree.size();
        std::cout << "featsFromMapNum  =  "   << featsFromMapNum   <<  "\t" << " kdtree_size_st   =  "  <<  kdtree_size_st  << std::endl;
    }
        updateKdtreeCount ++ ; 
}

/**
 * Update the poses of all variable nodes in the factor graph, that is, the poses of all historical key frames, and update the odometer trajectory.
 */
void correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // Clear odometer track
        globalPath.poses.clear();
        // Update the poses of all variable nodes in the factor graph, that is, the poses of all historical keyframes
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            // Update odometer track
            updatePath(cloudKeyPoses6D->points[i]);
        }
        // Clear the local map, reconstruct  ikdtree submap
        recontructIKdTree();
        ROS_INFO("ISMA2 Update");
        aLoopIsClosed = false;
    }
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

// Three major elements of loopback detection
// 1. Set the minimum time difference, too close is not necessary
// 2. Control the frequency of loopback to avoid frequent detection. Wait for each detection.
// 3. Recalculate the waiting time based on the current minimum distance
bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    // Current key frame
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; //  Current keyframe index
    int loopKeyPre = -1;

    // The closed-loop correspondence has already been added to the current frame and will no longer be added.
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;
    // Find the keyframe set closest to the current keyframe in historical keyframes
    std::vector<int> pointSearchIndLoop;                        //  Candidate keyframe index
    std::vector<float> pointSearchSqDisLoop;                    //  Candidate keyframe distance
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); //  Historical frame construction kdtree
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    // In the candidate keyframe set, find a frame that is far away from the current frame in time and set it as a candidate matching frame.
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;
    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    ROS_INFO("Find loop clousre frame ");
    return true;
}

/**
 * Extract the key frame feature point set of several adjacent frames before and after the key index key frame, and downsample
 */
void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
{
    // Extract the key frame feature point set of several adjacent frames before and after the key index of the key index.
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    auto surfcloud_keyframes_size = surfCloudKeyFrames.size() ;
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;

        if (keyNear < 0 || keyNear >= surfcloud_keyframes_size)
            continue;

        // *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        // Note: cloudKeyPoses6D stores T_w_b, and the point cloud is under the lidar system. When building the submap of icp, you need to convert it through the external parameter T_b_lidar, refer to the conversion of pointBodyToWorld
        *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); //  fast-lio No feature extraction is performed, and the default point cloud is surf.
    }

    if (nearKeyframes->empty())
        return;

    // Downsampling
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

void performLoopClosure()
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = slam_map_frame;

    if (cloudKeyPoses3D->points.empty() == true)
    {
        return;
    }

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // Current keyframe index, candidate closed-loop matching frame index
    int loopKeyCur;
    int loopKeyPre;
    // Find the keyframe set closest to the current keyframe in the historical keyframes, and select a frame that is far apart in time as a candidate closed-loop frame
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
    {
        return;
    }

    // extract
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); //  cue keyframe
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>()); //   history keyframe submap
    {
        // Extract the current keyframe feature point set and downsample
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0); //  Convert cur keyframe to world system
        // Extract the key frame feature point set of several adjacent frames before and after the closed-loop matching key frame, and downsample
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); //  Select historyKeyframeSearchNum keyframes to form a submap
        // If there are fewer feature points, return
        // if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        //     return;
        // Publish closed-loop matching keyframe local map
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // scan-to-map，call icp match
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    // Not converged, or the match is not good enough
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;

    std::cout << "icp  success  " << std::endl;

    // Publish the feature point cloud after the pose transformation of the current keyframe after closed-loop optimization.
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // The pose transformation between the current keyframe and the closed-loop keyframe obtained by closed-loop optimization
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // Current frame pose before closed-loop optimization
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // Current frame pose after closed-loop optimization
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); //  Get the pose of the previous frame relative to the current frame
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    // Closed-loop matching frame pose
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore() ; //  loop_clousre  noise from icp
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
    std::cout << "loopNoiseQueue   =   " << noiseScore << std::endl;

    // Add data required for loop closing factors
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    loopIndexContainer[loopKeyCur] = loopKeyPre; //   Use hash map to store loopback pairs
}

// Loopback detection thread
void loopClosureThread()
{
    if (loopClosureEnableFlag == false)
    {
        std::cout << "loopClosureEnableFlag   ==  false " << endl;
        return;
    }

    ros::Rate rate(loopClosureFrequency); //   Loopback frequency
    while (ros::ok() && startFlag)
    {
        rate.sleep();
        performLoopClosure();   //  Loopback detection
    }
}

/**
 * distance between two points
*/
float pointDistance(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - lidar_end_time) > 10.0)
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar scan end time: %lf",last_timestamp_imu, lidar_end_time);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void gnss_cbk(const sensor_msgs::NavSatFixConstPtr& msg_in)
{
    // cout << "gnss cbk\n";
    //  ROS_INFO("GNSS DATA IN ");
    double timestamp = msg_in->header.stamp.toSec();

    mtx_buffer.lock();

    // No time correction is performed
    if (timestamp < last_timestamp_gnss)
    {
        ROS_WARN("gnss loop back, clear buffer");
        gnss_buffer.clear();
    }

    last_timestamp_gnss = timestamp;

    // convert ROS NavSatFix to GeographicLib compatible GNSS message:
    gnss_data.time = msg_in->header.stamp.toSec();
    gnss_data.status = msg_in->status.status;
    gnss_data.service = msg_in->status.service;
    gnss_data.pose_cov[0] = msg_in->position_covariance[0];
    gnss_data.pose_cov[1] = msg_in->position_covariance[4];
    gnss_data.pose_cov[2] = msg_in->position_covariance[8];

    mtx_buffer.unlock();
   
    if(!gnss_inited){           //  initialization location
        gnss_data.InitOriginPosition(msg_in->latitude, msg_in->longitude, msg_in->altitude) ; 
        gnss_inited = true ;
        cout << "gnss inited\n";
    }else{                               //   loading finished
        gnss_data.UpdateXYZ(msg_in->latitude, msg_in->longitude, msg_in->altitude) ;   

        Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();
        gnss_pose(0,3) = gnss_data.local_N ;                 //    north
        gnss_pose(1,3) = gnss_data.local_E ;                 //     East
        gnss_pose(2,3) = -gnss_data.local_U ;                 //    ground

        Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar) ;
        gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
        gnss_pose  =  gnss_to_lidar  *  gnss_pose ;                    //  gnss is transferred to the lidar system, (the current Gnss_T_wrt_Lidar is just a rough initial value)

        nav_msgs::Odometry gnss_data_enu ;
        // add new message to buffer:
        gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
        gnss_data_enu.pose.pose.position.x =  gnss_pose(0,3) ;  //gnss_data.local_E ;   north
        gnss_data_enu.pose.pose.position.y =  gnss_pose(1,3) ;  //gnss_data.local_N;    East
        gnss_data_enu.pose.pose.position.z =  gnss_pose(2,3) ;  //  ground

        gnss_data_enu.pose.pose.orientation.x =  geoQuat.x ;                //  The posture of gnss is not appreciable, so the posture is only used for visualization, taken from imu
        gnss_data_enu.pose.pose.orientation.y =  geoQuat.y;
        gnss_data_enu.pose.pose.orientation.z =  geoQuat.z;
        gnss_data_enu.pose.pose.orientation.w =  geoQuat.w;

        gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0] ;
        gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1] ;
        gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2] ;

        // cout << "pushing data in gnss\n";
        gnss_buffer.push_back(gnss_data_enu);

        // visial gnss path in rviz:
        msg_gnss_pose.header.frame_id = slam_map_frame;
        msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);

        msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;  
        msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
        msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;

        gps_path.poses.push_back(msg_gnss_pose);

        //  save_gnss path
        PointTypePose thisPose6D;  
        thisPose6D.x = msg_gnss_pose.pose.position.x ;
        thisPose6D.y = msg_gnss_pose.pose.position.y ;
        thisPose6D.z = msg_gnss_pose.pose.position.z ;
        thisPose6D.intensity = 0;
        thisPose6D.roll =0;
        thisPose6D.pitch = 0;
        thisPose6D.yaw = 0;
        thisPose6D.time = lidar_end_time;
        gnss_cloudKeyPoses6D->push_back(thisPose6D);   
    }
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();         // The lidar pointer points to the oldest lidar data
        meas.lidar_beg_time = time_buffer.front(); // record earliest time

        // Update end time
        if (meas.lidar->points.size() <= 1) // time too little time is too short, not enough points
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // The end time of recording lidar is starting time + single frame scanning time
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) // The time of the last point is less than half of the single frame scanning time
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // The end time of recording lidar is starting time + single frame scanning time
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // The end time is set to the start time + the time of the last point (relative)
            // Dynamically update the average scanning time of each frame of lidar data
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec(); // Oldest IMU time
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) // Record imu data, imu time is less than current frame lidar end time
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front()); // Record the imu data in the current lidar frame to meas.imu
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void keyFrame_cbk(const nav_msgs::Path::ConstPtr &msg_keyframes){
    // Update keyframes
    pathKeyFrames = *msg_keyframes;
}

void keyFrame_id_cbk(const std_msgs::Header::ConstPtr &msg_keyframe_id){
    // First add the subscribed keyframe id to idKeyFramesBuff
    idKeyFramesBuff.push(msg_keyframe_id->seq);
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    // cout << "publish frame world\n";
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = slam_map_frame;
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    // cout << "publsih frame body\n";
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = base_frame;
    laserCloudmsg.header.seq = data_seq;
    pubLaserCloudFull_body.publish(laserCloudmsg);
    cloudBuff.push( pair<int, PointCloudXYZI::Ptr>(data_seq ,laserCloudIMUBody) );  // Cache all point clouds sent to the backend
    publish_count -= PUBFRAME_PERIOD;
}

void publish_frame_lidar(const ros::Publisher & pubLaserCloudFull_lidar)
{
    // cout << "publish frame lidar\n";
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*feats_undistort, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "lidar";
    pubLaserCloudFull_lidar.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = slam_map_frame;
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    // cout << "publish map\n";
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = slam_map_frame;
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;

}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    // cout << "publish odometry\n";
    odomAftMapped.header.frame_id = slam_map_frame;
    odomAftMapped.header.seq = data_seq;
    odomAftMapped.child_frame_id = base_frame;
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);

    // odoms[data_seq] = odomAftMapped.pose.pose;  // Save history odom
    odoms.push_back(odomAftMapped.pose.pose);

    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, slam_map_frame, base_frame ) );
}

void publish_path(const ros::Publisher pubPath)
{
    // cout << "publish path\n";
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = slam_map_frame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
        
        //  save  unoptimized pose
         V3D rot_ang(Log( state_point.rot.toRotationMatrix())); //   rotation vector
        PointTypePose thisPose6D;  
        thisPose6D.x = msg_body_pose.pose.position.x ;
        thisPose6D.y = msg_body_pose.pose.position.y ;
        thisPose6D.z = msg_body_pose.pose.position.z ;
        thisPose6D.roll = rot_ang(0) ;
        thisPose6D.pitch = rot_ang(1) ;
        thisPose6D.yaw = rot_ang(2) ;
        fastlio_unoptimized_cloudKeyPoses6D->push_back(thisPose6D);   
    }
    pose_publisher.publish(msg_body_pose);
}

void publish_path_update(const ros::Publisher pubPath)
{
    // cout << "publish path update\n";
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  Timestamp
    string odometryFrame = slam_map_frame;
    if (pubPath.getNumSubscribers() != 0)
    {
        /*** if path is too large, the rvis will crash ***/
        static int kkk = 0;
        kkk++;
        if (kkk % 10 == 0)
        {
            // path.poses.push_back(globalPath);
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
}

//  Publish gnss trajectory
void publish_gnss_path(const ros::Publisher pubPath)
{
    gps_path.header.stamp = ros::Time().fromSec(lidar_end_time);
    gps_path.header.frame_id = slam_map_frame;

    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        pubPath.publish(gps_path);
    }
}

struct pose
{
    Eigen::Vector3d  t ;
    Eigen::Matrix3d  R;
};

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
        ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

/**
 * Publish the feature point cloud of the local keyframe map
 */
void publishGlobalMap()
{
    /*** if path is too large, the rvis will crash ***/
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
    string odometryFrame = slam_map_frame;
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
        return;

    if (cloudKeyPoses3D->points.empty() == true)
        return;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    ;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kdtree finds the set of keyframes adjacent to the keyframe of the latest frame
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // Downsampling
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    // Extract feature point clouds corresponding to local adjacent key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
    {
        // Distance is too large
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
        *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
    }
    // downsampling, release
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
}

int main(int argc, char **argv)
{
    // allocateMemory();
    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }

    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    data_seq = 0;
    nh.param<bool>("recontructKdTree",recontructKdTree,false);
    nh.param<bool>("updateState",updateState,false);
    nh.param<int>("updateFrequency",updateFrequency,100);
    
    nh.param<string>("common/keyFrame_topic", keyFrame_topic,"/aft_pgo_path");
    nh.param<string>("common/keyFrame_id_topic", keyFrame_id_topic,"/key_frames_ids");
    nh.param<bool>("debug_mode_enable", debug_mode, 0);
    nh.param<bool>("pcd_save_enable", pcd_save_en, 0);

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;

    nh.param<float>("odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
    nh.param<float>("mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
    nh.param<float>("mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

    nh.param<float>("z_tolerance", z_tolerance, FLT_MAX);
    nh.param<float>("rotation_tolerance", rotation_tolerance, FLT_MAX);

    nh.param<int>("numberOfCores", numberOfCores, 2);
    nh.param<double>("mappingProcessInterval", mappingProcessInterval, 0.15);

    // save keyframes
    nh.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 20.0);
    nh.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
    nh.param<float>("surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
    nh.param<float>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

    // loop clousre
    nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false);
    nh.param<float>("loopClosureFrequency", loopClosureFrequency, 1.0);
    nh.param<int>("surroundingKeyframeSize", surroundingKeyframeSize, 50);
    nh.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
    nh.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
    nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
    nh.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

    // gnss
    nh.param<string>("gnss_topic", gnss_topic,"/mavros/global_position/global");
    nh.param<bool>("use_gps", use_gps, false);
    nh.param<vector<double>>("mapping/extrinR_Gnss2Lidar", extrinR_Gnss2Lidar, vector<double>());
    nh.param<vector<double>>("mapping/extrinT_Gnss2Lidar", extrinT_Gnss2Lidar, vector<double>());
    nh.param<bool>("useImuHeadingInitialization", useImuHeadingInitialization, false);
    nh.param<bool>("useGpsElevation", useGpsElevation, false);
    nh.param<float>("gpsCovThreshold", gpsCovThreshold, 2.0);
    nh.param<float>("poseCovThreshold", poseCovThreshold, 25.0);


    // Visualization
    nh.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
    nh.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
    nh.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

    // visual ikdtree map
    nh.param<bool>("visulize_IkdtreeMap", visulize_IkdtreeMap, false);

    // reconstruct ikdtree 
    nh.param<bool>("recontructKdTree", recontructKdTree, false);

    // savMap
    nh.param<bool>("savePCD", savePCD, false);
    nh.param<std::string>("savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

    // Get IMU topic
    std::string slam_pose_topic = "";
    ros::param::get("~imu_topic",imu_topic);
    ros::param::get("~slam_map_frame",slam_map_frame);
    ros::param::get("~base_frame",base_frame);
    ros::param::get("~slam_pose_topic",slam_pose_topic);

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    // downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    // ISAM2参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    path.header.stamp = ros::Time::now();
    path.header.frame_id = slam_map_frame;
    path_updated.header.stamp    = ros::Time::now();
    path_updated.header.frame_id = slam_map_frame;

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf)); //repeat?
    memset(res_last, -1000.0f, sizeof(res_last));

    //设置imu和lidar外参和imu参数等
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov)); // acceleration covariance
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    // Set gnss external parameters
    Gnss_T_wrt_Lidar<<VEC_FROM_ARRAY(extrinT_Gnss2Lidar);
    Gnss_R_wrt_Lidar<<MAT_FROM_ARRAY(extrinR_Gnss2Lidar);

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    // Initialization, where h_share_model defines · plane search and residual calculation
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    if (debug_mode) {
        fp = fopen(pos_log_dir.c_str(),"w");
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;
    }

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_keyframes = nh.subscribe(keyFrame_topic, 10, keyFrame_cbk);
    ros::Subscriber sub_keyframes_id = nh.subscribe(keyFrame_id_topic, 10, keyFrame_id_cbk);
    
    // ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    // ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);        //  Dense point cloud in world system
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);      //  Dense point cloud under body system
    ros::Publisher pubLaserCloudFull_lidar = nh.advertise<sensor_msgs::PointCloud2>
        ("/cloud_registered_lidar", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);         //  no used
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);                    //  no used
    ros::Publisher pubKeyFramesMap = nh.advertise<sensor_msgs::PointCloud2>
        ("/Keyframes_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>("/path_updated", 100000);                   //  isam updated path
    
    pubGnssPath = nh.advertise<nav_msgs::Path>("/gnss_path", 100000);
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/keyframe_submap", 1); // Publish the feature point cloud of the local keyframe map
    pubOptimizedGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/map_global_optimized", 1); // Publish the feature point cloud of the local keyframe map
    pose_publisher = nh.advertise<geometry_msgs::PoseStamped>(slam_pose_topic, 10);

    // loop clousre
    // Publish closed-loop matching keyframe local map
    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    // Publish the feature point cloud after the pose transformation of the current keyframe after closed-loop optimization.
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    // Publish closed-loop edges, which appear as connections between closed-loop frames in rviz
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/fast_lio_sam/mapping/loop_closure_constraints", 1);

    // gnss
    ros::Subscriber sub_gnss = nh.subscribe(gnss_topic, 200000, gnss_cbk);

    // Loopback detection thread
    std::thread loopthread(&loopClosureThread);

    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();

//         // 接收关键帧, 一直循环直到其中一个为空（理论上应该是idKeyFramesBuff先空）
//         {
//             while( !cloudBuff.empty() && !idKeyFramesBuff.empty() ){
//                 while( idKeyFramesBuff.front() > cloudBuff.front().first )
//                 {
//                     cloudBuff.pop();
//                 }
//                 // 此时idKeyFramesBuff.front() == cloudBuff.front().first
//                 assert(idKeyFramesBuff.front() == cloudBuff.front().first);
//                 idKeyFrames.push_back(idKeyFramesBuff.front());
//                 cloudKeyFrames.push_back( cloudBuff.front().second );
//                 idKeyFramesBuff.pop();
//                 cloudBuff.pop();
//             }
//             assert(pathKeyFrames.poses.size() <= cloudKeyFrames.size() );   // 有可能id发过来了，但是节点还未更新

//             // 记录最新关键帧的信息
//             if(pathKeyFrames.poses.size() >= 1){
//                 lastKeyFramesId = idKeyFrames[pathKeyFrames.poses.size() - 1];
//                 lastKeyFramesPose = pathKeyFrames.poses.back().pose;
//             }
//         }

        /// In Measure, store the current lidar data and the corresponding imu data sequence within the lidar scanning time.
        if (sync_packages(Measures))
        {
            // First frame lidar data
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time; //Record the absolute time of the first frame
                p_imu->first_lidar_time = first_lidar_time; //Record the absolute time of the first frame
                flg_first_scan = false;
                continue;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            // According to the imu data sequence and lidar data, forward propagation corrects the distortion of the point cloud. Interval sampling or feature extraction has been completed before.
            // feats_undistort is the point cloud after distortion correction, lidar system
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();                                               // State prediction value of body after forward propagation
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // global system lidar position

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            // Check whether the current lidar data time and the earliest lidar data time are enough
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment(); // According to the position of the lidar under the W system, redetermine the bounding box corner points of the local map and remove the far points.

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size(); // Number of points after downsampling the current frame

            /*** initialize the map kdtree ***/
            if (ikdtree.Root_Node == nullptr)
            {
                if (feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // point to world system
                    }
                    // Point cloud after downsampling the current frame in world system, initialize lkd-tree
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            // lidar --> imu
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            if (visulize_IkdtreeMap) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
                publish_map(pubLaserCloudMap);
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time); // Forecast, update
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // lidar coordinates in world system
            geoQuat.x = state_point.rot.coeffs()[0];                                // The posture quaternion of the current imu in the world system
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            getCurPose(state_point); //   Update transformTobeMapped

            /*back end*/
            // 1.Calculate the pose transformation between the current frame and the previous frame. If the change is too small, it will not be set as a key frame. Otherwise, it will be set as a key frame.
            // 2.Add laser odometry factor, GPS factor, closed-loop factor
            // 3.Perform factor graph optimization
            // 4.Obtain the optimized pose and pose covariance of the current frame
            // 5.Add cloudKeyPoses3D, cloudKeyPoses6D, update transformTobeMapped, and add the corner points and plane point sets of the current key frame
            saveKeyFramesAndFactor();
            // Update the poses of all variable nodes in the factor graph, that is, the poses of all historical key frames, update the odometry trajectory, and reconstruct the ikdtree
            correctPoses();
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            /******* Publish points *******/
            if (path_en){
                publish_path(pubPath);
                publish_gnss_path(pubGnssPath);                        //   Publish gnss trajectory
                publish_path_update(pubPathUpdate);             //   Publish the path optimized by isam2
                static int jjj = 0;
                jjj++;
                if (jjj % 100 == 0)
                {
                    publishGlobalMap();             //  Publish local point cloud feature map
                }
            }
            if (scan_pub_en || pcd_save_en)
                publish_frame_world(pubLaserCloudFull);        //   Publish point cloud under world system
            if (scan_pub_en && scan_body_pub_en) {
                publish_frame_body(pubLaserCloudFull_body);         //  Publish point cloud under imu system
                publish_frame_lidar(pubLaserCloudFull_lidar);
            }


            // if(savePCD)  saveMap();

            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (debug_mode)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    // if (pcl_wait_save->size() > 0 && pcd_save_en)
    cout << "going to save fileeeee!!!!!\n";
    if (pcl_wait_save->size() > 0 && true)
    {
        cout << "saving file!!!!!!!!!!!!!!\n";
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (debug_mode)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    startFlag = false;
    loopthread.join(); //  Detached thread

    return 0;
}