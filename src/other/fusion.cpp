#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;

/// Global variables
RNG rng(12345);
double min_range_;
double max_range_;


// Global calibration values from Kinect (depth)
float fx_rgb = 524.8658126516821;
float fy_rgb = 526.0833409797511;
float cx_rgb = 312.2262287922412;
float cy_rgb = 255.4394087221328;
float fx_d = 595.1658098859201;
float fy_d = 596.9074027626567;
float cx_d = 310.6772546302307;
float cy_d = 247.6954910343287;


/*
height: 480
width: 640
distortion_model: plumb_bob
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
*/



// PCL Visualizer to view the pointcloud
pcl::visualization::PCLVisualizer viewer ("Simple visualizing window");
// Image to depth mask
Mat mask = Mat::zeros(480, 640, CV_8UC1);


class Converter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  image_transport::Subscriber depth_sub_;
  ros::Subscriber pcl_sub_; 
  ros::Publisher pcl_pub_;
  


public:
  Converter()
    : it_(nh_)
  {
    // Create a ROS subscriber for the input iamge
    //image_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1, &Converter::image_cb, this);
    // Create a ROS publisher for the input iamge
    //image_pub_ = it_.advertise("/my_image", 1);

    // Create a ROS subscriber for the depth iamge
    //depth_sub_ = it_.subscribe("/camera/depth_registered/sw_registered/image_rect_raw", 1, &Converter::depth_cb, this); 
 
    // Create a ROS subscriber for the input point cloud
    pcl_sub_ = nh_.subscribe ("camera/depth_registered/points", 1, &Converter::cloud_cb, this);
    // Create a ROS publisher for the output point cloud
    pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2> ("/door_handles", 1);

    // Open window
    cv::namedWindow("Image viewer");
    cv::namedWindow("Depth viewer");
      
  }

  ~Converter()
  {
    cv::destroyWindow("Image viewer");
    cv::destroyWindow("Depth viewer");
    
  }

  void image_cb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }


    // **************** Kmeans color clustering *********************

    Mat src;
    src=cv_ptr->image;
    Mat samples(src.rows * src.cols, 3, CV_32F);
    for( int y = 0; y < src.rows; y++ )
      for( int x = 0; x < src.cols; x++ )
        for( int z = 0; z < 3; z++)
          samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];


    int clusterCount = 2;
    Mat labels;
    int attempts = 3;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 3, 1), attempts, KMEANS_PP_CENTERS, centers );


    Mat clustered( src.size(), src.type() );
    for( int y = 0; y < src.rows; y++ )
      for( int x = 0; x < src.cols; x++ )
      { 
        int cluster_idx = labels.at<int>(y + x*src.rows,0);
        clustered.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
        clustered.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
        clustered.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
      }

    //***************** Contour detection + bounding box ******************
    clustered.convertTo(clustered, CV_8U);

    cv::Mat bwImage = clustered > 128;
    //cv::cvtColor(clustered, bwImage, CV_BGR2GRAY);
    //threshold( bwImage, bwImage, 128, 255,THRESH_BINARY);
    /// Reduce noise with a kernel 3x3
    //blur( bwImage, bwImage, Size(3,3) );
    Canny( bwImage, bwImage, 100, 300, 3 );
    cv::Point vertices[4];

    // Find contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( bwImage, contours, hierarchy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point (0, 0) );

    // Threshold contour size
   for (vector<vector<Point> >::iterator contours_it = contours.begin(); contours_it!=contours.end(); )
    {
      if (contours_it->size()<50 || contours_it->size()>150)
          contours_it=contours.erase(contours_it);
      else
          ++contours_it;
    }

   
    /// Approximate contours to polygons + get rotated bounding rects
    vector<RotatedRect> minRect( contours.size() );
    for( size_t i = 0; i< contours.size(); i++ )
    {
       minRect[i] = minAreaRect( Mat(contours[i]) );
       Scalar color = Scalar( 0,255,0);
       
       // rotated rectangle
       Point2f rect_points[4]; 
       minRect[i].points( rect_points );
       // Read center of rotated rect
       cv::Point2f center = minRect[i].center; // center 

       // Compute angle choosing the longer edge of the rotated rect to compute the angle
        float  angle;
        cv::Point2f edge1 = cv::Vec2f(rect_points[1].x, rect_points[1].y) - cv::Vec2f(rect_points[0].x, rect_points[0].y);
        cv::Point2f edge2 = cv::Vec2f(rect_points[2].x, rect_points[2].y) - cv::Vec2f(rect_points[1].x, rect_points[1].y);
        cv::Point2f usedEdge = edge1;
        if(cv::norm(edge2) > cv::norm(edge1))
            {usedEdge = edge2;}
        cv::Point2f reference = cv::Vec2f(1,0); // horizontal edge
        angle = 180.0f/CV_PI * acos((reference.x*usedEdge.x + reference.y*usedEdge.y) / (cv::norm(reference) *cv::norm(usedEdge)));
        if (angle >-45 && angle <45)
        //if (angle>0 || angle<0)
        {

          // draw rotated rect
          for( int j = 0; j < 4; j++ )
          {
            line( clustered, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
            vertices[j] = rect_points[j];
          }
          //Fill mask
          //cv::fillConvexPoly(mask, vertices, 4, cv::Scalar(255,255,255));

          //draw contour
          drawContours( clustered, contours, i, cv::Scalar(255,255,255), -1, 8, hierarchy, 0, Point() );
          drawContours( mask, contours, i, cv::Scalar(255,255,255), -1, 8, hierarchy, 0, Point() );
        
          // draw center
          cv::circle(clustered, center, 5, cv::Scalar(255,0,0)); 

          // Save final contours
          //final_contours[final_contours_it]=contours[i];
          //final_contours_it++;
       }	
    }

    //erode(mask, mask, Mat(), Point(-1, -1), 2, 1, 1);

    // Delete everything outside contour
    //drawContours(mask, contours, -1, Scalar(255), CV_FILLED); // CV_FILLED fills the connected components found

    // normalize so imwrite(...)/imshow(...) shows the mask correctly!
    //normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

 
    // ***********************************************************************
    


    // Update GUI Window
    cv::imshow("Image viewer", clustered);
    cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }


  // ***********************************************************************
  // ***********************************************************************
  // ***********************************************************************



void depth_cb(const sensor_msgs::ImageConstPtr& image)
{
    cv_bridge::CvImagePtr cv_depth;
    try
    {
      cv_depth = cv_bridge::toCvCopy(image, "32FC1");
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    
    Mat src, masked_src;
    src = cv_depth->image;
    
    //Run mask on depth map
    src.copyTo(masked_src, mask);

    // Delete current mask
    mask = Mat::zeros(480, 640, CV_8UC1);

    // Update GUI Window
    cv::imshow("Depth viewer", masked_src);
    cv::waitKey(3);


  // Generate Pointcloud 

  Mat depthImage;
  depthImage = masked_src;

  pcl::PointCloud<pcl::PointXYZ>::Ptr outputPointcloud (new pcl::PointCloud <pcl::PointXYZ>); 
  float rgbFocalInvertedX = 1/fx_rgb;	// 1/fx
  float rgbFocalInvertedY = 1/fy_rgb;	// 1/fy

  pcl::PointXYZ newPoint;
  for (int i=0;i<depthImage.rows;i++)
  {
     for (int j=0;j<depthImage.cols;j++)
     {
	float depthValue = depthImage.at<float>(i,j);
        if (depthValue != 0)                // if depthValue is not NaN
	{
          // Find 3D position respect to rgb frame:
	  newPoint.z = depthValue/1000;
	  newPoint.x = (j - cx_rgb) * newPoint.z * rgbFocalInvertedX;
	  newPoint.y = (i - cy_rgb) * newPoint.z * rgbFocalInvertedY;
	  outputPointcloud->push_back(newPoint);
	}
     }
  }
  
  // Manipulate PCL
   
  // Crop points out of ROI
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fin(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (outputPointcloud);//cloud_rot
  pass.setFilterFieldName ("y");
  pass.setFilterLimits (-0.5, 0.5);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*cloud_fin);
  
  // Visualize pointcloud
  viewer.addCoordinateSystem();
  viewer.addPointCloud (cloud_fin, "scene_cloud");
  viewer.spinOnce();
  viewer.removePointCloud("scene_cloud");

  // Publish the data
  cloud_fin->header.frame_id = "camera_rgb_optical_frame";
  pcl_pub_.publish (cloud_fin);


}
 

  // ***********************************************************************
  // ***********************************************************************
  // ***********************************************************************


  void 
  cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    // Container for original & filtered data
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2; 
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    pcl::PCLPointCloud2 cloud_filtered;

    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud);

    // Perform the actual filtering (not used for now)
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud (cloudPtr);
    sor.setLeafSize (0.1, 0.1, 0.1);
    sor.filter (cloud_filtered);
   
    
    // Convert to XYZRGB format
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*cloudPtr, *cloud_rgb);

    // Convert to XYZ format
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(*cloudPtr, *cloud_xyz);



    // **************** Do Something with the Pointcloud here *********************
    
/*
    // Rotate points to world_frame (?)
    Eigen::Affine3f transform_1 = Eigen::Affine3f::Identity();
    // Define a rotation of 3.28 radians around X axis. Points are given 180 degrees turned.
    transform_1.rotate (Eigen::AngleAxisf (3.28, Eigen::Vector3f::UnitX())); 
    // Define a translation of 0.45 meters on the y axis.
    transform_1.translation() << 0.0, 0.45, 0.0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rot(new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::transformPointCloud (*cloud_rgb, *cloud_rot, transform_1);
*/

    // Crop points out of ROI
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_xyz);//cloud_rot
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-0.5, 0.5);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cloud_cropped);

/*   
    // Rotate points back to camera_frame (?) -> POINTCLOUD WILL HAVE MOVED I DONT KNOW WHY
    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
    // Define a translation of 0.35 meters on the y axis.
    transform_2.translation() << 0.0, 0.45, 0.0;
    // Define a rotation of 3.28 radians around X axis. Points are given 180 degrees turned.
    transform_2.rotate (Eigen::AngleAxisf (-3.28, Eigen::Vector3f::UnitX())); 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_fin(new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::transformPointCloud (*cloud_cropped, *cloud_fin, transform_2);
*/
    
    // ****************************************************************************

/*
    // Plane segmentation
    // Create a shared plane model pointer directly
    pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloud_cropped));
    // Create the RANSAC object
    pcl::RandomSampleConsensus<pcl::PointXYZRGB> sac (model);
    // perform the segmenation step
    bool result = sac.computeModel ();
    // get inlier indices
    boost::shared_ptr<vector<int> > inliers (new vector<int>);
    sac.getInliers (*inliers);
    cout << "Found model with " << inliers->size () << "inliers";
    // get model coefficients
    Eigen::VectorXf coeff;
    sac.getModelCoefficients (coeff);
    cout << ", plane normal is: " << coeff[0] << ", " << coeff[1] << ",";
    // perform a refitting step
    Eigen::VectorXf coeff_refined;
    model->optimizeModelCoefficients(*inliers, coeff, coeff_refined);
    model->selectWithinDistance(coeff_refined, 0.03,*inliers);
    cout << "After refitting, model contains" << inliers->size () << " inliers";
    cout << ", plane normal is: " << coeff_refined[0] << ", " << coeff_refined[1] << ",  " << coeff_refined[2] << "." << endl;
    // Projection
    pcl::PointCloud<pcl::PointXYZRGB> proj_points;
    model-> projectPoints (* inliers, coeff_refined, proj_points);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(&proj_points);
*/
    // ****************************************************************************
/*
    // Normals
    pcl::PointCloud<pcl::Normal>::Ptr normals_out(new pcl::PointCloud<pcl::Normal>);
pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> norm_est;
    // Specify method for normal estimation
    norm_est.setNormalEstimationMethod (norm_est.AVERAGE_3D_GRADIENT);
    // Specify max depth change factor
    norm_est.setMaxDepthChangeFactor(0.02f);
    // Specify smoothing area size
    norm_est.setNormalSmoothingSize(10.0f);
    // Set the input points
    norm_est.setInputCloud (cloud_cropped);
    // Estimate the surface normals and
    // store the result in "normals_out"
    norm_est.compute (*normals_out);

    // Create a shared plane model pointer directly
    pcl::SampleConsensusModelNormalPlane<pcl::PointXYZRGB, pcl::Normal>::Ptr model (new pcl::SampleConsensusModelNormalPlane<pcl::PointXYZRGB, pcl::Normal> (cloud_cropped));
    // Set normals
    model->setInputNormals(normals_out);
    // Set the normal angular distance weight.
    model->setNormalDistanceWeight(0.5f);
    // Create the RANSAC object
    pcl::RandomSampleConsensus<pcl::PointXYZRGB> sac (model, 0.03);
    // perform the segmenation step
    bool result = sac.computeModel ();
    // get inlier indices
    std::vector<int> inliers;
    sac.getInliers (inliers);
    pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud_cropped, inliers, *inlier_cloud);
    // Create a Convex Hull representation of the projected inliers
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ConvexHull<pcl::PointXYZRGB> chull;
    chull.setInputCloud (*inlier_cloud);
    chull.reconstruct (*cloud_hull);
    // segment those points that are in the polygonal prism
    pcl::ExtractPolygonalPrismData<pcl::PointXYZRGB> ex;
    ex.setInputCloud (outliers);
    ex.setInputPlanarHull (cloud_hull);
    pcl::PointIndices::Ptr output (new PointIndices);
    ex.segment (*output);
*/


    // ****************************************************************************

    // Planar segmentation
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PCDWriter writer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.02);
  
    int i=0, nr_points = (int) cloud_cropped->points.size ();
  while (cloud_cropped->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_cropped);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_cropped);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_cropped = *cloud_f;
  }

   
    // Visualize pointcloud
    viewer.addCoordinateSystem();
    viewer.addPointCloud (cloud_f, "scene_cloud");
    viewer.spinOnce();
    viewer.removePointCloud("scene_cloud");

    // Convert to ROS data type
    //sensor_msgs::PointCloud2 output;
    //pcl_conversions::fromPCL(cloud_filtered, output);

    // Publish the data
    cloud_f->header.frame_id = "camera_rgb_optical_frame";
    pcl_pub_.publish (cloud_f);

  }    

};

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
    
  // Run code
  Converter ic;

  // Spin
  ros::spin ();
}
