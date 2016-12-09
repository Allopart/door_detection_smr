#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
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


// PCL Visualizer to view the pointcloud
// pcl::visualization::PCLVisualizer viewer ("Door handle RGBD_method ");
// Image to depth mask
Mat mask = Mat::zeros(480, 640, CV_8UC1);
cv::Point2f center;
cv::Point2f rect_points[4];

class Converter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  //image_transport::Publisher image_pub_;
  image_transport::Subscriber depth_sub_;
  image_transport::Subscriber mask_sub_;
  ros::Publisher pcl_pub_;
  ros::Publisher turn_pub_;
  


public:
  Converter()
    : it_(nh_)
  {


    // Open window
    cv::namedWindow("Image viewer");
    //cv::namedWindow("Depth viewer");
    cv::namedWindow("Mask viewer");


    // Create a ROS subscriber for the input iamge
    image_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1, &Converter::image_cb, this); // /cropped/image_rect 
    // Create a ROS publisher for the input image
    //image_pub_ = it_.advertise("/my_image", 1);

    // Create a ROS subscriber for the depth image
    depth_sub_ = it_.subscribe("/cropped/depth_rect", 1, &Converter::depth_cb, this); // /camera/depth_registered/sw_registered/image_rect_raw

    // Create a ROS subscriber for the input image
    mask_sub_ = it_.subscribe("/cropped/mask", 1, &Converter::mask_cb, this);
 
    // Create a ROS publisher for the output point cloud
    pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2> ("/door_detection/handle_rgbd", 1);
    
    // Create a ROS publisher for the handle turn
    turn_pub_ = nh_.advertise<std_msgs::String> ("/door_detection/handle_turn", 10);
  

  }

/*
  ~Converter()
  {
    cv::destroyWindow("Image viewer");
    cv::destroyWindow("Depth viewer");
    
  }
*/

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

/*   int x_max_roi = 0;
   int x_min_roi = src.cols;
   int y_max_roi = 0;
   int y_min_roi = src.rows;
    
   for( int y = 0; y < src.rows; y++ )
   {
     for( int x = 0; x < src.cols; x++ )
     {
        if (src.at<Vec3b>(y,x)[0] != 0 && src.at<Vec3b>(y,x)[1] != 0 && src.at<Vec3b>(y,x)[2] != 0)
        {
 	   if (x>x_max_roi) x_max_roi=x;
           if (y>y_max_roi) y_max_roi=y;
           if (x<x_min_roi) x_min_roi=x;
           if (y<y_min_roi) y_min_roi=y;
        }
     }
   }

   int width = x_max_roi - x_min_roi;
   int height = y_max_roi - y_min_roi; 
*/

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

    // Approximate contours to polygons + get rotated bounding rects
    vector<RotatedRect> minRect( contours.size() );
    for( size_t i = 0; i< contours.size(); i++ )
    {
       minRect[i] = minAreaRect( Mat(contours[i]) );
       Scalar color = Scalar( 0,255,0);
       
       // Refine rotated rectangle
       Point2f rect_points[4]; 
       minRect[i].points( rect_points );

       // Read center of rotated rect
       center = minRect[i].center; // center 

       // Compute angle choosing the longer edge of the rotated rect to compute the angle
        float  angle;
        cv::Point2f edge1 = cv::Vec2f(rect_points[1].x, rect_points[1].y) - cv::Vec2f(rect_points[0].x, rect_points[0].y);
        cv::Point2f edge2 = cv::Vec2f(rect_points[2].x, rect_points[2].y) - cv::Vec2f(rect_points[1].x, rect_points[1].y);
        cv::Point2f usedEdge = edge1;
        if(cv::norm(edge2) > cv::norm(edge1))
            {usedEdge = edge2;}
        cv::Point2f reference = cv::Vec2f(1,0); // horizontal edge
        angle = 180.0f/CV_PI * acos((reference.x*usedEdge.x + reference.y*usedEdge.y) / (cv::norm(reference) *cv::norm(usedEdge)));

        // Delete those contours that are not straight
        if (angle >-50 && angle <50)
        {
          // Draw rotated rect
          for( int j = 0; j < 4; j++ )
          {
            line( clustered, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
            vertices[j] = rect_points[j];
          }
          
          // Draw contour
          drawContours( clustered, contours, i, cv::Scalar(255,255,255), -1, 8, hierarchy, 0, Point() );
          drawContours( mask, contours, i, cv::Scalar(255,255,255), -1, 8, hierarchy, 0, Point() );
        
          // Draw center
          cv::circle(clustered, center, 15, cv::Scalar(255,0,0)); 
       }	
    }


    // Update GUI Window
    cv::imshow("Image viewer", clustered);
    cv::waitKey(1);

/*    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
*/  
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

    Mat depthImage;  
    depthImage = masked_src;

    // Generate Pointcloud 
 
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
  
/*  
    // Visualize pointcloud
    viewer.addCoordinateSystem();
    viewer.addPointCloud (cloud_fin, "scene_cloud");
    viewer.spinOnce();
    viewer.removePointCloud("scene_cloud");
*/

    // Publish the data
    cloud_fin->header.frame_id = "camera_rgb_optical_frame";
    ros::Time time_st = ros::Time::now ();
    cloud_fin->header.stamp = time_st.toNSec()/1e3;
    pcl_pub_.publish (cloud_fin);


  }
 
  // ***********************************************************************
  // ***********************************************************************
  // ***********************************************************************


  void mask_cb(const sensor_msgs::ImageConstPtr& image)
  {
    cv_bridge::CvImagePtr cv_mask;
    try
    {
      cv_mask = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
   
    Mat src;
    src = cv_mask->image;
             
    std_msgs::String str;
    bool done = false;
    int i = 1;
    
    if (src.at<uchar>(center.y, center.x )==255) //Check center is inside masked image
    {
      while (done==false)
      {
	Scalar left = src.at<uchar>(center.y, center.x - i);
        Scalar right = src.at<uchar>(center.y, center.x + i);
        if (left.val[0]==0 ) //&& left.val[1] ==0 && left.val[2]==0)
	{
          str.data= "Clockwise"; 
          turn_pub_.publish(str);
	  done=true;
	}else if(right.val[0]==0)// && right.val[1] ==0 && right.val[2]==0)
	{
          str.data= "Anticlockwise";
          turn_pub_.publish(str);  
          done=true;
	}
	i++;   
      }
    }
  
    // Draw center
    cv::circle(src, center, 15, cv::Scalar(125));

    // Draw rotated rect
    for( int j = 0; j < 4; j++ )
    {
      line(src, rect_points[j], rect_points[(j+1)%4], Scalar( 0,255,0), 1, 8 );
    }

    // Update GUI Window
    cv::imshow("Mask viewer", src);
    cv::waitKey(1);

  }

};

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "RGBD_method");
    
  // Run code
  Converter ic;

  // Spin
  ros::spin ();
}
