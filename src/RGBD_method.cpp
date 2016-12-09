#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <std_msgs/String.h>
#include <boost/bind.hpp>
// PCL specific includes
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
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath> 

using namespace sensor_msgs;
using namespace message_filters;
using namespace pcl;
using namespace cv;


ros::Publisher pcl_pub;
ros::Publisher turn_pub;


/// Global variables
RNG rng(12345);
double min_range_;
double max_range_;

int mask_coeff[4];


// Global calibration values from Kinect (depth)
float fx_rgb = 524.8658126516821;
float fy_rgb = 526.0833409797511;
float cx_rgb = 312.2262287922412;
float cy_rgb = 255.4394087221328;
float fx_d = 595.1658098859201;
float fy_d = 596.9074027626567;
float cx_d = 310.6772546302307;
float cy_d = 247.6954910343287;


void callback(const sensor_msgs::ImageConstPtr& mask_sub, const sensor_msgs::ImageConstPtr& img_sub, const sensor_msgs::ImageConstPtr& depth_sub)
{

  cv_bridge::CvImagePtr mask_ptr;
  cv_bridge::CvImagePtr img_ptr;
  cv_bridge::CvImagePtr depth_ptr;

  try
  {
    mask_ptr = cv_bridge::toCvCopy(mask_sub, sensor_msgs::image_encodings::MONO8);
    img_ptr = cv_bridge::toCvCopy(img_sub, sensor_msgs::image_encodings::BGR8);
    depth_ptr = cv_bridge::toCvCopy(depth_sub, "32FC1");
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  
  // Variables used throughout the whole code
  Mat mask_handle = Mat::zeros(480, 640, CV_8UC1);
  cv::Point2f center;
  Point2f rect_points[4];


  // ***********************************************************************
  // **************************** RGB IMAGE ********************************
  // ***********************************************************************

  Mat src_img;
  src_img=img_ptr->image;

  
// **************** Kmeans color clustering *********************

   
    
    Mat samples(src_img.rows * src_img.cols, 3, CV_32F);
    for( int y = 0; y < src_img.rows; y++ )
      for( int x = 0; x < src_img.cols; x++ )
        for( int z = 0; z < 3; z++)
          samples.at<float>(y + x*src_img.rows, z) = src_img.at<Vec3b>(y,x)[z];


    int clusterCount = 2;
    Mat labels;
    int attempts = 3;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 3, 1), attempts, KMEANS_PP_CENTERS, centers );

    Mat clustered( src_img.size(), src_img.type() );
    for( int y = 0; y < src_img.rows; y++ )
      for( int x = 0; x < src_img.cols; x++ )
      { 
        int cluster_idx = labels.at<int>(y + x*src_img.rows,0);

        clustered.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
        clustered.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
        clustered.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        /*
	if (cluster_idx==1)
        {     
          bwImage.at<uchar>(y,x) = 0;
        }else
        {
          bwImage.at<uchar>(y,x) = 255;
        }
        */
      }

 

    //***************** Contour detection + bounding box ******************
    clustered.convertTo(clustered, CV_8U);

    // Extract ROI
    Mat roi, roi_img;
    roi_img = src_img( Rect(mask_coeff[0], mask_coeff[1],mask_coeff[2], mask_coeff[3]) );
    roi = clustered( Rect(mask_coeff[0], mask_coeff[1],mask_coeff[2], mask_coeff[3]) );

    // Convert Roi to binary
    cv::Mat bwImage(roi.size(), CV_8U);
    cvtColor(roi,bwImage,CV_RGB2GRAY);
    cv::threshold(bwImage, bwImage, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Save mask_handle for future use
    Mat mask_handle_save = mask_handle;

    //Create black image to add final contours too
    Mat final_contours = Mat::zeros(480, 640, CV_8UC1);

    // Fuse CNN mask with handle_mask
    bwImage.copyTo(mask_handle(Rect(mask_coeff[0], mask_coeff[1],mask_coeff[2], mask_coeff[3])));

    // Apply canny edge detector
    Canny( mask_handle, mask_handle, 100, 300, 3 );
    cv::Point vertices[4];

    // Find contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(mask_handle, contours, hierarchy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point (0, 0) );

    // Threshold contour size
    for (vector<vector<Point> >::iterator contours_it = contours.begin(); contours_it!=contours.end(); )
    {
      if (contours_it->size()<20 || contours_it->size()>180)//20 150
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
       minRect[i].points( rect_points );

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
        if (angle >-75 && angle <75)
        {

          // Read center of rotated rect
          center = minRect[i].center; // center       

          // Draw rotated rect
          for( int j = 0; j < 4; j++ )
          {
            line( clustered, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
            vertices[j] = rect_points[j];
          }
          
          // Draw contour
          drawContours( clustered, contours, i, cv::Scalar(255,255,255), -1, 8, hierarchy, 0, Point() );       
          drawContours( final_contours, contours, i, cv::Scalar(255,255,255), -1, 8, hierarchy, 0, Point() );
        
          // Draw center
          cv::circle(clustered, center, 5, cv::Scalar(255,0,0)); 
        }	
    }
  
    // Dilate slightly the final contours mask
    Mat element (5,5,CV_8U, Scalar(1));
    dilate(final_contours, final_contours, element, Point(-1, -1), 1, 1, 1);
   
    // For display purposes
    Mat initial_roi(480, 640, CV_8UC3, Scalar(0,0,0));
    roi_img.copyTo(initial_roi(Rect(mask_coeff[0], mask_coeff[1],mask_coeff[2], mask_coeff[3])));
    Mat clustered_roi(480, 640, CV_8UC3, Scalar(0,0,0));
    roi.copyTo(clustered_roi(Rect(mask_coeff[0], mask_coeff[1],mask_coeff[2], mask_coeff[3])));

    // Update GUI Window
    cv::imshow("Original image",  initial_roi);
    cv::imshow("Clustered image",  clustered_roi);
    cv::imshow("Binary mask",  final_contours);
    cv::waitKey(1);  


   // ***********************************************************************
   // ************************** DEPTH IMAGE ********************************
   // ***********************************************************************

    Mat depth_src, masked_src;
    depth_src = depth_ptr->image;
    
    //Run mask on depth map
    depth_src.copyTo(masked_src, final_contours);
    
    // Delete current mask
    mask_handle = Mat::zeros(480, 640, CV_8UC1);

    Mat depthImage;  
    depthImage = masked_src;

/*     
    // Update GUI Window
    cv::imshow("Depth viewer", masked_src);
    cv::waitKey(1);
*/

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
   
    // Crop points that are too low or too hgh to be handles
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
    pcl_pub.publish (cloud_fin);


   // ***********************************************************************
   // ***************************** TURN ************************************
   // ***********************************************************************

    Mat mask_src;
    mask_src = mask_ptr->image;
            
    std_msgs::String str;
    bool done = false;
    int i = 1;
    
    if (mask_src.at<uchar>(center.y, center.x )==255) //Check center is inside masked image
    {
      while (done==false)
      {
	Scalar left = mask_src.at<uchar>(center.y, center.x - i);
        Scalar right = mask_src.at<uchar>(center.y, center.x + i);
        if (left.val[0]==0 ) //&& left.val[1] ==0 && left.val[2]==0)
	{
          str.data= "Clockwise"; 
          turn_pub.publish(str);
	  done=true;
	}else if(right.val[0]==0)// && right.val[1] ==0 && right.val[2]==0)
	{
          str.data= "Anticlockwise";
          turn_pub.publish(str);  
          done=true;
	}
	i++;   
      }
    }
    // Draw center
    cv::circle(mask_src, center, 5, cv::Scalar(125));

    // Draw rotated rect
    for( int j = 0; j < 4; j++ )
    {
      line(mask_src, rect_points[j], rect_points[(j+1)%4], Scalar( 0,255,0), 1, 8 );
    }

/*
    // Update GUI Window
    cv::imshow("Mask viewer", mask_src);
    cv::waitKey(1);
*/

}


void mask_callback (const std_msgs::Int32MultiArray::ConstPtr& array)
{

   int i = 0;
   // print all the remaining numbers
   for(std::vector<int>::const_iterator it = array->data.begin(); it != array->data.end(); ++it)
   {
     mask_coeff[i]=*it;
     i++;
   }

return;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "RGB_Method2r");

  cv::namedWindow("Original image");
  cv::namedWindow("Clustered image");
  cv::namedWindow("Binary mask");

  
  ros::NodeHandle nh;

  ros::Subscriber mask_coeff = nh.subscribe("/dn_object_detect/mask_coeff", 1, mask_callback);

  message_filters::Subscriber<sensor_msgs::Image> mask_sub(nh, "/cropped/mask", 1);
  message_filters::Subscriber<sensor_msgs::Image> img_sub(nh, "/camera/rgb/image_rect_color", 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/cropped/depth_rect", 1);


  pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("/door_detection/handle_rgbd", 1);
  turn_pub = nh.advertise<std_msgs::String> ("/door_detection/handle_turn", 10);


  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), mask_sub, img_sub, depth_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));


  ros::spin ();
  return 0;

}
