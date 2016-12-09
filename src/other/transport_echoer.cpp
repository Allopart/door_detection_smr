#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ros::NodeHandle nh;
  image_transport::ImageTransport itt(nh);
  image_transport::Publisher pub = itt.advertise("my_image", 1);
  pub.publish(msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  cv::namedWindow("view");
  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/rgb/image_color", 1, imageCallback);
  ros::spin();
  cv::destroyWindow("view");
}
