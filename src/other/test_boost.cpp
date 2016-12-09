#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

using namespace sensor_msgs;
using namespace message_filters;

void callback(const PointCloud2ConstPtr& image1, const PointCloud2ConstPtr& image2)
{
   std::cout<< " I am here! " <<std::endl;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vision_node");

  ros::NodeHandle nh;
  message_filters::Subscriber<PointCloud2> image1_sub(nh, "image1", 1);
  message_filters::Subscriber<PointCloud2> image2_sub(nh, "image2", 1);

  typedef sync_policies::ApproximateTime<PointCloud2, PointCloud2> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image1_sub, image2_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();

  return 0;
}
