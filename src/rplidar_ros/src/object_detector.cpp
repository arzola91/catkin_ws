#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/Int16.h"

#define RAD2DEG(x) ((x)*180./M_PI)

class SubAndPub
{
public:
	SubAndPub()
	{
		sub = n.subscribe<sensor_msgs::LaserScan>("/scan", 1000, &SubAndPub::callback, this);
		pub = n.advertise<std_msgs::Int16>("/object_flag",1);
	}

	void callback(const sensor_msgs::LaserScan::ConstPtr& scan)
	{
		int count = scan->scan_time / scan->time_increment;
    		ROS_INFO("I heard a laser scan %s[%d]:", scan->header.frame_id.c_str(), count);
    
    		int flag = 0;

         	int myints[] = {345, 350, 355,0, 5, 10, 15};
		for(int i = 0; i < 7; i++) {
       			float degree = RAD2DEG(scan->angle_min + scan->angle_increment * myints[i]);
       			ROS_INFO(": [%f, %f]", degree, scan->ranges[myints[i]]);
       			if (scan->ranges[myints[i]]<.3)  {
	  			flag = 1;
			} 
    		}	

    		ROS_INFO("flag = %d",flag);
    		
    
    		std_msgs::Int16 msg;
    		msg.data = flag;
    		pub.publish(msg);
		flag = 0;
	}

private:
	ros::NodeHandle n;
	ros::Publisher pub;
	ros::Subscriber sub;
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "SubAndPub");
    SubAndPub SAPObject;

    ros::spin();

    return 0;
}
