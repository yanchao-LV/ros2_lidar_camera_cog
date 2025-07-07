// ~/ws_livox/src/hik_camera_ros/src/hik_camera_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// 海康SDK头文件
#include "MvCameraControl.h"

class HikCameraNode : public rclcpp::Node
{
public:
  HikCameraNode() : Node("hik_camera_node")
  {
    // 初始化相机
    initCamera();
    
    // 创建图像发布者
    image_pub_ = image_transport::create_publisher(this, "image_raw");
    
    // 创建相机信息管理器
    camera_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
      this, "hik_camera");
    
    // 设置相机信息（从文件加载或手动设置）
    if (camera_info_manager_->loadCameraInfo("file://$(find hik_camera_ros)/config/camera_info.yaml")) {
      RCLCPP_INFO(this->get_logger(), "Loaded camera info from YAML");
    } else {
      RCLCPP_WARN(this->get_logger(), "Using default camera info");
    }
    
    // 创建定时器，定期采集图像
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(33),  // 约30fps
      std::bind(&HikCameraNode::captureImage, this));
  }
  
  ~HikCameraNode()
  {
    // 释放相机资源
    stopCamera();
  }

private:
  void initCamera()
  {
    // 相机初始化代码（参考海康SDK文档）
    int nRet = MV_OK;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    
    // 枚举设备
    nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (nRet != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "Enum devices failed: %d", nRet);
      return;
    }
    
    // 选择第一个设备
    if (stDeviceList.nDeviceNum > 0) {
      RCLCPP_INFO(this->get_logger(), "Found %d devices", stDeviceList.nDeviceNum);
      
      // 打开设备
      nRet = MV_CC_CreateHandle(&handle_, stDeviceList.pDeviceInfo[0]);
      if (nRet != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Create handle failed: %d", nRet);
        return;
      }
      
      nRet = MV_CC_OpenDevice(handle_);
      if (nRet != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Open device failed: %d", nRet);
        MV_CC_DestroyHandle(handle_);
        return;
      }
      
      // 设置相机参数（示例：设置曝光时间）
      MV_EXPOSURE_PARAM stExposure;
      stExposure.enExposureMode = MV_EXPOSURE_MODE_TIMED;
      stExposure.nExposureTime = 10000;  // 10ms
      nRet = MV_CC_SetExposureParam(handle_, &stExposure);
      if (nRet != MV_OK) {
        RCLCPP_WARN(this->get_logger(), "Set exposure failed: %d", nRet);
      }
      
      // 开始采集
      nRet = MV_CC_StartGrabbing(handle_);
      if (nRet != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Start grabbing failed: %d", nRet);
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        return;
      }
      
      RCLCPP_INFO(this->get_logger(), "Camera initialized successfully");
      is_camera_ready_ = true;
    } else {
      RCLCPP_ERROR(this->get_logger(), "No devices found");
    }
  }
  
  void stopCamera()
  {
    if (is_camera_ready_) {
      MV_CC_StopGrabbing(handle_);
      MV_CC_CloseDevice(handle_);
      MV_CC_DestroyHandle(handle_);
      is_camera_ready_ = false;
    }
  }
  
  void captureImage()
  {
    if (!is_camera_ready_) return;
    
    // 从相机获取一帧图像
    MV_FRAME_OUT_INFO_EX stImageInfo = {0};
    unsigned char* pData = nullptr;
    unsigned int nDataSize = 0;
    
    // 获取图像数据
    int nRet = MV_CC_GetImageForBGR(handle_, &pData, &nDataSize, &stImageInfo, 1000);
    if (nRet == MV_OK) {
      // 创建OpenCV图像
      cv::Mat image(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
      
      // 转换为ROS消息
      sensor_msgs::msg::Image::SharedPtr msg =
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
      
      // 设置时间戳和帧ID
      msg->header.stamp = this->now();
      msg->header.frame_id = "hik_camera";
      
      // 获取相机信息
      sensor_msgs::msg::CameraInfo::SharedPtr camera_info =
        std::make_shared<sensor_msgs::msg::CameraInfo>(
          camera_info_manager_->getCameraInfo());
      camera_info->header = msg->header;
      
      // 发布图像和相机信息
      image_pub_.publish(msg, camera_info);
    }
  }
  
  rclcpp::TimerBase::SharedPtr timer_;
  image_transport::Publisher image_pub_;
  std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  
  void* handle_ = nullptr;
  bool is_camera_ready_ = false;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HikCameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
