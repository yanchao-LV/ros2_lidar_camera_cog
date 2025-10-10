// 手动创建的 visibility_control.hpp，替代缺失的包文件
#ifndef ROSIDL_RUNTIME_CPP_VISIBILITY_CONTROL_HPP_
#define ROSIDL_RUNTIME_CPP_VISIBILITY_CONTROL_HPP_

#ifdef __cplusplus
extern "C"
{
#endif

// 定义导出/导入宏（ROS 2 消息必须的可见性控制）
#if defined _WIN32 || defined __CYGWIN__
  #ifdef ROSIDL_RUNTIME_CPP_BUILDING_DLL
    #define ROSIDL_RUNTIME_CPP_PUBLIC __declspec(dllexport)
  #else
    #define ROSIDL_RUNTIME_CPP_PUBLIC __declspec(dllimport)
  #endif
  #define ROSIDL_RUNTIME_CPP_LOCAL
#else
  #if __GNUC__ >= 4
    #define ROSIDL_RUNTIME_CPP_PUBLIC __attribute__ ((visibility ("default")))
    #define ROSIDL_RUNTIME_CPP_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define ROSIDL_RUNTIME_CPP_PUBLIC
    #define ROSIDL_RUNTIME_CPP_LOCAL
  #endif
#endif

#ifdef __cplusplus
}
#endif

#endif  // ROSIDL_RUNTIME_CPP_VISIBILITY_CONTROL_HPP_
