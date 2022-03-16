# LearnCpp
## 学习使用VSCode使用git
1. 还是现在github上创建项目把
2. 然后下载下来，需要更改的直接提交就完事了

- ? 在网页上删除了history为啥同步还不显示了呢？？
## 依赖库的安装
[OpenCV](https://github.com/2hanhan/ubuntu-and-.../blob/main/OpenCV_3-4-1.sh)
[eigen3](https://github.com/2hanhan/ubuntu-and-.../blob/main/eigen3.3.0.sh)
[Pangolin](https://github.com/stevenlovegrove/Pangolin)这个东西一般不用换版本参考官方吧

## 几个sh文件的说明
1. `build.sh` 删除重新编译所有的文件包括第三方的库
2. `11.sh` 不cmake直接编译自己的程序
3. `22.sh` 重新cmake并编译自己的程序
## 学习cpp
- 尝试一些slam的代码的lib
- lib的链接参考[CMakeLists.txt](/CMakeLists.txt)都有注释应该没问题
### 1、c++调用boost序列化存储
- [/Examples/boost/useboost.cc](/Examples/boost/useboost.cc)
- [/include/lboost.h](/include/lboost.h)
- [/src/lboost.cc](/src/lboost.cc)
### 2、c++读取yaml文件
- [/Examples/yaml/readyaml.cc](/Examples/yaml/readyaml.cc)
### 3、eigen3库的
- [/Examples/eigen/useeigen.cc](/Examples/eigen/useeigen.cc)
### 4、c++调用python
- [一些总结or吐槽](/Examples/python/python.md)
#### 基础调用
- [/Examples/python/usepython.cc](/Examples/python/usepython.cc)
- [/Examples/python/testpy.py](/Examples/python/testpy.py)
#### 调用opencv传递cv::Mat参数
- [/Examples/python/cvmat2py.cc](/Examples/python/cvmat2py.cc)
- [/Examples/python/cvmat2py.py](/Examples/python/cvmat2py.py)
