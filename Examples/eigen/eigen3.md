[TOC]
# eigen3lib的使用

# 向量
```c++
Eigen::Vector3f u;//3行*1列 列向量
```
## 向量二元操作
```c++
Eigen::Vector3f v;
u.dot(v);//向量内积 u·v
u.cross(v)；//向量叉积 u×v
```
## 共轭
```c++
u.adjoint();//返回u的共轭向量，若u为实向量，则返回结果与u相同。
```

# 矩阵
```c++
Eigen::Matrix3d A;//3*3方阵
Eigen::Matrix<double, 3, 6> B;//3行*6列矩阵
```
## 矩阵赋值
```c++
A<<1,0,0,
    0,1,0,
    0,0,1;
A = Eigen::Matrix3d::Identity();//单位矩阵
```
## 转置
```c++
A.transpose();
```
## 矩阵块操作
### 取行
```c++
Eigen::Vector3d n0,n1,n2;
A12.row(0) = n0.transpose();//第0行
A12.row(1) = n1.transpose();//第1行
A12.row(2) = n2.transpose();//第2行
```
### 取列
```c++
Eigen::Vector3d n0,n1,n2;
A.col(0) = n0; //第0列
A.col(1) = n1; //第1列
A.col(2) = n2; //第2列
```
### 取任意大小的块
```c++
B.block<1, 3>(2, 0)=n1.transpose();//每块大小<1,3>1行*3列 位置(2,0)第2行，第0列的块
B.leftCols<3>() = Eigen::Matrix3d::Identity();//按照列从左边取<3>列
B.rightCols<3>() = Eigen::Matrix3d::Identity();//按照列右边取<3>列
```


## 矩阵分解
分解矩阵A
### Cholesky分解 
将A分解成 下三角矩阵L*L^T
```c++
Eigen::Matrix3d L,LT;
L = Eigen::LLT<Eigen::Matrix3d>(A).matrixL();
LT = Eigen::LLT<Eigen::Matrix3d>(A).matrixL().transpose();
```
原理介绍：[https://blog.csdn.net/wfei101/article/details/81951888](https://blog.csdn.net/wfei101/article/details/81951888)

# 坐标变换
## 坐标轴
```C++
Eigen::Vector3d::UnitX();  //x轴单位向量
Eigen::Vector3d::UnitY();  //x轴单位向量
Eigen::Vector3d::UnitZ();  //x轴单位向量
```
## 旋转
### 旋转矩阵
```c++
Eigen::Matrix3d R =Eigen::Matrix3d::Identity();
R =  Q.toRotationMatrix()
```
### 旋转四元数
```c++
double x,y,z,w;
Eigen::Quaterniond Q(w,x,y,z);   
Eigen::Quaterniond Q(R);   
```
### 欧拉角

# 数据类型转化
##  double数字转化为矩阵
```c++
double nums[1][9];
Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> A(nums[0]);
```