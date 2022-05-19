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
Eigen::Matrix3d rotation_matrix =Eigen::Matrix3d::Identity();

// 旋转矩阵 => 旋转四元数
Eigen::Quaterniond quaternion(rotation_matrix);

Eigen::Quaterniond quaternion;
quaternion = rotation_matrix;

// 旋转矩阵 => 欧拉角(Z - Y - X，即Row、Pitch、Yaw)
Eigen::Vector3d eulerAngle = rotation_matrix.eulerAngles(2, 1, 0);

// 旋转矩阵 => 旋转向量
Eigen::AngleAxisd rotation_vector(rotation_matrix);

Eigen::AngleAxisd rotation_vector;
rotation_vector = rotation_matrix;

Eigen::AngleAxisd rotation_vector;
rotation_vector.fromRotationMatrix(rotation_matrix);
```
### 旋转四元数
```c++
double x,y,z,w;
Eigen::Quaterniond quaternion(w, x, y, z);

//  旋转四元数 => 旋转矩阵
Eigen::Matrix3d rotation_matrix;
rotation_matrix = quaternion.matrix();

Eigen::Matrix3d rotation_matrix;
rotation_matrix = quaternion.toRotationMatrix();

//  旋转四元数 => 欧拉角(Z - Y - X，即Row、Pitch、Yaw)
Eigen::Vector3d eulerAngle = quaternion.matrix().eulerAngles(2, 1, 0);

//  旋转四元数 => 旋转向量
Eigen::AngleAxisd rotation_vector(quaternion);

Eigen::AngleAxisd rotation_vector;
rotation_vector = quaternion;
```
### 欧拉角
```c++
Eigen::Vector3d eulerAngle(yaw, pitch, roll);

// 欧拉角 => 四元数
Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitX()));
Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitZ()));
Eigen::Quaterniond quaternion;
quaternion = yawAngle * pitchAngle * rollAngle;

// 欧拉角 => 旋转矩阵
Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitX()));
Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitZ()));
Eigen::Matrix3d rotation_matrix;
rotation_matrix = yawAngle * pitchAngle * rollAngle;


// 欧拉角 => 旋转向量
Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitX()));
Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitZ()));
Eigen::AngleAxisd rotation_vector;
rotation_vector = yawAngle * pitchAngle * rollAngle;
```
### 旋转向量
```c++
Eigen::AngleAxisd rotation_vector(alpha, Vector3d(x, y, z));

// 旋转向量 => 旋转矩阵
Eigen::Matrix3d rotation_matrix;
rotation_matrix = rotation_vector.matrix();

Eigen::Matrix3d rotation_matrix;
rotation_matrix = rotation_vector.toRotationMatrix();

// 旋转向量 => 四元数
Eigen::Quaterniond quaternion(rotation_vector);

Eigen::Quaterniond quaternion;
Quaterniond quaternion;

Eigen::Quaterniond quaternion;
quaternion = rotation_vector;

// 旋转向量 => 欧拉角(Z - Y - X，即Row、Pitch、Yaw)
Eigen::Vector3d eulerAngle = rotation_vector.matrix().eulerAngles(2, 1, 0);
```

# 数据类型转化
##  double数字转化为矩阵
```c++
double nums[1][9];
Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> A(nums[0]);
```