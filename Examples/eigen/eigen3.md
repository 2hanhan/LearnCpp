[TOC]
# eigen3lib的使用

# 向量
```c++
Eigen::Vector3f u;
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
Eigen::Matrix3d A;
```
## 矩阵赋值
```c++
A<<1,0,0,
    0,1,0,
    0,0,1;
```
## 矩阵块操作
### 取行
```c++
```
### 取列
```c++
Eigen::Vector3f n1,n2,n0;
A.col(0) = n2; //第0列
A.col(1) = n1; //第1列
A.col(2) = n0; //第2列
```
## 矩阵分解
分解矩阵A
### Cholesky分解 
将A分解成 下三角矩阵L*L^T
```c++
Eigen::Matrix3d L,LT;
L = Eigen::LLT<Eigen::Matrix3d>(cov).matrixL();
LT = Eigen::LLT<Eigen::Matrix3d>(cov).matrixL().transpose();
```

# 坐标变换
## 坐标轴
```C++
Eigen::Vector3f::UnitX();  //x轴单位向量
Eigen::Vector3f::UnitY();  //x轴单位向量
Eigen::Vector3f::UnitZ();  //x轴单位向量
```
## 旋转
```c++
double x,y,z,w;
Eigen::Quaterniond Qm(w,x,y,z);   
```