# ceres solver
## 使用流程
### step 1 构建优化问题
```c++
ceres::Problem problem;
ceres::LossFunction *loss_function;//损失核函数
loss_function = new ceres::CauchyLoss(1.0);//柯西核函数
```
### step 2 添加参与优化变量
```c++
ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);//添加参与优化的变量
//[参数1：参与优化变量的  *double valus]
//[参数2：参与优化变量的维度 int size]
//[参数3：用于重构优化参数的维数LocalParameterization*local_parameterization]
problem.SetParameterBlockConstant(para_Ex_Pose[i]); //设置固定该参数
```
### step 3 构建残差项
```c++
 ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
 problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]); //构建重投影的残差，参与优化的参数有：共同观测的2帧的相机位姿、para_Ex_Pose、相机逆深度
 //[参数1：代价函数包含了参数模块的维度信息，内部使用仿函数定义误差函数的计算方式。]
 //[参数2：损失函数：用于处理参数中的异常值，避免错误量测对估计的影响，常用参数包括HuberLoss、CauchyLoss等；该参数可以取NULL或nullptr，此时损失函数为单位函数。]
 //[参数3：参数模块：待优化的参数，可一次性传入所有参数的指针容器vector<double*>或依次传入所有参数的指针double*</double*>。]
```
#### 代价函数的构造
```c++
```
### step 4 求解优化问题
```c++
ceres::Solver::Options options;//设置求解方式
options.linear_solver_type = ceres::DENSE_SCHUR;
options.num_threads = 10;
options.trust_region_strategy_type = ceres::DOGLEG;
options.max_num_iterations = NUM_ITERATIONS;//迭代数
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);//求解
```