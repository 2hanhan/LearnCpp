# g2o优化

## 一般g2o使用步骤
### step 1 初始化g2o优化器
```c++
//构造求解器
g2o::SparseOptimizer optimizer;
//使用线性方程求解器
g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
//6*3参数
g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//设置求解下降方式
g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//设置算法
optimizer.setAlgorithm(solver);
```
### step 2 添加顶点
```c++
int i = 0;
//**顶点** SE3:6参数
g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
//使用g2o::SE3Quat()构造顶点
vSE3->setEstimate(g2o::SE3Quat());
//设置顶点编号
vSE3->setId(i); 
// 只有第0帧关键帧不优化（参考基准）
vSE3->setFixed(i==0);
// 向优化器中添加顶点
optimizer.addVertex(vSE3);
int j = 0;
//**顶点** Point:3参数
g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
//使用Eigen::Matrix<double,3,1>()构造顶点
vPoint->setEstimate(Eigen::Matrix<double,3,1>());
//设置顶点编号
vPoint->setId(j);
//设置边缘化
vPoint->setMarginalized(true);
//向优化器中添加顶点
optimizer.addVertex(vPoint);
```
### step 3 添加边
```c++
int n = 0;
// 创建边
g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
//一个边的两个头一个连接SE3、一个连接Point
edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(j)));
edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i)));
//设置观测
edge->setMeasurement(Eigen::Matrix<double,2,1>());
//设置信息矩阵
edge->setInformation(Eigen::Matrix2d::Identity());
//设置鲁棒核函数
g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    edge->setRobustKernel(rk);
rk->setDelta(thHuber2D);
//添加边
optimizer.addEdge(edge);
```
### step 4 开始优化
```c++
optimizer.setVerbose(true);//true会输出一些额外的信息
optimizer.initializeOptimization();
optimizer.optimize(nIterations);
```
### step 5 取出优化结果
```c++
// 获取到优化后的位姿SE3
g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
g2o::SE3Quat SE3quat = vSE3->estimate();
// 获取优化后点Point
g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(j));
```
## 各个参数的含义作用
### 核函数
#### 作用
1. 防止误差增长过大占得权重比较大，但是这个错误可能是个误匹配，为了优化这个边可能消耗过大算力，并且把整体带偏了。
#### 可选参数






















