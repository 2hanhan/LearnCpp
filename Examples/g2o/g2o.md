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
edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(j)));//point
edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i)));//SE3
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
optimizer.setVerbose(true);//true会输出一些额外调试的信息
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
### 优化器
1. 稀疏优化器
```c++
g2o::SparseOptimizer optimizer;
```
### 求解器
#### 线性求解器
1. camera的`SE3`6自由度位姿,point的位置`x,y,z`3自由度
```c++
g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
```
2. camera的`Sim3`6自由度位姿+1尺度，point的位置`x,y,z`3自由度
```c++
g2o::BlockSolver_7_3::LinearSolverType *linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
```
### 待优化参数设置
1. camera的`SE3`6自由度位姿，point的位置`x,y,z`3自由度
```c++
g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
```
2. camera的`Sim3`6自由度位姿+1尺度，point的位置`x,y,z`3自由度
```c++
g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
```
### 迭代下降算法
1. LM下降算法
```c++
g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
```
2. GaussNewton下降算法
```c++
g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
```
### 顶点设置
1. camera的`SE3`6自由度位姿
```c++
g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
```
2. camera的`Sim3`6自由度位姿+1尺度
```c++
g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
// 可以设置固定尺度
vSim3->_fix_scale = true;
```
添加顶点
```c++
vPoint->setEstimate(/*设置顶点的数值*/);
vPoint->setId(/*顶点id*/);
vPoint->setMarginalized(true);
optimizer.addVertex(vPoint);//添加顶点
```

### 边设置
1. 连接`SE3`与`x,y,z`的边
```c++
g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));//point
edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));//SE3
```
2. 连接`Sim3`和`Sim3`的边
```
g2o::EdgeSim3 *edge = new g2o::EdgeSim3();
edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF1->mnId)));
edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF2->mnId)));
```

### 核函数
#### 作用
1. 防止误差增长过大占得权重比较大，但是这个错误可能是个误匹配，为了优化这个边可能消耗过大算力，并且把整体带偏了。
#### 可选参数

## 自定义误差Edge

### 1元边
定义1元边
```c++
class EdgeSE3
    : public g2o::BaseUnaryEdge<3 /*误差项维度*/,
                                CLASS_A /*观测量数据类型,可以是自定义数据类型*/,
                                g2o::VertexSBAPointXYZ /*i顶点类型*/>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3PrioriMapGICP();

    virtual bool read(std::istream &) { return false; }
    virtual bool write(std::ostream &) const { return false; }

    void computeError()
    {
        _error =measurement()*111;//误差大小 measurement()等同于后边设置的观测的那个变量
    }

    void linearizeOplus()
    {
         _jacobianOplusXi = ; //对i顶点的导数
    }
};
```
设置Edge的连接
```c++
EdgeSE3 *e = new ORB_SLAM3::EdgeSE3();
e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex())); //设置i顶点
e->setMeasurement(/*观测量数据类型的变量*/);                                        //设置观测
```


### 2元边
定义2元边
```c++
class EdgeSE3ProjectXYZ: public g2o::BaseBinaryEdge<2 /*误差项维度*/,
                                 Eigen::Vector2d /*观测量数据类型*/,
                                 g2o::VertexSBAPointXYZ /*i顶点类型*/,
                                 g2o::VertexSE3Expmap /*j顶点类型*/>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3ProjectXYZ();
    bool read(std::istream &is);
    bool write(std::ostream &os) const;
    void computeError()
    {

        _error = ; //误差大小
    }

    void linearizeOplus()
    {

        _jacobianOplusXi = ; //对i顶点的导数

        _jacobianOplusXj = ; //对j顶点的导数
    }
}
```
设置Edge的连接
```c++
EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex())); //设置i顶点
e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex())); //设置j顶点
e->setMeasurement(/*观测量数据类型的变量*/);                                        //设置观测
```



















