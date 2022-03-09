#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <math.h>

int main(int argc, char **argv)
{

    {
        Eigen::Matrix3d m = Eigen::Matrix3d::Identity(); //生成单位矩阵
        std::cout << "m^T:" << std::endl
                  << m.transpose() << std::endl; // transpose() 转置后输出

        Eigen::Vector3d v1(3, 0, 0);
        Eigen::Vector3d v2(0, 4, 0);
        Eigen::Vector3d v3(0, 0, 5);
        std::cout << "v1^T:" << std::endl
                  << v1 << std::endl
                  << "v2^T:" << std::endl
                  << v2.transpose() << std::endl
                  << "v3^T:" << std::endl
                  << v3.transpose() << std::endl;

        // m.block<1, 3>(0, 0)
        // 分块每块大小<1行，3列>
        // 取出（第0行，第0列）个块
        m.block<1, 3>(0, 0) = v1.transpose();
        m.block<1, 3>(1, 0) = v2.transpose();
        m.block<1, 3>(2, 0) = v3.transpose();

        std::cout << "m:" << std::endl
                  << m << std::endl;

        Eigen::Vector3d VecAll1;
        VecAll1.fill(1); //向量各个值全部填充为1
        std::cout << "VecAll1^T:" << std::endl
                  << VecAll1.transpose() << std::endl;

        Eigen::Vector3d normal = m.inverse() * VecAll1;
        double NormalLen = normal.norm(); //向量的模
        normal = normal / NormalLen;      //v1 v2 v3 构成平面的单位法向量
        double d_plane_o = 1 / NormalLen; //平面到原点的距离

        std::cout << "normal^T:" << std::endl
                  << normal.transpose() << std::endl;
        std::cout << d_plane_o << std::endl;
    }
    return 0;
}