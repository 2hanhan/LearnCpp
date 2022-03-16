/**
 * @file useNet.cc
 * @author gq
 * @brief
 * @version 0.1
 * @date 2022-03-15
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Python.h>            //python API
#include <numpy/arrayobject.h> //numpy API

#define HISTO_LENGTH 30
#define TH_HIGH 100
#define TH_LOW 100
#define mfNNratio 0.6

/**
 * @brief
 * 从路径中的txt中加载image图像名称
 * @param strPath
 * @param strPathToSequence
 * @param vstrImageFilenames
 */
void LoadImagesName(const std::string strPath, const std::string &strPathToSequence, std::vector<std::string> &vstrImageFilenames)
{
    std::cout << "加载:" << strPathToSequence << std::endl;
    std::ifstream fs;
    std::string strPathTimeFile = strPathToSequence;
    fs.open(strPathTimeFile.c_str());
    while (!fs.eof())
    {
        std::string s;
        std::getline(fs, s);

        if (!s.empty())
        {
            s = strPath + s;
            std::cout << s << std::endl;
            vstrImageFilenames.push_back(s);
        }
    }
}

/**
 * @brief Get the Feature object
 * 将python网络提取的特征numpy格式转化为c++支持的的数组类型
 * @param Py_result
 * @param global_desc
 * @param keypoints
 * @param local_desc
 */
void GetFeature(PyObject *Py_result,
                cv::Mat &global_desc,
                std::vector<cv::KeyPoint> &keypoints,
                cv::Mat &local_desc)
{

    PyArrayObject *Py_global_desc, *Py_keyPoints, *Py_local_desc;
    std::vector<std::vector<double>> array1;
    std::vector<std::vector<double>> array2;
    std::vector<std::vector<double>> array3;
    std::vector<float> thisData;

    //将元组解开，如果只return 1个，则不需要
    PyArg_UnpackTuple(Py_result, "ref", 3, 3, &Py_global_desc, &Py_keyPoints, &Py_local_desc);

    // - global_desc
    /*全局描述子，大小是固定的
    npy_intp *Py_global_desc_shape = PyArray_DIMS(Py_global_desc);
    int global_desc_row = Py_global_desc_shape[0];
    int global_desc_col = Py_global_desc_shape[1];
    std::cout << "[" << global_desc_row << "]"
              << "[" << global_desc_col << "]" << std::endl;
    */
    float *pdata = global_desc.ptr<float>(0);
    for (int i = 0; i < 4096; ++i)
    {
        // std::cout << i << "[" << *(float *)PyArray_GETPTR1(Py_global_desc, i) << "]";
        // std::cout << std::endl;
        pdata[i] = *(float *)PyArray_GETPTR1(Py_global_desc, i);
    }
    // std::cout << "global_desc转化完成" << std::endl;
    //  - KeyPoints
    //  ? HF-Net返回的是int类型的坐标，但是cv::KeyPoint中构造函数是接收的是float
    npy_intp *Py_keyPoints_shape = PyArray_DIMS(Py_keyPoints);
    int keyPoints_row = Py_keyPoints_shape[0];
    // int keyPoints_col = Py_keyPoints_shape[1];//一个keypoint点有2坐标位置
    //  std::cout << "[" << keyPoints_row << "]"<< "[" << keyPoints_col << "]" << std::endl;

    for (int row = 0; row < keyPoints_row; ++row)
    {
        // std::cout << row << "[" << *(int *)PyArray_GETPTR2(Py_keyPoints, row, 0) << "," << *(int *)PyArray_GETPTR2(Py_keyPoints, row, 1) << "]";
        // std::cout << std::endl;
        double x = *(int *)PyArray_GETPTR2(Py_keyPoints, row, 0);
        double y = *(int *)PyArray_GETPTR2(Py_keyPoints, row, 1);
        cv::KeyPoint keypoint(x, y, 1);
        keypoint.octave = 0;
        keypoints.push_back(keypoint);
    }
    // std::cout << "KeyPoints转化完成" << std::endl;
    //  - local_desc
    // npy_intp *Py_local_desc_shape = PyArray_DIMS(Py_local_desc);
    local_desc.create(keyPoints_row, 256, CV_32F);
    // int local_desc_row = Py_local_desc_shape[0];
    // int local_desc_col = Py_local_desc_shape[1];
    // std::cout << "[" << local_desc_row << "]" << "[" << local_desc_col << "]" << std::endl;
    for (int i = 0; i < keyPoints_row; ++i)
    {
        float *pdata = local_desc.ptr<float>(i);
        for (int j = 0; j < 256; ++j)
        {
            // std::cout << *(float *)PyArray_GETPTR2(Py_keyPoints, i, j) << " ";
            float temp = *(float *)PyArray_GETPTR2(Py_keyPoints, i, j);
            pdata[j] = temp;
        }
        // std::cout <<std::endl;
    }
    // std::cout << "local_desc转化完成" << std::endl;
}

/**
 * @brief
 * 计算全局描述子距离
 * @param global_desc1
 * @param global_desc2
 * @return double
 */
double GlbalDescriptorDistance(cv::Mat global_desc1, cv::Mat global_desc2)
{
    cv::Mat glbDis = global_desc1 - global_desc2;
    double GlbleDistance = glbDis.dot(glbDis); // 向量内积
    return GlbleDistance;
}

/**
 * @brief
 * 计算局部描述子直接的距离
 * @param a
 * @param b
 * @return float
 */
float LocalDescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const float *pa = a.ptr<float>();
    const float *pb = b.ptr<float>();

    float dist = 0;

    for (int i = 0; i < 256; i++)
    {
        // std::cout << pa[i] << "," << pb[i] << "   ";
        dist += fabs(pa[i] - pb[i]);
    }
    // std::cout << std::coutendl;
    return dist;
}

/**
 * @brief
 * 根据keypoints和descriptors绘制图像特征匹配关系
 * @param img_1
 * @param img_2
 * @param keypoints_1
 * @param keypoints_2
 * @param descriptors_1
 * @param descriptors_2
 */
void drawmatch(cv::Mat img_1, cv::Mat img_2, std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::KeyPoint> &keypoints_2, cv::Mat &descriptors_1, cv::Mat &descriptors_2)
{
    // step 1  绘制特征点
    for (auto p : keypoints_1)
    {
        cv::circle(img_1, p.pt, 2, cv::Scalar(255, 0, 0), -1);
    }
    for (auto p : keypoints_2)
    {
        cv::circle(img_2, p.pt, 2, cv::Scalar(255, 0, 0), -1);
    }
    // std::cout << "绘制特征点" << std::endl;

    // HF-NET提取特征点
    int nmatches = 0;
    std::vector<int> vnMatches12;
    vnMatches12 = std::vector<int>(keypoints_1.size(), -1);

    // Step 1 构建旋转直方图，HISTO_LENGTH = 30
    std::vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        // 每个bin里预分配500个，因为使用的是vector不够的话可以自动扩展容量
        rotHist[i].reserve(500);

    const float factor = HISTO_LENGTH / 360.0f;
    // std::cout << "构建旋转直方图" << std::endl;

    // 匹配点对距离，注意是按照F2特征点数目分配空间
    std::vector<int> vMatchedDistance(keypoints_2.size(), INT_MAX);
    // 从帧2到帧1的反向匹配，注意是按照F2特征点数目分配空间
    std::vector<int> vnMatches21(keypoints_2.size(), -1);
    std::vector<cv::DMatch> matches;

    // 遍历帧1中的所有特征点
    for (size_t i1 = 0, iend1 = keypoints_1.size(); i1 < iend1; i1++)
    {

        cv::KeyPoint kp1 = keypoints_1[i1];
        int level1 = kp1.octave;
        // 只使用原始图像上提取的特征点
        if (level1 > 0)
            continue;
        // std::cout << "遍历帧1中的所有特征点" << std::endl;

        // 取出参考帧F1中当前遍历特征点对应的描述子
        cv::Mat d1 = descriptors_1.row(i1);
        // std::cout << "取出参考帧F1中当前遍历特征点对应的描述子" << std::endl;
        int bestDist1 = INT_MAX; //最佳描述子匹配距离，越小越好
        int bestDist2 = INT_MAX; //次佳描述子匹配距离
        int bestIdx2 = -1;       //最佳候选特征点在F2中的index

        // Step 3 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
        for (size_t i2 = 0, iend2 = keypoints_2.size(); i2 < iend2; i2++)
        {

            cv::KeyPoint kp2 = keypoints_2[i2];
            cv::Mat d2 = descriptors_2.row(i2);
            // 计算两个特征点描述子距离
            int dist = LocalDescriptorDistance(d1, d2);
            //std::cout << "[" << dist << " " << LocalDescriptorDistance(d1, d2) << "]";

            if (vMatchedDistance[i2] <= dist)
                continue;

            // 如果当前匹配距离更小，更新最佳次佳距离
            if (dist < bestDist1)
            {

                bestDist2 = bestDist1;
                bestDist1 = dist;
                bestIdx2 = i2;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if (bestDist1 <= TH_LOW)
        {
            if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
            {
                cv::DMatch temp;
                temp.queryIdx = bestIdx2; // 2描述子的索引
                temp.trainIdx = i1;       // 1描述子的索引
                matches.push_back(temp);  // ? 这个东西也没有返回啊，留着做匹配点的可视化吗
                nmatches++;
            }
        }
    }
    std::cout << "matches: " << matches.size() << std::endl; //输出匹配点的数目
                                                             // //绘制结果
    cv::Mat img_match;
    // Mat img_goodmatches;
    // BfMatch(descriptors_1, descriptors_2, matches);//调用BfMatch函数
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    // drawMatches(img_1, keypoints_1,img_2, keypoints_2,good_matches,img_goodmatches);

    cv::imshow("all matches", img_match);
    // imshow("good matches", img_goodmatches);
    // HF-Net提取特征点

    cv::waitKey(2000);
}

int main(int argc, char **argv)
{
    // step 0 加载参数、python配置等文件
    std::string data_path = "Data";
    std::string PathToquery, PathTodb;
    std::vector<std::string> querys, dbs;
    PathToquery = "Data/query.txt";
    PathTodb = "Data/db.txt";
    LoadImagesName(data_path, PathToquery, querys);
    LoadImagesName(data_path, PathTodb, dbs);
    cv::Mat image_query;
    cv::Mat image_db;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./Examples/python/')");
    PyObject *pModule, *pDict;
    pModule = PyImport_ImportModule("useNet");
    pDict = PyModule_GetDict(pModule);
    std::cout << "c++加载 model 成功" << std::endl;
    // step 1 实例化class调用
    PyObject *pClass = PyDict_GetItemString(pDict, "HFNet");
    PyObject *pConstruct = PyInstanceMethod_New(pClass); //得到class构造函数
    PyObject *cons_args = Py_BuildValue("(s)", "./Examples/python/model/hfnet");
    PyObject *pInstance = PyObject_CallObject(pConstruct, cons_args); //实例化class
    std::cout << "c++ 实例化python class 完成" << std::endl;

    PyObject *result1, *result2;
    cv::Mat global_desc;
    global_desc.create(4096, 1, CV_32F);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat local_desc;
    cv::Mat global_desc2;
    global_desc2.create(4096, 1, CV_32F);
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat local_desc2;
    //用于匹配画图的
    cv::Mat global_desc1;
    global_desc2.create(4096, 1, CV_32F);
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat local_desc1;

    // step 2 根据图像名称遍历读取
    for (int i = 0; i < querys.size(); ++i)
    {
        image_query = cv::imread(querys[i]);
        // cv::imshow("image_query", image_query);
        const char *image_name1 = querys[i].data();
        // step 3 调用网络返回结果
        result1 = PyObject_CallMethod(pInstance, "inference", "s", image_name1);
        // step 4 将返回值转化为c++格式
        keypoints.clear();
        GetFeature(result1, global_desc, keypoints, local_desc);

        int best_dbs = -1;
        double MinGlbleDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < dbs.size(); ++j)
        {
            // image_db = cv::imread(dbs[j]);
            //  cv::imshow("image_db", image_db);
            //  cv::waitKey(100);
            const char *image_name2 = dbs[j].data();
            result2 = PyObject_CallMethod(pInstance, "inference", "s", image_name2);
            keypoints2.clear();
            GetFeature(result2, global_desc2, keypoints2, local_desc2);
            double GlbleDistance = GlbalDescriptorDistance(global_desc, global_desc2);
            // std::cout << GlbleDistance << std::endl;
            if (MinGlbleDistance > GlbleDistance)
            {
                MinGlbleDistance = GlbleDistance;
                best_dbs = j;
                global_desc1 = global_desc2;
                keypoints1 = keypoints2;
                local_desc1 = local_desc2;
            }
        }
        std::cout << "image_query" << i + 1 << " <=> image_db" << best_dbs + 1 << std::endl;
        image_db = cv::imread(dbs[best_dbs]);
        drawmatch(image_query, image_query, keypoints, keypoints1, local_desc, local_desc1);
        //  cv::destroyAllWindows();
    }

    Py_Finalize();

    return 0;
}