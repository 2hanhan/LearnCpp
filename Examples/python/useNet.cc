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

#include <Python.h>            //python API
#include <numpy/arrayobject.h> //numpy API

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

void GetFeature(PyObject *Py_result,
                cv::Mat global_desc,
                std::vector<cv::KeyPoint> &keypoints,
                cv::Mat local_desc)
{

    PyArrayObject *Py_global_desc, *Py_keyPoints, *Py_local_desc;
    std::vector<std::vector<double>> array1;
    std::vector<std::vector<double>> array2;
    std::vector<std::vector<double>> array3;
    std::vector<float> thisData;

    //将元组解开，如果只return 1个，则不需要
    PyArg_UnpackTuple(Py_result, "ref", 3, 3, &Py_global_desc, &Py_keyPoints, &Py_local_desc);
    //获取矩阵维度

    npy_intp *Py_keyPoints_shape = PyArray_DIMS(Py_keyPoints);
    npy_intp *Py_local_desc_shape = PyArray_DIMS(Py_local_desc);

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
    std::cout << "global_desc获取完成" << std::endl;
    // - KeyPoints
    // ? HF-Net返回的是int类型的坐标，但是cv::KeyPoint中构造函数是接收的是float
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
    // - local_desc
    local_desc.create(keyPoints_row, 256, CV_32F);
    int local_desc_row = Py_local_desc_shape[0];
    int local_desc_col = Py_local_desc_shape[1];

    for (int i = 0; i < keyPoints_row; ++i)
    {
        float *pdata = local_desc.ptr<float>(i);
        for (int j = 0; j < 256; j++)
        {
            float temp = *(float *)PyArray_GETPTR2(Py_keyPoints, i, j);
            if (i < 10)
            {
                std::cout << "[" << temp << "]";
            }

            pdata[j] = temp;
        }
        // std::cout << std::endl;
    }
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
    // cv::Mat image_query;
    // cv::Mat image_db;

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

    result1 = PyObject_CallMethod(pInstance, "inference", "s", "Data/db1.jpg");
    cv::Mat global_desc;
    global_desc.create(4096, 1, CV_32F);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat local_desc;

    GetFeature(result1, global_desc, keypoints, local_desc);
    /*
        // step 2 根据图像名称遍历读取
        for (int i = 0; i < querys.size(); ++i)
        {
            // image_query = cv::imread(querys[i]);
            // cv::imshow("image_query", image_query);
            const char *image_name1 = querys[i].data();
            // step 3 调用网络返回结果
            result1 = PyObject_CallMethod(pInstance, "inference", "s", image_name1);

            for (int j = 0; j < dbs.size(); ++j)
            {
                // image_db = cv::imread(dbs[j]);
                //  cv::imshow("image_db", image_db);
                //  cv::waitKey(100);
                const char *image_name2 = dbs[j].data();
                result2 = PyObject_CallMethod(pInstance, "inference", "s", image_name2);
                cv::Mat local_desc2;
                cv::Mat global_desc2;
                std::vector<cv::KeyPoint> keypoints2;
            }
            // cv::destroyAllWindows();
        }
    */
    Py_Finalize();

    return 0;
}