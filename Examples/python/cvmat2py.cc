/**
 * @file useNet.cc
 * @author gq
 * @brief
 * @version 0.1
 * @date 2022-03-14
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <Python.h>
#include <numpy/arrayobject.h>

/**
 * @brief
 * 使用numpy库实现格式转化
 * @param pDict
 * @return int
 */
PyObject *cvmat2py(cv::Mat &image)
{
    import_array();
    int row, col;
    col = image.cols; //列宽
    row = image.rows; //行高
    int channel = image.channels();
    int irow = row, icol = col * channel;
    npy_intp Dims[3] = {row, col, channel}; //图像维度信息
    PyObject *pyArray = PyArray_SimpleNewFromData(channel, Dims, NPY_UBYTE, image.data);
    PyObject *ArgArray = PyTuple_New(1);
    PyTuple_SetItem(ArgArray, 0, pyArray);
    return ArgArray;
}

int main()
{
    // step 1 读取图像
    cv::Mat image = cv::imread("Data/db1.jpg", CV_LOAD_IMAGE_COLOR);
    // cv::imshow("dad", image);
    // cv::waitKey(1000);
    std::cout << "读取完毕" << std::endl;
    // step 2 引入modle以及相关函数
    Py_Initialize();
    PyRun_SimpleString("import sys"); // 执行 python 中的短语句
    PyRun_SimpleString("sys.path.append('./Examples/python/')");
    PyObject *pModule, *pDict, *pArg;
    pModule = PyImport_ImportModule("cvmat2py"); // myModel:Python文件名
    pDict = PyModule_GetDict(pModule);
    PyObject *pFunc = PyDict_GetItemString(pDict, "load_image"); //从字典创建python函数对象
    // step 3 cv::Mat 转化为PyObject
    pArg = cvmat2py(image);
    // step 4 调用python函数
    PyObject *pValue = PyObject_CallObject(pFunc, pArg);
    return 0;
}