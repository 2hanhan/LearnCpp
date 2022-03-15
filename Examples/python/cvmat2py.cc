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
 * 这个耗时也太长了吧
 * @param image 
 * @return PyObject* 
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

    // step 1 引入modle以及相关函数
    Py_Initialize();
    PyRun_SimpleString("import sys"); // 执行 python 中的短语句
    PyRun_SimpleString("sys.path.append('./Examples/python/')");
    PyObject *pModule, *pDict;
    pModule = PyImport_ImportModule("cvmat2py"); // myModel:Python文件名
    pDict = PyModule_GetDict(pModule);
    PyObject *pFunc1 = PyDict_GetItemString(pDict, "load_image"); //从字典创建python函数对象
    PyObject *pFunc2 = PyDict_GetItemString(pDict, "load_image_name");
    clock_t time_start, time_end;
    time_start = clock();
    for (int i = 0; i < 1000; i++)
    {

        // step 2 cv::Mat 转化为PyObject
        cv::Mat image = cv::imread("Data/db1.jpg", CV_LOAD_IMAGE_COLOR);
        PyObject *pArg1 = cvmat2py(image);
        // step 3 调用python函数
        PyObject *pValue1 = PyObject_CallObject(pFunc1, pArg1);
    }
    time_end = clock();
    std::cout << "cv::Mat传递耗时：" << (time_end - time_start) * 1000 << "ms" << std::endl;

    time_start = clock();
    for (int i = 0; i < 1000; i++)
    {

        // step 2 字符串传递图像名
        PyObject *pArg2 = Py_BuildValue("s", "Data/db1.jpg");
        PyObject *pValue2 = PyObject_CallObject(pFunc2, pArg2);
    }
    time_end = clock();
    std::cout << "string传递耗时：" << (time_end - time_start) * 1000 << "ms" << std::endl;

    return 0;
}