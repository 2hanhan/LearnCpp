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

#include <Python.h>

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

int main(int argc, char **argv)
{
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
    // - 使用class调用
    PyObject *pClass = PyDict_GetItemString(pDict, "HFNet");
    PyObject *pConstruct = PyInstanceMethod_New(pClass); //得到class构造函数
    PyObject *cons_args = Py_BuildValue("(s)", "./Examples/python/model/hfnet");

    PyObject *pInstance = PyObject_CallObject(pConstruct, cons_args); //实例化class
    std::cout << "c++ 实例化python class 完成" << std::endl;

    PyObject *result1, *result2;

    for (int i = 0; i < querys.size(); ++i)
    {
        // image_query = cv::imread(querys[i]);
        // cv::imshow("image_query", image_query);
        const char*  image_name1 =  querys[i].data();
        result1 = PyObject_CallMethod(pInstance, "inference", "s", image_name1);
        std::cout << querys[i] << std::endl;

        for (int j = 0; j < dbs.size(); ++j)
        {
            // image_db = cv::imread(dbs[j]);
            //  cv::imshow("image_db", image_db);
            //  cv::waitKey(100);
            const char*  image_name2 =  dbs[j].data();
            result2 = PyObject_CallMethod(pInstance, "inference", "s", image_name2);
        }
        // cv::destroyAllWindows();
    }

    return 0;
}