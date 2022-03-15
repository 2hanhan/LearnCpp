/**
 * @file usepython.cc
 * @author gq
 * @brief
 * @version 0.1
 * @date 2022-03-14
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <iostream>
#include <Python.h>
//#include <python3.6/Python.h>

int main()
{

    // - 基础的调用

    // 初始化Python
    //在使用Python系统前，必须使用Py_Initialize对其
    //进行初始化。它会载入Python的内建模块并添加系统路
    //径到模块搜索路径中。这个函数没有返回值，检查系统
    //是否初始化成功需要使用Py_IsInitialized。
    Py_Initialize();

    // 检查初始化是否成功
    if (!Py_IsInitialized())
    {
        return -1;
    }

    // - 添加当前路径
    // PyRun_SimpleString(FILE *fp, char *filename);
    //直接运行python的代码
    PyRun_SimpleString("import sys");
    // PyRun_SimpleString("print '---import sys---'");
    //下面这个./表示当前运行程序的路径，如果使用../则为上级路径，根据此来设置
    PyRun_SimpleString("sys.path.append('./Examples/python/')");
    PyRun_SimpleString("print(sys.path)");
    // - 引入模块
    PyObject *pModule, *pDict, *preturn; // python的对象的指针
    //要调用的python文件名
    pModule = PyImport_ImportModule("testpy"); //加载模型不用加后缀名.py
    if (!pModule)
    {
        printf("can't find your_file.py");
        getchar();
        return -1;
    }
    //获取模块字典属性
    pDict = PyModule_GetDict(pModule);
    if (!pDict)
    {
        return -1;
    }
    std::cout << "加载模型成功" << std::endl;

    PyObject *pFunc;
    // - 直接调用函数
    //直接获取模块中的函数
    PyRun_SimpleString("print('使用string获取模块中的函数')");
    pFunc = PyObject_GetAttrString(pModule, "Hello"); //从字符串创建python函数对象

    //参数类型转换，传递一个字符串
    //将c/c++类型的字符串转换为python类型，元组中的python类型查看python文档
    PyRun_SimpleString("print('生成python的数据类型')");
    PyObject *pArg = Py_BuildValue("(s)", "Hello");

    PyRun_SimpleString("print('实现py函数调用')");
    //调用直接获得的函数，并传递参数
    PyEval_CallObject(pFunc, pArg);

    // - 从字典属性中获取函数
    PyRun_SimpleString("print('使用字典获取模块中的函数')");
    pFunc = PyDict_GetItemString(pDict, "Add"); //从字典创建python函数对象
    //参数类型转换，传递两个整型参数
    pArg = Py_BuildValue("(i, i)", 1, 2); //创建python的数据类型
    //调用函数，并得到python类型的返回值
    PyObject *result = PyEval_CallObject(pFunc, pArg); //实现python的函数调用
    // c用来保存c/c++类型的返回值
    int c;
    //将python类型的返回值转换为c/c++类型
    PyArg_Parse(result, "i", &c); // 数据类型转化python -> c++
    //输出返回值
    std::cout << "a + b = c =" << c << std::endl;

    // - 通过字典属性获取模块中的类
    PyRun_SimpleString("print('---------通过字典属性获取模块中的class-----------')");
    PyObject *pClass = PyDict_GetItemString(pDict, "Test");

    //实例化获取的类
    PyRun_SimpleString("print('实例化获取的class')");
    PyObject *pInstance = PyInstanceMethod_New(pClass);
    //调用类的方法
    PyRun_SimpleString("print('调用class中Method')");
    result = PyObject_CallMethod(pInstance, "SayHello", "(Oss)", pInstance, "zyh", "12");
    //输出返回值
    char *name = NULL;
    PyRun_SimpleString("print('输出返回结果')");
    PyArg_Parse(result, "s", &name);
    printf("%s\n", name);

    // - 使用文件路径传递图像参数
     PyRun_SimpleString("print('---------使用文件路径传递图像参数-----------')");
    pFunc = PyDict_GetItemString(pDict, "show_image"); //从字典创建python函数对象
    pArg = Py_BuildValue("(s)", "Data/db1.jpg");
    preturn = PyEval_CallObject(pFunc, pArg);
    name = NULL;
    PyArg_Parse(preturn, "s", &name);
    printf("%s\n", name);

    Py_DECREF(pModule);
    Py_DECREF(pDict);
    Py_DECREF(pFunc);
    Py_DECREF(pArg);
    Py_DECREF(result);
    Py_DECREF(pClass);
    Py_DECREF(pInstance);
    Py_DECREF(preturn);
    //释放python
    Py_Finalize();
    // getchar();

    return 0;
}
