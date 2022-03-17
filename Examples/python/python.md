# c++调用python
## python文件
`test.py`
```python
def Hello(s):
    print ("Hello World")
    print(s)

def Add(a, b):
    print('a=', a)
    print ('b=', b)
    return a + b

class Test:
    def __init__(self):
        print("Init")
    def SayHello(self, name):
        print ("Hello,", name)
        return name
```
## 头文件包含
```c++
#include <python3.6/Python.h>//应该是调用什么版本解使用什么版本的python
#include <Python.h>//使用的方式和cmakelist有些关系
```
## 初始化python调用
```c++
Py_Initialize(); 初始化
PyRun_SimpleString("import sys");   //设置文件路径
PyRun_SimpleString("sys.path.append('./')");//./表示当前执行程序的路径
//中间在使用下面的一些具体的函数调用方式实现调用python
Py_Finalize();
```
## 具体的函数调用
- 直接调用代码
```c++
PyRun_SimpleString(）; //使用字符串实现直接写的python程序
```
- 指向python的对象的指针
```c++
PyObject *object；//创建python的object的指针，可以是model class fun arg return
Py_DECREF(object);//销毁object,感觉和c++的new和delete比较像
```
- 引入python文件的model
```c++
pModule = PyImport_ImportModule("test"); // 导入python的文件modle，不需要写后缀名.py
pDict = PyModule_GetDict(pModule); // 以字典形式存储model的信息，用于后续的调用
```
 - python数据创建
```c++
//创建python的数据类型的object
PyObject *Arg = Py_BuildValue("(s)", "dawd a");
PyObject *Arg = Py_BuildValue("(i, i)", 1, 2);
//创建好多参数
PyObject* cons_args = PyTuple_New(2);
PyObject* cons_arg1 = PyLong_FromLong(1);
PyObject* cons_arg2 = PyLong_FromLong(999);
PyTuple_SetItem(cons_args, 0, cons_arg1);
PyTuple_SetItem(cons_args, 1, cons_arg2)

```
- 函数调用
```c++
PyObject *pFunc = PyObject_GetAttrString(pModule, "Hello"); //从字符串创建python函数object
PyObject *pFunc = PyDict_GetItemString(pDict, "Add"); //从字典创建python函数object
PyObject *result = PyEval_CallObject(pFunc, pArg); //实现python的函数object的调用并接受return
int c;
PyArg_Parse(result, "i", &c); // 数据类型转化python -> c++
```
- class调用

1. 就离谱网上好多教程把`构造函数`当成`class的实例化对象`使用，后面在class的`def __init__()`中加了个`print()`发现没有输出才发现这个问题。
2. 还有一个问题，对于`python`代码中调用`class`的`func`貌似在`c++`的调用中会有问题，我这里发现不能这么调用，所有解决方案就是把`class`的声明、构造、实例化全部拆开到`c++`中调用
3. `PyObject_CallMethod()`这个函数传递参数`"s"`的时候不能使用`string`需要转化为`const char*  image_name1 =  querys[i].data();`才能正常传递参数

```c++
PyObject *pClass = PyDict_GetItemString(pDict, "HFNet");//获取class
PyObject *pConstruct = PyInstanceMethod_New(pClass); //获取class构造函数
PyObject *cons_args = Py_BuildValue("(s)", "./Examples/python/model/hfnet");//设置实例化所需要的参数
PyObject *pInstance = PyObject_CallObject(pConstruct, cons_args); //实例化class
// PyObject_CallMethod(pInstance, methodname, "O", args) 
// 参数0：[class的object]
// 参数1：[class的fun的名称]
// 参数2：[输入参数的类型] [O表示object][s表示char*，string的类型需要进行转化] 
// ? 这里有个疑问，self参数需要输入吗？网上的example我看到过有的也有不传递的
// 参数...：[class内的fun的各个参数]
std::vector<std::string> querys;
const char*  image_name1 =  querys[i].data();
result1 = PyObject_CallMethod(pInstance, "inference", "s", image_name1);
```

## 链接CMakeLists.txt
 - 最终版本以项目目录下的`CMakeLists.txt`为准好事
```CMake
#set(PYTHON_INCLUDE_DIR /usr/include/python3.6)#这个好像并不影响
#python的虚拟环境，需要添加对应的.so，bash中 source /venv/bin/activate
set(PYTHON_LIBRARY /usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6.so)
#看到有人添加
#include_directories( /usr/include/python3.6)
#的这里我没弄感觉也能编译	
add_executable(usepython
        Examples/python/usepython.cc)
target_link_libraries(usepython -lpython3.6m)
##写法2还是用下面这种吧，这样头文件包含就可以直接#include <Python.h>不用指定python版本
#下面这种写法写可以
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_executable(usepython
        Examples/python/usepython.cc)
target_link_libraries(usepython ${PYTHON_LIBRARIES})
```
## Example
简单的调用例子如下
[c++](usepython.cc)
[python](testpy.py)
## c++ python的数据类型转化
### 图像数据格式cv::Mat传递
```c++
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
```
### numpy数据格式转化
- `numpy.ndarray`中的`numpy.float32、numpy.int32`转化为`c++`格式的数据
- 使用以下函数遍历实现每个一维、二维数据的读取
```c++
*(float *)PyArray_GETPTR1(Py_global_desc, i);
*(int *)PyArray_GETPTR1(Py_global_desc, i, j);
```
-具体的例子可以参考函数[useNet.cc中`void GetFeature()`](useNet.cc)

## 一些奇怪的错误
1. `import cv2`放在`python`文件中，没有缩进就会直接报错，放在函数内部就不会出现错误，后来有没有这个问题了，可能是`c++`写example的时候使用`{}`分别引入两次python导致的？？`{}`不是会自己提供作用阈吗？，还是这个接口是全局的?
2. 之前使用`c++`多线程开`cv::imshow()`程序会崩溃，没想到用`c++`自己用一个`cv::imshow()`,然后python也开一个`cv.imshow()`也会崩溃。啊这当时还以为`cv::Mat`转为`PyObject`的格式出了问题呢，结果不是的。