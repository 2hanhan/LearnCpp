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
```c++
PyRun_SimpleString(）; //使用字符串实现直接写的python程序
//指向python的对象的指针
PyObject *object,*pDict；//创建python的object的指针，可以是model class fun arg return
Py_DECREF(object);//销毁object,感觉和c++的new和delete比较像
//引入python文件
pModule = PyImport_ImportModule("test"); // 导入python的文件modle，不需要写后缀名.py
pDict = PyModule_GetDict(pModule); // 以字典形式存储model的信息，用于后续的调用
//python数据创建
Arg = Py_BuildValue("(i, i)", 1, 2); //创建python的数据类型的object
// 函数调用
PyObject *pFunc = PyObject_GetAttrString(pModule, "Hello"); //从字符串创建python函数object
pFunc = PyDict_GetItemString(pDict, "Add"); //从字典创建python函数object
PyObject *result = PyEval_CallObject(pFunc, pArg); //实现python的函数object的调用并接受return
int c;
PyArg_Parse(result, "i", &c); // 数据类型转化python -> c++
//class调用
PyObject *pClass = PyDict_GetItemString(pDict, "Test");//从字典创建python的class
PyObject *pInstance = PyInstanceMethod_New(pClass);//创建class的object
// PyObject_CallMethod(pInstance, methodname, "O", args) 
// 参数0：[class的object]
// 参数1：[class的fun的名称]
// 参数2：[输入参数的类型] [O表示object][s表示string] 
// ? 这里有个疑问，self参数需要输入吗？网上的example我看到过有的也有不传递的
// 参数...：[class内的fun的各个参数]
result = PyObject_CallMethod(pInstance, "SayHello", "(Os)", pInstance, "Charity");
```

## 相关链接CMakeLists.txt
```CMake
#看到有人添加
#include_directories( /usr/include/python3.6)
#的这里我没弄感觉也能编译	
add_executable(usepython
        Examples/python/usepython.cc)
target_link_libraries(usepython -lpython3.6m)
```
## Example
简单的调用例子如下
[c++](usepython.cc)
[python](testpy.py)
## c++ python的数据类型转化
### 图像数据格式cv::Mat传递