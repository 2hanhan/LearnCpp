
/**
 * @file boost01.cc
 * @author wgq
 * @brief boost序列化的demo测试实现
 * @version 0.1
 * @date 2022-03-08
 *  一般通过在类内定义如下模版实现
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & data_name;//类内的数据名称
        ar & ptr_name;//类内的指针名称 或 数组名称
    }
 * @copyright Copyright (c) 2022
 *
 */
#include "lboost.h"
int main()
{
    // - 非入侵的序列化---------------------------------------------------
    {
        // create and open a character archive for output
        std::ofstream ofs("Save/gps_position.txt");
        boost::archive::text_oarchive oa(ofs); //创建写入序列化文件对象

        // create class instance
        const BASIC::gps_position g(35, 59, 24.567f);
        // write class instance to archive
        oa << g; //对象保存
        // close archive
        ofs.close();

        // ... some time later restore the class instance to its orginal state
        // create and open an archive for input
        std::ifstream ifs("Save/gps_position.txt", std::ios::binary);
        boost::archive::text_iarchive ia(ifs); //创建读出序列化文件对象
        // read class state from archive
        BASIC::gps_position newg;
        ia >> newg; //加载到对象
        // close archive
        ifs.close();
    }

    // - 入侵样式的序列化----------------------------------------------
    {
        std::ofstream ofs1("Save/gps_position1.txt");
        boost::archive::text_oarchive oa1(ofs1); //创建写入序列化文件对象

        // create class instance
        const BASIC::gps_position1 g1(30, 9, 54.567f);
        // write class instance to archive
        oa1 << g1; //对象保存
        // close archive
        ofs1.close();

        // ... some time later restore the class instance to its orginal state
        // create and open an archive for input
        std::ifstream ifs1("Save/gps_position1.txt", std::ios::binary);
        boost::archive::text_iarchive ia1(ifs1); //创建读出序列化文件对象
        // read class state from archive
        BASIC::gps_position newg1;
        ia1 >> newg1; //加载到对象
        ifs1.close(); // close archive
    }
    // - 序列化对象包含序列化对象-------------------------------------
    {
        // create and open a character archive for output
        std::ofstream ofs("Save/bus_stop.txt");
        boost::archive::text_oarchive oa(ofs); //创建写入序列化文件对象

        // create class instance
        BASIC::gps_position g1(01, 29, 21.517f);
        BASIC::gps_position g2(35, 59, 24.567f);

        std::string busname = "dadadada";
        BASIC::bus_stop bus(busname, g1, g2);
        // write class instance to archive
        oa << bus; //对象保存
        // close archive
        ofs.close();

        // ... some time later restore the class instance to its orginal state
        // create and open an archive for input
        std::ifstream ifs("Save/bus_stop.txt", std::ios::binary);
        boost::archive::text_iarchive ia(ifs); //创建读出序列化文件对象
        // read class state from archive
        BASIC::bus_stop newbus;
        ia >> newbus; //加载到对象
        // close archive
        ifs.close();
        newbus.show();
    }
    // - 序列化基类--------------------------------------
    {
        // create and open a character archive for output
        std::ofstream ofs("Save/bus_stop_corner.txt");
        boost::archive::text_oarchive oa(ofs); //创建写入序列化文件对象

        // create class instance
        BASIC::gps_position g1(01, 29, 21.517f);
        BASIC::gps_position g2(35, 59, 24.567f);

        std::string st1 = "shenyang";
        std::string st2 = "fengtian";
        BASIC::bus_stop_corner bus(g1, g2, st1, st1);
        // write class instance to archive
        oa << bus; //对象保存
        // close archive
        ofs.close();

        // ... some time later restore the class instance to its orginal state
        // create and open an archive for input
        std::ifstream ifs("Save/bus_stop_corner.txt", std::ios::binary);
        boost::archive::text_iarchive ia(ifs); //创建读出序列化文件对象
        // read class state from archive
        BASIC::bus_stop_corner newbus_cor;
        ia >> newbus_cor; //加载到对象
        // close archive
        ifs.close();
        newbus_cor.show_cor();
    }
    // - 序列化对象中指针指向的对象 STL的标准库里面的内容也能使用
    {
        // create and open a character archive for output
        std::ofstream ofs("Save/bus_route.txt");
        boost::archive::text_oarchive oa(ofs); //创建写入序列化文件对象

        std::cout << "写入" << std::endl;
        BASIC::bus_route bus_rou;
        for (int i = 0; i < 10; ++i)
        {
            BASIC::gps_position g1(i, i * 10, i * 1.2);
            BASIC::gps_position g2(i, i * 5, i * 0.6);

            std::string busname = "bus" + std::to_string(i);
            BASIC::bus_stop bus(busname, g1, g2);

            bus_rou.insert_bus_route(i, bus);
            bus_rou.push_bus_route(bus);
        }
        bus_rou.show_bus_route();

        oa << bus_rou; //对象保
        ofs.close();

        std::cout << "读取" << std::endl;
        std::ifstream ifs("Save/bus_route.txt", std::ios::binary);
        boost::archive::text_iarchive ia(ifs); //创建读出序列化文件对象

        BASIC::bus_route newbus_cor;
        ia >> newbus_cor; //加载到对象
        ifs.close();
        newbus_cor.show_bus_route();
    }

    return 0;
}