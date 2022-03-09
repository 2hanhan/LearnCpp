#ifndef L_BOOST_H
#define L_BOOST_H

/**
 * @file lboost.h
 * @author wgq
 * @brief
 * @version 0.1
 * @date 2022-03-08
 *
 * ORB_SLAM3使用boost库进行地图的保存与加载
 * 这里根据网上的一些文档参考写一些demo
 * 尝试boost进行序列化的与反序列化的方式
 * 整体上的保存与加载算法属于递归调用的算法
 * 类的包含关系的存储加载使用递归实现
 * - 侵入样式的序列化 gps_position
 * - 非侵入样式的序列化 gps_position1
 * - 序列化的对象内部有序列化的类 bus_stop
 * - 序列化类内的指针 bus_route 指针可以使用数组存放 也可以使用 STL容器存放
 * - 序列化可以加载不同版本的类的数据
 * 
 * - 序列化还能够对于不同版本的class实现，
 * - 通过引入#include <boost/serialization/split_member.hpp>可以拆解save与load
 * - 不同版本的class可能包含的对象、数据不一致，设置version号，可以实现兼容
 * - 这里不用维护不同的class不具体写了
 * @copyright Copyright (c) 2022
 *
 */

#include <iostream>
#include <fstream>
#include <string>
#include <boost/archive/text_iarchive.hpp> //进行txt格式的读取
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp> //包含基类的序列化
#include <boost/serialization/vector.hpp>      //序列化容器

namespace BASIC
{
    /* @brief
     * 侵入式的实现序列化
     * serialize函数定义在类的内部
     */
    class gps_position
    {
    private:
        friend class boost::serialization::access;
        // When the class Archive corresponds to an output archive, the
        // & operator is defined similar to <<.  Likewise, when the class Archive
        // is a type of input archive the & operator is defined similar to >>.
        /**
         * @brief
         * 对于每个通过序列化“被存储”的类，必须存在一个函数去实现“存储”其所有状态数据。
         * 对于每个通过序列化“被装载”的类，必须存在一个函数来实现“装载”其所有状态数据。
         * 在例子中，这些函数是 模板成员函数serialize
         * @tparam Archive
         * @param ar
         * @param version
         */
        template <class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar &degrees;
            ar &minutes;
            ar &seconds;
        }

    public:
        int degrees;
        int minutes;
        float seconds;

    public:
        gps_position();
        gps_position(gps_position &gps);
        gps_position(int d, int m, float s);
    };

    /**
     * @brief
     * 非侵入样式的序列化
     */
    class gps_position1
    {
    public:
        int degrees;
        int minutes;
        float seconds;
        gps_position1();
        gps_position1(int d, int m, float s);
    };

    /**
     * @brief
     * 序列化的对象内部有序列化的类
     */
    class bus_stop
    {
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar &busname;
            ar &latitude;
            ar &longitude;
        }

    public:
        std::string busname;
        gps_position latitude;
        gps_position longitude;

    public:
        bus_stop();
        bus_stop(std::string busname, gps_position latitude, gps_position longitude);
        // See item # 14 in Effective C++ by Scott Meyers.
        // re non-virtual destructors in base classes.
        virtual ~bus_stop()
        {
        }
        void show();
    };

    /**
     * @brief
     * 派生类与基类的序列化
     */
    class bus_stop_corner : public bus_stop
    {
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            // serialize base class information
            // 序列化基类的信息
            ar &boost::serialization::base_object<bus_stop>(*this);
            ar &street1;
            ar &street2;
        }
        std::string street1;
        std::string street2;
        virtual std::string description() const
        {
            return street1 + " and " + street2;
        }

    public:
        bus_stop_corner() {}
        bus_stop_corner(gps_position &lat_, gps_position &long_,
                        std::string &s1_, std::string &s2_)
            : bus_stop("0000", lat_, long_), street1(s1_), street2(s2_)
        {
        }
        void show_cor();
    };

    /**
     * @brief
     * 因为指针本身不存储数据
     * 所有不能仅仅存储指针、需要对指针指向的对象进行序列化
     * load完所有的数据后，会将新的指针存储到类之中
     */
    class bus_route
    {
        friend class boost::serialization::access;
        bus_stop *stops[10];
        std::vector<bus_stop *> stops_v;

        template <class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {

            ar &stops;
            /*
             * 上面的代码也可选下面的代替，但是boost能够自动识别数组类型
             * for(int i = 0; i < 10; ++i)
             * ar & stops[i];
             */
            ar &stops_v;
        }

    public:
        bus_route() {}
        void insert_bus_route(int i, bus_stop &bus);
        void push_bus_route(bus_stop &bus);
        void show_bus_route();
    };

} // namespace BASIC

namespace boost
{
    namespace serialization
    {

        /**
         * @brief
         * 非侵入样式的序列化，需要类里面有足够数量的public成员才好使用
         * @tparam Archive
         * @param ar
         * @param g
         * @param version
         */
        template <class Archive>
        void serialize(Archive &ar, BASIC::gps_position1 &g, const unsigned int version)
        {
            ar &g.degrees;
            ar &g.minutes;
            ar &g.seconds;
        }

    } // namespace serialization
} // namespace boost

#endif // BOOST_H