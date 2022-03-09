#include <lboost.h>

namespace BASIC
{

    gps_position::gps_position() {}
    gps_position::gps_position(gps_position &gps)
    {
        degrees = gps.degrees;
        minutes = gps.minutes;
        seconds = gps.seconds;
    }
    gps_position::gps_position(int d, int m, float s) : degrees(d), minutes(m), seconds(s)
    {
    }

    gps_position1::gps_position1() {}
    gps_position1::gps_position1(int d, int m, float s) : degrees(d), minutes(m), seconds(s)
    {
    }

    bus_stop::bus_stop() {}
    bus_stop::bus_stop(std::string busname, gps_position latitude, gps_position longitude)
        : busname(busname), latitude(latitude), longitude(longitude)
    {
    }
    void bus_stop::show()
    {
        std::cout << busname << std::endl;
        std::cout << "d:" << latitude.degrees << "m:" << latitude.minutes << "s:" << latitude.seconds << std::endl;
        std::cout << "d:" << longitude.degrees << "m:" << longitude.minutes << "s:" << longitude.seconds << std::endl;
    }

    void bus_stop_corner::show_cor()
    {
        std::cout << "st1:" << street1 << "st2:" << street1 << std::endl;
        bus_stop::show();
    }

    void bus_route::insert_bus_route(int i, bus_stop &bus)
    {
        stops[i] = new bus_stop(bus.busname, bus.latitude, bus.longitude);
    }
    void bus_route::push_bus_route(bus_stop &bus)
    {
        stops_v.push_back(&bus);
    }
    void bus_route::show_bus_route()
    {
        std::cout << "基本数据类型" << std::endl;
        for (int i = 0; i < 10; ++i)
        {
            std::cout << "No." << i << std::endl;
            stops[i]->show();
        }

        std::cout << "STL容器" << std::endl;
        for (int i = 0; i < 10; ++i)
        {
            std::cout << "No." << i << std::endl;
            stops_v[i]->show();
        }
    }
}