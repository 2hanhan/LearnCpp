#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>

//./Examples/bin/readyaml ./Examples/yaml/orbsalm.yaml 
/**
 * @brief
 * 输入yaml路径进行读取
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char **argv)
{

    if (argc < 2)
    {
        std::cout << "请输入yaml文件路径" << std::endl;
        return 1;
    }

    std::string strSettingsFile = argv[1];
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
        exit(-1);
    }

    double fx, fy, cx, cy;
    int nFeatures;
    std::string SaveAtlasToFile;

    fx = fsSettings["Camera1.fx"];
    fy = fsSettings["Camera1.fy"];
    cx = fsSettings["Camera1.cx"];
    cy = fsSettings["Camera1.cy"];

    std::cout << "fx:" << fx << std::endl;
    std::cout << "fy:" << fy << std::endl;
    std::cout << "cx:" << cx << std::endl;
    std::cout << "cy:" << cy << std::endl;

    nFeatures = fsSettings["ORBextractor.nFeatures"];
    std::cout << "nFeatures:" << nFeatures << std::endl;

    SaveAtlasToFile = std::string(fsSettings["System.SaveAtlasToFile"]);
    std::cout << "SaveAtlasToFile:" << SaveAtlasToFile << std::endl;

    return 0;
}
