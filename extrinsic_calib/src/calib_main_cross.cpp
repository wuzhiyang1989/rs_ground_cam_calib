#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "CameraFactory.h"

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

typedef struct
{
    typedef struct
    {
        float u;
        float v;
    } imgPoint;
    imgPoint down_left;
    imgPoint down_right;
    imgPoint up_left;
    imgPoint up_right;

    void print()
    {
        std::cout << "down: left:(" << down_left.u << ", " << down_left.v << "), right:(" << down_right.u << ", " << down_right.v << ")" << std::endl;
        std::cout << "up: left:(" << up_left.u << ", " << up_left.v << "), right:(" << up_right.u << ", " << up_right.v << ")" << std::endl;
    }
} imgPoint_crosswalk;

typedef struct
{
    typedef struct
    {
        double x;
        double y;
        double z;
    } tagPoint;
    tagPoint down_left;
    tagPoint down_right;
    tagPoint up_left;
    tagPoint up_right;

    void print()
    {
        // std::cout.precision(10);
        std::cout << "down: left:(" << down_left.x << ", " << down_left.y << ", " << down_left.z << "), right:(" << down_right.x << ", " << down_right.y << ", " << down_right.z << ")" << std::endl;
        std::cout << "up: left:(" << up_left.x << ", " << up_left.y << ", " << up_left.z << "), right:(" << up_right.x << ", " << up_right.y << ", " << up_right.z << ")" << std::endl;
    }
} tagPoint_crosswalk;

void imagepoints_read(std::ifstream &imgpoints_file,
                      std::vector<imgPoint_crosswalk> &imgpoint_crosswalk_vec);
void targetpoints_read(std::ifstream &tagpoints_file,
                       std::vector<tagPoint_crosswalk> &tagpoint_crosswalk_vec);
void get3dPos_onroad(camodocal::CameraPtr camera,
                     const std::vector<cv::Point2f> &imagepoint,
                     std::vector<Eigen::Vector3d> &tagpoint,
                     Eigen::Vector3d plane_func_normal,
                     double plane_fund_d);
double get_distance_2points(Eigen::Vector3d p1, Eigen::Vector3d p2);
double get_distance_2points(cv::Point3f p1, cv::Point3f p2);

int main(int argc, char** argv)
{
    namespace fs = ::boost::filesystem;

    std::string calibDir;

    //========= Handling Program options =========
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("calib,c", boost::program_options::value<std::string>(&calibDir)->default_value("data"), "Directory containing camera calibration files.");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    std::cout << "calibDir = " << calibDir << std::endl;

    // Check if directory containing camera calibration files exists
    if (!boost::filesystem::exists(calibDir))
    {
        std::cout << "# ERROR: Directory " << calibDir << " does not exist." << std::endl;
        return -1;
    }

    {
        boost::filesystem::path calibFilePath(calibDir);
        std::ostringstream oss;
        oss << "camchain-fisheye-pinhole-equi.yaml";
        calibFilePath /= oss.str();
        YAML::Node config = YAML::LoadFile(calibFilePath.string());
        std::cout << "config = " << config["cam0"] << std::endl;
        const std::string camera_model = config["cam0"]["camera_model"].as<std::string>();
        std::cout << "camera_model = " << camera_model << std::endl;

        return 0;
    }

    // read camera params
    camodocal::CameraPtr camera;
    {
        boost::filesystem::path calibFilePath(calibDir);

        std::ostringstream oss;
        oss << "camera_calib_equi.yaml";
        calibFilePath /= oss.str();

        camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calibFilePath.string());
        if (camera.get() == 0)
        {
            std::cout << "# ERROR: Unable to read calibration file: " << calibFilePath.string() << std::endl;

            return -1;
        }
        std::cout << "# INFO: camera->cameraName() = " << camera->cameraName() << std::endl;
    }

    switch (camera->getParameters().modelType())
    {
    case camodocal::Camera::KANNALA_BRANDT:
        std::cout << "# INFO: Camera model: Kannala-Brandt" << std::endl;
        break;
    case camodocal::Camera::MEI:
        std::cout << "# INFO: Camera model: Mei" << std::endl;
        break;
    case camodocal::Camera::PINHOLE:
        std::cout << "# INFO: Camera model: Pinhole" << std::endl;
        break;
    case camodocal::Camera::SCARAMUZZA:
        std::cout << "# INFO: Camera model: Scaramuzza-Omnidirect" << std::endl;
        break;
    }

    // read image-target points
    // down_left, down_right
    // up_left, up_right
    std::vector<imgPoint_crosswalk> imgpoint_crosswalk_vec;
    std::vector<tagPoint_crosswalk> tagpoint_crosswalk_vec;
    std::vector<cv::Point2f> m_imagePoints;
    std::vector<cv::Point3d> m_scenePoints;
    std::vector<cv::Point3f> m_scenePoints_final;
    cv::Point3d start_point;    // original point
    {
        boost::filesystem::path imgpointsFilePath(calibDir);
        boost::filesystem::path tagpointsFilePath(calibDir);
        std::ostringstream oss_imgpoints, oss_tagpoints;
        oss_imgpoints << "image_points_equi.txt";
        oss_tagpoints << "target_points.txt";
        imgpointsFilePath /= oss_imgpoints.str();
        tagpointsFilePath /= oss_tagpoints.str();
        std::ifstream imgpoints_file(imgpointsFilePath.c_str());
        std::ifstream tagpoints_file(tagpointsFilePath.c_str());
        std::cout << "image points path : " << imgpointsFilePath.c_str() << std::endl;
        std::cout << "target points path : " << tagpointsFilePath.c_str() << std::endl;
        if (!imgpoints_file.is_open() || !tagpoints_file.is_open())
        {
            printf("cannot find file %s or %s\n", imgpointsFilePath.c_str(), tagpointsFilePath.c_str());
            return -1;
        }

        imagepoints_read(imgpoints_file, imgpoint_crosswalk_vec);
        targetpoints_read(tagpoints_file, tagpoint_crosswalk_vec);

        std::cout << "imgpoint_crosswalk_vec.size() = " << imgpoint_crosswalk_vec.size() << std::endl;
        std::cout << "tagpoint_crosswalk_vec.size() = " << tagpoint_crosswalk_vec.size() << std::endl;

        int index_control = 0;
        for (auto it = imgpoint_crosswalk_vec.begin(); it != imgpoint_crosswalk_vec.end(); it++) {
            cv::Point2f imgPoint_dl, imgPoint_dr, imgPoint_ul, imgPoint_ur;
            imgPoint_dl.x = it->down_left.u;
            imgPoint_dl.y = it->down_left.v;
            imgPoint_dr.x = it->down_right.u;
            imgPoint_dr.y = it->down_right.v;
            imgPoint_ul.x = it->up_left.u;
            imgPoint_ul.y = it->up_left.v;
            imgPoint_ur.x = it->up_right.u;
            imgPoint_ur.y = it->up_right.v;

            std::cout << "imgPoint_dl = " << imgPoint_dl << std::endl;
            std::cout << "imgPoint_dr = " << imgPoint_dr << std::endl;
            std::cout << "imgPoint_ul = " << imgPoint_ul << std::endl;
            std::cout << "imgPoint_ur = " << imgPoint_ur << std::endl;

            if (index_control < 18 && index_control > 10) {
                m_imagePoints.push_back(imgPoint_dl);
                m_imagePoints.push_back(imgPoint_dr);
                m_imagePoints.push_back(imgPoint_ul);
                m_imagePoints.push_back(imgPoint_ur);
            }

            index_control++;
        }

        index_control = 0;
        for (auto it = tagpoint_crosswalk_vec.begin(); it != tagpoint_crosswalk_vec.end(); it++) {
            cv::Point3d scenePoint_dl, scenePoint_dr, scenePoint_ul, scenePoint_ur;
            scenePoint_dl.x = it->down_left.x;    // m ==> cm
            scenePoint_dl.y = it->down_left.y;
            scenePoint_dl.z = it->down_left.z;
            scenePoint_dr.x = it->down_right.x;
            scenePoint_dr.y = it->down_right.y;
            scenePoint_dr.z = it->down_right.z;
            scenePoint_ul.x = it->up_left.x;
            scenePoint_ul.y = it->up_left.y;
            scenePoint_ul.z = it->up_left.z;
            scenePoint_ur.x = it->up_right.x;
            scenePoint_ur.y = it->up_right.y;
            scenePoint_ur.z = it->up_right.z;

            // std::cout.precision(10);
            std::cout << "scenePoint_dl = " << scenePoint_dl << std::endl;
            std::cout << "scenePoint_dr = " << scenePoint_dr << std::endl;
            std::cout << "scenePoint_ul = " << scenePoint_ul << std::endl;
            std::cout << "scenePoint_ur = " << scenePoint_ur << std::endl;

            if (index_control == tagpoint_crosswalk_vec.size() / 2 - 1) {
                start_point.x = scenePoint_dl.x;
                start_point.y = scenePoint_dl.y;
                start_point.z = scenePoint_dl.z;
            }

            if (index_control < 18 && index_control > 10) {
                m_scenePoints.push_back(scenePoint_dl);
                m_scenePoints.push_back(scenePoint_dr);
                m_scenePoints.push_back(scenePoint_ul);
                m_scenePoints.push_back(scenePoint_ur);
            }
            index_control++;
        }
    }

    // get relative position, frame center is the center point
    for (int i = 0; i < m_scenePoints.size(); ++i) {
        // std::cout << "m_scenePoints.at(" << i << ") = " << m_scenePoints.at(i) << std::endl;
        m_scenePoints.at(i).x -= start_point.x;
        m_scenePoints.at(i).y -= start_point.y;
        m_scenePoints.at(i).z -= start_point.z;
        cv::Point3f final_point;
        final_point.x = (float)m_scenePoints.at(i).x * 1000;        // m ===> mm
        final_point.y = (float)m_scenePoints.at(i).y * 1000;
        final_point.z = (float)m_scenePoints.at(i).z * 1000;
        m_scenePoints_final.push_back(final_point);
        std::cout << "m_scenePoints_final.at(" << i << ") = " << m_scenePoints_final.at(i) << std::endl;
    }

    // compute extrinsic between road ground and camera
    Eigen::Matrix3d R_cam_g;
    cv::Mat rvec;
    cv::Mat tvec;
    Eigen::Vector3d t_cam_g;
    {

        cv::Mat R0;
        // compute intrinsic camera parameters and extrinsic parameters for each of the views
        // note: rvec--rotation vector, in cv:Mat format(3 x 1)
        std::cout << "m_scenePoints.size() = " << m_scenePoints.size() << std::endl;
        std::cout << "m_imagePoints.size() = " << m_imagePoints.size() << std::endl;
        camera->estimateExtrinsics(m_scenePoints_final, m_imagePoints, rvec, tvec);

        cv::Rodrigues(rvec, R0);
        R_cam_g << R0.at<double>(0,0), R0.at<double>(0,1), R0.at<double>(0,2),
                   R0.at<double>(1,0), R0.at<double>(1,1), R0.at<double>(1,2),
                   R0.at<double>(2,0), R0.at<double>(2,1), R0.at<double>(2,2);

        t_cam_g << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

        std::cout << "R_cam_g = \n" << R_cam_g << std::endl;
        std::cout << "t_cam_g = " << t_cam_g.transpose() << std::endl;
    }

    Eigen::Matrix3d R_inverse;
    R_inverse = R_cam_g.inverse();
    Eigen::Matrix4d T, T_cam_cross, T_out;
    T.setIdentity();
    T.block<3,3>(0,0) = R_inverse;
    T_cam_cross.setIdentity();
    T_cam_cross.block<3,3>(0,0) = R_cam_g;
    T_cam_cross.block<3,1>(0,3) = t_cam_g;
    T_out = T * T_cam_cross;
    std::cout << "T_out = \n" << T_out << std::endl;

    // save extrinsic calib result
    cv::Mat extrinsic_calib_mat;
    cv::eigen2cv(T_cam_cross, extrinsic_calib_mat);
    boost::filesystem::path calibresultFilePath(calibDir);
    std::string calibresult_file = calibresultFilePath.string() + "/cam_ground_extrinsic_calib_result.yaml";
    cv::FileStorage fs_write(calibresult_file, cv::FileStorage::WRITE);
    std::cout << "saved extrinsic calib result file: " << calibresult_file << std::endl; 
    fs_write << "H_cam_ground" << extrinsic_calib_mat;
    fs_write.release();

    // verification
    // compute projection err
    {
        // read image
        boost::filesystem::path imgFilePath(calibDir);
        std::ostringstream oss;
        oss << "equi_undist_img.bmp";
        imgFilePath /= oss.str();
        cv::Mat image = cv::imread(imgFilePath.string());

        std::vector<cv::Point2f> estImagePoints;
        camera->projectPoints(m_scenePoints_final, rvec, tvec, estImagePoints);

        // visualize observed and reprojected points
        float errorMax = std::numeric_limits<float>::min();
        int errorMax_pos = 0;
        float erravg;
        float errorSum = 0.0f;
        int drawShiftBits = 4;
        int drawMultiplier = 1 << drawShiftBits;
        cv::Scalar green(0, 255, 0);
        cv::Scalar red(0, 0, 255);
        cv::Scalar blue(255, 0, 0);
        for (size_t i = 0; i < m_imagePoints.size(); ++i)
        {
            cv::Point2f pObs = m_imagePoints.at(i);
            cv::Point2f pEst = estImagePoints.at(i);

            cv::circle(image,
                       cv::Point(cvRound(pObs.x * drawMultiplier),
                                 cvRound(pObs.y * drawMultiplier)),
                       2, 
                       green,
                       2,
                       CV_AA,
                       drawShiftBits);

            cv::circle(image,
                       cv::Point(cvRound(pEst.x * drawMultiplier),
                                 cvRound(pEst.y * drawMultiplier)),
                       2,
                       red,
                       2,
                       CV_AA,
                       drawShiftBits);

            float error = cv::norm(pObs - pEst);

            errorSum += error;
            if (error > errorMax)
            {
                errorMax = error;
                errorMax_pos = i;
            }
            std::cout << "pixel_err_p" << i << " = " << error << std::endl;
        }

        // plot origin point
        std::vector<cv::Point2f> estImagePoint_origin_vec;
        std::vector<cv::Point3f> targetPoint_origin_vec;
        cv::Point3f targetPoint_origin(0);
        targetPoint_origin_vec.push_back(targetPoint_origin);
        camera->projectPoints(targetPoint_origin_vec, rvec, tvec, estImagePoint_origin_vec);
        cv::circle(image,
                    cv::Point(cvRound(estImagePoint_origin_vec.at(0).x * drawMultiplier),
                              cvRound(estImagePoint_origin_vec.at(0).y * drawMultiplier)),
                    200,
                    blue,
                    2,
                    CV_AA,
                    drawShiftBits);

        erravg = errorSum / m_imagePoints.size();
        std::ostringstream oss_text;
        oss_text << "Reprojection error: avg = " << erravg
                 << "   max = " << errorMax << ", at " << errorMax_pos;

        std::cout << "Reprojection error: avg = " << erravg << ", max = " << errorMax << ", at " << errorMax_pos << std::endl;

        cv::Point2f pObs_max = m_imagePoints.at(errorMax_pos);
        cv::Point2f pEst_max = estImagePoints.at(errorMax_pos);
        std::cout << "pObs_max = " << pObs_max << std::endl;
        std::cout << "pEst_max = " << pEst_max << std::endl;
        cv::circle(image,
                    cv::Point(cvRound(pObs_max.x * drawMultiplier),
                                cvRound(pObs_max.y * drawMultiplier)),
                    5, 
                    green,
                    2,
                    CV_AA,
                    drawShiftBits);

        cv::circle(image,
                    cv::Point(cvRound(pEst_max.x * drawMultiplier),
                                cvRound(pEst_max.y * drawMultiplier)),
                    5,
                    red,
                    2,
                    CV_AA,
                    drawShiftBits);       

        cv::putText(image,
                    oss_text.str(),
                    cv::Point(10, image.rows - 10),
                    cv::FONT_HERSHEY_COMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1,
                    CV_AA);

        // save result image
        boost::filesystem::path resultimgFilePath(calibDir);
        std::string result_file = resultimgFilePath.string() + "/reprojection_result.bmp";
        cv::imwrite(result_file, image);

        cv::Mat output_show_img;
        cv::resize(image, output_show_img, cv::Size(1280, 520));
        cv::imshow("image", output_show_img);
        cv::waitKey(0);
    }

    // get 3d points pos on the ground from the 2d points
    {
        // 1.get the ground plane func under camera's coordinates
        Eigen::Vector3d plane_normal_vec, t_cam_g;
        Eigen::Matrix3d R_cam_g = T_cam_cross.block<3,3>(0,0);      // 3rows and 3 cols matrix started from (0,0)
        t_cam_g = T_cam_cross.block<3,1>(0,3);
        plane_normal_vec = R_cam_g.block<3,1>(0,2);
        double d = -plane_normal_vec.transpose() * t_cam_g;
        std::cout << "plane func:\n" << "n = " << plane_normal_vec.transpose() << "\nd = " << d << std::endl;

        // 2.get 3d points pos on the ground
        // cv::Point2f m_imagePoint = m_imagePoints.at(0);
        std::vector<Eigen::Vector3d> tagpoints_measured;
        get3dPos_onroad(camera,
                        m_imagePoints,
                        tagpoints_measured,
                        plane_normal_vec,
                        d);

        // 3. compare distance
        double d_0_9_meaure = get_distance_2points(tagpoints_measured.at(0), tagpoints_measured.at(9));
        double d_0_17_mesure = get_distance_2points(tagpoints_measured.at(0), tagpoints_measured.at(17));

        double d_0_9_truth = get_distance_2points(m_scenePoints_final.at(0), m_scenePoints_final.at(9));
        double d_0_17_truth = get_distance_2points(m_scenePoints_final.at(0), m_scenePoints_final.at(17));

        std::cout << "d_0_9_meaure = " << d_0_9_meaure << ", d_0_17_mesure = " << d_0_17_mesure << std::endl;
        std::cout << "d_0_9_truth = " << d_0_9_truth << ", d_0_17_truth = " << d_0_17_truth << std::endl;

        // 4. compute pos err
        // convert pos to camera frame
        std::vector<Eigen::Vector3d> tagpoints_truth;
        for (auto it = m_scenePoints_final.begin(); it != m_scenePoints_final.end(); it++) {
            cv::Point3f p_cv = *it;
            Eigen::Vector3d p_world;
            p_world(0) = p_cv.x;
            p_world(1) = p_cv.y;
            p_world(2) = p_cv.z;            
            Eigen::Vector3d p_cam;
            p_cam = R_cam_g * p_world + t_cam_g;
            tagpoints_truth.push_back(p_cam);
        }
        // compute pos err
        double d_err_max = 0;
        double d_err_sum = 0;
        double d_err_avg = 0;
        for (int i = 0; i < tagpoints_truth.size(); i++) {
            Eigen::Vector3d tagpoint_measured, tagpoint_truth;
            tagpoint_measured = tagpoints_measured.at(i);
            tagpoint_truth = tagpoints_truth.at(i);
            double d_err = get_distance_2points(tagpoint_measured, tagpoint_truth);
            if (d_err_max < d_err) {
                d_err_max = d_err;
            }
            d_err_sum += d_err;
            std::cout << "d_err_p" << i << " = " << d_err << std::endl;
        }
        d_err_avg = d_err_sum / tagpoints_truth.size();
        std::cout << "d_err_max = " << d_err_max << ", d_err_avg = " << d_err_avg << std::endl;
    }

    return 0;
}

double get_distance_2points(Eigen::Vector3d p1, Eigen::Vector3d p2)
{
    Eigen::Vector3d delt = p1 -p2;
    double d1, d2, d3;
    d1 = delt(0);
    d2 = delt(1);
    d3 = delt(2);
    return sqrt(d1 * d1 + d2 * d2 + d3 * d3);
}

double get_distance_2points(cv::Point3f p1, cv::Point3f p2)
{
    double d1, d2, d3;
    d1 = p1.x - p2.x;
    d2 = p1.y - p2.y;
    d3 = p1.z - p2.z;
    return sqrt(d1 * d1 + d2 * d2 + d3 * d3);
}

void get3dPos_onroad(camodocal::CameraPtr camera,
                     const std::vector<cv::Point2f> &imagepoint,
                     std::vector<Eigen::Vector3d> &tagpoint,
                     Eigen::Vector3d plane_func_normal,
                     double plane_fund_d)
{
    for (auto it = imagepoint.begin(); it != imagepoint.end(); it++) {
        cv::Point2f img_p = *it;
        Eigen::Vector3d P_ray;
        camera->liftProjective(Eigen::Vector2d(img_p.x, img_p.y), P_ray);
        // get Intersection of plane and ray
        double t = -plane_fund_d / (plane_func_normal.transpose() * P_ray);
        Eigen::Vector3d tag_p = P_ray * t;
        tagpoint.push_back(tag_p);
    }
}

void imagepoints_read(std::ifstream &imgpoints_file,
                      std::vector<imgPoint_crosswalk> &imgpoint_crosswalk_vec)
{
    std::vector<std::string> vec;
    std::string temp;
    while (getline(imgpoints_file, temp)) {
        if (temp.size() != 0) {
            vec.push_back(temp);
        }
    }

    for (auto it = vec.begin(); it != vec.end(); it++)
    {
        std::istringstream is(*it);
        std::string s;
        int pam = 0;
        imgPoint_crosswalk imgpoint_crosswalk;
        while (is >> s) {
            switch (pam)
            {
            case 0:
                imgpoint_crosswalk.down_left.u = atof(s.c_str());
                break;
            case 1:
                imgpoint_crosswalk.down_left.v = atof(s.c_str());
                break;
            case 2:
                imgpoint_crosswalk.down_right.u = atof(s.c_str());
                break;
            case 3:
                imgpoint_crosswalk.down_right.v = atof(s.c_str());
                break;
            case 4:
                imgpoint_crosswalk.up_left.u = atof(s.c_str());
                break;
            case 5:
                imgpoint_crosswalk.up_left.v = atof(s.c_str());
                break;
            case 6:
                imgpoint_crosswalk.up_right.u = atof(s.c_str());
                break;
            case 7:
                imgpoint_crosswalk.up_right.v = atof(s.c_str());
                break;
            default:
                break;
                std::cout << "err: invalid" << std::endl;
            }

            pam++;
        }
        imgpoint_crosswalk.print();
        imgpoint_crosswalk_vec.push_back(imgpoint_crosswalk);
    }
}

void targetpoints_read(std::ifstream &tagpoints_file,
                       std::vector<tagPoint_crosswalk> &tagpoint_crosswalk_vec)
{
    std::vector<std::string> vec;
    std::string temp;
    while (getline(tagpoints_file, temp)) {
        vec.push_back(temp);
    }

    for (auto it = vec.begin(); it != vec.end(); it++)
    {
        std::istringstream is(*it);
        std::string s;
        int pam = 0;
        tagPoint_crosswalk tagpoint_crosswalk;
        while (is >> s) {
            switch (pam)
            {
            case 0:
                tagpoint_crosswalk.down_left.x = atof(s.c_str());
                break;
            case 1:
                tagpoint_crosswalk.down_left.y = atof(s.c_str());
                break;
            case 2:
                tagpoint_crosswalk.down_left.z = atof(s.c_str());
                break;
            case 3:
                tagpoint_crosswalk.down_right.x = atof(s.c_str());
                break;
            case 4:
                tagpoint_crosswalk.down_right.y = atof(s.c_str());
                break;
            case 5:
                tagpoint_crosswalk.down_right.z = atof(s.c_str());
                break;
            case 6:
                tagpoint_crosswalk.up_left.x = atof(s.c_str());
                break;
            case 7:
                tagpoint_crosswalk.up_left.y = atof(s.c_str());
                break;
            case 8:
                tagpoint_crosswalk.up_left.z = atof(s.c_str());
                break;
            case 9:
                tagpoint_crosswalk.up_right.x = atof(s.c_str());
                break;
            case 10:
                tagpoint_crosswalk.up_right.y = atof(s.c_str());
                break;
            case 11:
                tagpoint_crosswalk.up_right.z = atof(s.c_str());
                break;
            default:
                break;
                std::cout << "err: invalid" << std::endl;
            }

            pam++;
        }
        tagpoint_crosswalk.print();
        tagpoint_crosswalk_vec.push_back(tagpoint_crosswalk);
    }
}


