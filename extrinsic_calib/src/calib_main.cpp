#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>      // Header file needed to use setprecision
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "CameraFactory.h"
#include "PinholeCamera.h"

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <limits.h>     // header for INT_MAX

// #include <Python.h>

typedef struct
{
    float u;
    float v;
} imgPoint;

typedef struct
{
    double x;
    double y;
    double z;
} tagPoint;

void imagetargetpoints_only_read(std::ifstream &imgtagpoints_file,
                            std::vector<imgPoint> &imgpoint_vec,
                            std::vector<tagPoint> &tagpoint_vec);
void imagetargetpoints_read(std::ifstream &imgtagpoints_file,
                            std::vector<imgPoint> &imgpoint_vec,
                            std::vector<tagPoint> &tagpoint_vec,
                            Eigen::Matrix3d &R_global_local);
bool get3dPos_onroad_single(camodocal::CameraPtr camera,
                            const cv::Point2f &imagepoint,
                            Eigen::Vector3d &tagpoint_c,
                            Eigen::Vector3d &tagpoint_g,
                            Eigen::Vector3d plane_func_normal,
                            double plane_fund_d,
                            Eigen::Matrix4d T_cam_ground);
void get3dPos_onroad_list(camodocal::CameraPtr camera,
                          const std::vector<cv::Point2f> &imagepoints,
                          std::vector<Eigen::Vector3d> &tagpoints_c,
                          std::vector<Eigen::Vector3d> &tagpoints_g,
                          Eigen::Vector3d plane_func_normal,
                          double plane_fund_d,
                          Eigen::Matrix4d T_cam_ground);
double get_distance_2points(Eigen::Vector3d p1, Eigen::Vector3d p2);
double get_distance_2points(cv::Point3f p1, cv::Point3f p2);

static double point_plane_dist(cv::Point3f point, float *plane_func)
{
    double dt = 0.0;
    double mA, mB, mC, mD, mX, mY, mZ;

    //std::cout << "point = " << point << std::endl;

    mA = plane_func[0];
    mB = plane_func[1];
    mC = plane_func[2];
    mD = plane_func[3];

    mX = point.x;
    mY = point.y;
    mZ = point.z;

    //std::cout << "mA * mA + mB * mB + mC * mC = " << mA * mA + mB * mB + mC * mC << std::endl;

    if (mA * mA + mB * mB + mC * mC) {
        dt = fabs(mA * mX + mB * mY + mC * mZ - mD) / sqrt(mA * mA + mB * mB + mC * mC);
    } else {
        dt = sqrt(mX * mX + mY * mY + mZ * mZ);
    }

    return dt;
}

static void cvFitPlane(const CvMat* points, float* plane)
{
	// Estimate geometric centroid.
	int nrows = points->rows;
	int ncols = points->cols;
	int type = points->type;
	CvMat* centroid = cvCreateMat(1, ncols, type);
	cvSet(centroid, cvScalar(0));
	for (int c = 0; c<ncols; c++){
		for (int r = 0; r < nrows; r++)
		{
			centroid->data.fl[c] += points->data.fl[ncols*r + c];
		}
		centroid->data.fl[c] /= nrows;
	}
	// Subtract geometric centroid from each point.
	CvMat* points2 = cvCreateMat(nrows, ncols, type);
	for (int r = 0; r<nrows; r++)
		for (int c = 0; c<ncols; c++)
			points2->data.fl[ncols*r + c] = points->data.fl[ncols*r + c] - centroid->data.fl[c];
	// Evaluate SVD of covariance matrix.
	CvMat* A = cvCreateMat(ncols, ncols, type);
	CvMat* W = cvCreateMat(ncols, ncols, type);
	CvMat* V = cvCreateMat(ncols, ncols, type);
	cvGEMM(points2, points, 1, NULL, 0, A, CV_GEMM_A_T);
	cvSVD(A, W, NULL, V, CV_SVD_V_T);
	// Assign plane coefficients by singular vector corresponding to smallest singular value.
	plane[ncols] = 0;
	for (int c = 0; c<ncols; c++){
		plane[c] = V->data.fl[ncols*(ncols - 1) + c];
		plane[ncols] += plane[c] * centroid->data.fl[c];
	}
	// Release allocated resources.
	cvReleaseMat(&centroid);
	cvReleaseMat(&points2);
	cvReleaseMat(&A);
	cvReleaseMat(&W);
	cvReleaseMat(&V);
}

bool plane_fit(std::vector<cv::Point3f> plane_3d_points, float *plane_func)
{
    CvMat *points_mat = cvCreateMat(plane_3d_points.size(), 3, CV_32FC1);
    cv::Point3f plane_3d_point;

    for (int i = 0; i < plane_3d_points.size(); i++) {
        plane_3d_point = plane_3d_points.at(i);
        points_mat->data.fl[i * 3 + 0] = plane_3d_point.x;
        points_mat->data.fl[i * 3 + 1] = plane_3d_point.y;
        points_mat->data.fl[i * 3 + 2] = plane_3d_point.z;
    }

    cvFitPlane(points_mat, plane_func);
}


int main(int argc, char** argv)
{
    namespace fs = ::boost::filesystem;

    std::string calibDir;
    bool flag_720p;

    //========= Handling Program options =========
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("calib,c", boost::program_options::value<std::string>(&calibDir)->default_value("data"), "Directory containing camera calibration files.")
        ("flag,f", boost::program_options::value<bool>(&flag_720p)->default_value(false), "flag for 720p.");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    std::cout << "calibDir = " << calibDir << std::endl;
    std::cout << "flag_720p = " << flag_720p << std::endl;

    // Check if directory containing camera calibration files exists
    if (!boost::filesystem::exists(calibDir))
    {
        std::cout << "# ERROR: Directory " << calibDir << " does not exist." << std::endl;
        return -1;
    }

    {
        // yaml-cpp demo
        boost::filesystem::path calibFilePath(calibDir);
        std::ostringstream oss;
        oss << "camchain-fisheye-pinhole-equi.yaml";
        calibFilePath /= oss.str();
        YAML::Node config = YAML::LoadFile(calibFilePath.string());
        std::cout << "config = " << config["cam0"] << std::endl;
        const std::string camera_model = config["cam0"]["camera_model"].as<std::string>();
        std::vector<double>  distortion_coeffs = config["cam0"]["distortion_coeffs"].as<std::vector<double>>();
        const std::string distortion_model = config["cam0"]["distortion_model"].as<std::string>();
        std::vector<double>  intrinsics = config["cam0"]["intrinsics"].as<std::vector<double>>();
        std::vector<int>  resolution = config["cam0"]["resolution"].as<std::vector<int>>();

        std::cout << "distortion_coeffs = " << distortion_coeffs[0] << std::endl;
        std::cout << "intrinsics = " << intrinsics[0] << std::endl;

        // save opencv yaml file
        {
            boost::filesystem::path calibFilePath(calibDir);
            std::ostringstream oss;
            oss << "camera_calib_equi_opencv.yaml";
            calibFilePath /= oss.str();
            cv::FileStorage fs(calibFilePath.string(), cv::FileStorage::WRITE);
            fs << "model_type" << "KANNALA_BRANDT";
            fs << "camera_name" << "rs_camera";
            fs << "image_width" << resolution[0];
            fs << "image_height" << resolution[1];
            // projection: fx, fy, cx, cy
            fs << "projection_parameters";
            fs << "{" << "k2" << distortion_coeffs[0]
                    << "k3" << distortion_coeffs[1]
                    << "k4" << distortion_coeffs[2]
                    << "k5" << distortion_coeffs[3]
                    << "mu" << intrinsics[0]
                    << "mv" << intrinsics[1]
                    << "u0" << intrinsics[2]
                    << "v0" << intrinsics[3] << "}";
            fs.release();
        }
    }

    // read camera params
    camodocal::CameraPtr camera_ori;
    {
        boost::filesystem::path calibFilePath(calibDir);

        std::ostringstream oss;
        oss << "camera_calib_equi_opencv.yaml";
        calibFilePath /= oss.str();

        camera_ori = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calibFilePath.string());
        if (camera_ori.get() == 0)
        {
            std::cout << "# ERROR: Unable to read calibration file: " << calibFilePath.string() << std::endl;

            return -1;
        }
        std::cout << "# INFO: camera->cameraName() = " << camera_ori->cameraName() << std::endl;
    }

    switch (camera_ori->getParameters().modelType())
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

    // do image undistortion
    Eigen::Matrix3d cameramatrix_input;
    Eigen::Matrix3d cameramatrix_output;
    cv::Mat image_ori, image_undist;
    {
        // read image
        boost::filesystem::path imgFilePath(calibDir);
        std::ostringstream oss;
        oss << "ext_calib.bmp";
        imgFilePath /= oss.str();
        image_ori = cv::imread(imgFilePath.string());
        cv::Mat ori_show_img;
        cv::resize(image_ori, ori_show_img, cv::Size(1280, 720));
        cv::imshow("image_ori", ori_show_img);
        cv::waitKey(10);

        cv::Mat map1, map2;
        double empty_pixels;
        camera_ori->initUndistortMap(map1, map2,
                                     cameramatrix_input,
                                     cameramatrix_output,
                                     empty_pixels,
                                     1.5);
        camera_ori->undistortImage(image_ori,
                                   image_undist,
                                   map1, map2,
                                   empty_pixels);

        // save result image
        boost::filesystem::path resultimgFilePath(calibDir);
        std::string result_file = resultimgFilePath.string() + "/image_undistorted.bmp";
        cv::imwrite(result_file, image_undist);

        // save ideal pinhole camera to yaml
        {
            boost::filesystem::path calibFilePath(calibDir);
            std::ostringstream oss;
            oss << "camera_calib_ideal_pinhole_opencv.yaml";
            calibFilePath /= oss.str();
            cv::FileStorage fs(calibFilePath.string(), cv::FileStorage::WRITE);
            fs << "model_type" << "PINHOLE";
            fs << "camera_name" << "ideal_pinhole";
            fs << "image_width" << image_undist.size().width;
            fs << "image_height" << image_undist.size().height;
            // radial distortion: k1, k2
            // tangential distortion: p1, p2
            fs << "distortion_parameters";
            fs << "{" << "k1" << 0
                    << "k2" << 0
                    << "p1" << 0
                    << "p2" << 0 << "}";
            // projection: fx, fy, cx, cy
            fs << "projection_parameters";
            fs << "{" << "fx" << cameramatrix_output(0, 0)
                    << "fy" << cameramatrix_output(1, 1)
                    << "cx" << cameramatrix_output(0, 2)
                    << "cy" << cameramatrix_output(1, 2) << "}";
            fs.release();
        }
    }

    // now should construct ideal pinhole camera without distortion
    camodocal::PinholeCameraPtr camera_ideal(new camodocal::PinholeCamera);
    {
        boost::filesystem::path calibFilePath(calibDir);
        std::ostringstream oss;
        oss << "camera_calib_ideal_pinhole_opencv.yaml";
        calibFilePath /= oss.str();
        camodocal::PinholeCamera::Parameters params = camera_ideal->getParameters();
        params.readFromYamlFile(calibFilePath.string());
        camera_ideal->setParameters(params);
    }

    // read image-target points
    std::vector<imgPoint> imgpoint_vec;
    std::vector<tagPoint> tagpoint_vec;
    std::vector<cv::Point2f> m_imagePoints_dist;
    std::vector<cv::Point2f> m_imagePoints_undist;
    std::vector<cv::Point3d> m_scenePoints;
    std::vector<cv::Point3f> m_scenePoints_final;
    cv::Point3d start_point;    // original point
    Eigen::Matrix3d R_global_local;
    {
        boost::filesystem::path imgtagpointsFilePath(calibDir);
        std::ostringstream oss_imgtag_pairpoints;
        oss_imgtag_pairpoints << "image_target_pairpoints.txt";
        imgtagpointsFilePath /= oss_imgtag_pairpoints.str();
        std::ifstream imgtagpoints_file(imgtagpointsFilePath.c_str());
        std::cout << "image target_points path : " << imgtagpointsFilePath.c_str() << std::endl;
        if (!imgtagpoints_file.is_open())
        {
            printf("cannot find file %s\n", imgtagpointsFilePath.c_str());
            return -1;
        }

        imagetargetpoints_read(imgtagpoints_file, imgpoint_vec, tagpoint_vec, R_global_local);

        std::cout << "imgpoint_vec.size() = " << imgpoint_vec.size() << std::endl;
        std::cout << "tagpoint_vec.size() = " << tagpoint_vec.size() << std::endl;

        for (auto it = imgpoint_vec.begin(); it != imgpoint_vec.end(); it++) {
            m_imagePoints_dist.push_back(cv::Point2f(it->u, it->v));
            // undist
            Eigen::Vector2d pixel_dist(it->u, it->v);
            Eigen::Vector2d pixel_undist;
            camera_ori->pixel_undistortion(cameramatrix_input,
                                           cameramatrix_output,
                                           pixel_dist,
                                           pixel_undist);
            std::cout << "pixel_undist = " << pixel_undist.transpose() << std::endl;
            m_imagePoints_undist.push_back(cv::Point2f(pixel_undist.x(), pixel_undist.y()));
        }

        for (auto it = tagpoint_vec.begin(); it != tagpoint_vec.end(); it++) {
            cv::Point3d scenePoint;
            scenePoint.x = it->x;    // m ==> cm
            scenePoint.y = it->y;
            scenePoint.z = it->z;

            std::cout.precision(10);
            std::cout << "scenePoint = " << scenePoint << std::endl;

            m_scenePoints.push_back(scenePoint);
        }

        start_point.x = tagpoint_vec.at(0).x;
        start_point.y = tagpoint_vec.at(0).y;
        start_point.z = tagpoint_vec.at(0).z;
    }

    // read test control points
    std::vector<imgPoint> imgpoint_test_vec;
    std::vector<tagPoint> tagpoint_test_vec;
    std::vector<cv::Point2f> m_imagePoints_test_dist;
    std::vector<cv::Point3d> m_scenePoints_test;
    std::vector<cv::Point3f> m_scenePoints_test_final;
    {
        boost::filesystem::path imgtagpointsFilePath(calibDir);
        std::ostringstream oss_imgtag_pairpoints;
        oss_imgtag_pairpoints << "image_target_pairpoints_evalute.txt";
        imgtagpointsFilePath /= oss_imgtag_pairpoints.str();
        std::ifstream imgtagpoints_file(imgtagpointsFilePath.c_str());
        std::cout << "image target evalute points path : " << imgtagpointsFilePath.c_str() << std::endl;
        if (!imgtagpoints_file.is_open()) {
            printf("cannot find file %s\n", imgtagpointsFilePath.c_str());
        } else {
            imagetargetpoints_only_read(imgtagpoints_file, imgpoint_test_vec, tagpoint_test_vec);

            std::cout << "imgpoint_test_vec.size() = " << imgpoint_test_vec.size() << std::endl;
            std::cout << "tagpoint_test_vec.size() = " << tagpoint_test_vec.size() << std::endl;

            for (auto it = imgpoint_test_vec.begin(); it != imgpoint_test_vec.end(); it++) {
                m_imagePoints_test_dist.push_back(cv::Point2f(it->u, it->v));
            }

            for (auto it = tagpoint_test_vec.begin(); it != tagpoint_test_vec.end(); it++) {
                cv::Point3d scenePoint;
                scenePoint.x = it->x;    // m ==> cm
                scenePoint.y = it->y;
                scenePoint.z = it->z;

                std::cout.precision(10);
                std::cout << "scenePoint = " << scenePoint << std::endl;

                m_scenePoints_test.push_back(scenePoint);
            }

            // get relative position, frame center is the center point
            for (int i = 0; i < m_scenePoints_test.size(); ++i) {
                // std::cout << "m_scenePoints.at(" << i << ") = " << m_scenePoints.at(i) << std::endl;
                // std::cout << "start_point = " << start_point << std::endl;
                m_scenePoints_test.at(i).x -= start_point.x;
                m_scenePoints_test.at(i).y -= start_point.y;
                m_scenePoints_test.at(i).z -= start_point.z;
                // std::cout << " m_scenePoints.at(i) = " <<  m_scenePoints.at(i) << std::endl;
                cv::Point3f final_point;
                final_point.x = (float)m_scenePoints_test.at(i).x * 1000;        // m ===> mm
                final_point.y = (float)m_scenePoints_test.at(i).y * 1000;
                final_point.z = (float)m_scenePoints_test.at(i).z * 1000;
                m_scenePoints_test_final.push_back(final_point);
                std::cout << "m_scenePoints_test_final.at(" << i << ") = " << m_scenePoints_test_final.at(i) << std::endl;
            }
        }
    }

    // get relative position, frame center is the center point
    for (int i = 0; i < m_scenePoints.size(); ++i) {
        // std::cout << "m_scenePoints.at(" << i << ") = " << m_scenePoints.at(i) << std::endl;
        // std::cout << "start_point = " << start_point << std::endl;
        m_scenePoints.at(i).x -= start_point.x;
        m_scenePoints.at(i).y -= start_point.y;
        m_scenePoints.at(i).z -= start_point.z;
        // std::cout << " m_scenePoints.at(i) = " <<  m_scenePoints.at(i) << std::endl;
        cv::Point3f final_point;
        final_point.x = (float)m_scenePoints.at(i).x * 1000;        // m ===> mm
        final_point.y = (float)m_scenePoints.at(i).y * 1000;
        final_point.z = (float)m_scenePoints.at(i).z * 1000;
        m_scenePoints_final.push_back(final_point);
        std::cout << "m_scenePoints_final.at(" << i << ") = " << m_scenePoints_final.at(i) << std::endl;
    }

    // plane fit
    float plane_func[4];
    plane_fit(m_scenePoints_final, plane_func);
    std::cout << "plane_func = " << plane_func[0] << ", " << plane_func[1] << ", " << plane_func[2] << ", " << plane_func[3] << std::endl;
    for (int i = 0; i < m_scenePoints_final.size(); i++) {
        double point_dis = point_plane_dist(m_scenePoints_final.at(i), plane_func);
        std::cout << "point_dis = " << point_dis << std::endl;
    }
    Eigen::Matrix3d R_global_local_real;
    Eigen::Vector3d vectorAfter(0, 0, 1);
    Eigen::Vector3d vectorBefore;
    vectorBefore(0) = plane_func[0];
    vectorBefore(1) = plane_func[1];
    vectorBefore(2) = plane_func[2];
    R_global_local_real = Eigen::Quaterniond::FromTwoVectors(vectorBefore, vectorAfter).toRotationMatrix();
    std::cout << "R_global_local_real = \n" << R_global_local_real << std::endl;
    R_global_local_real.setIdentity();

    // convert to local ground frame
    Eigen::Matrix4d T_ground_local2global;
    T_ground_local2global.setIdentity();
    T_ground_local2global.block<3,3>(0,0) = R_global_local_real.inverse();
    for (int i = 0; i < m_scenePoints_final.size(); ++i) {
        cv::Point3f point = m_scenePoints_final.at(i);
        Eigen::Vector4d point_homo(point.x, point.y, point.z, 1);
        point_homo = T_ground_local2global * point_homo;
        point.x = point_homo(0);
        point.y = point_homo(1);
        point.z = point_homo(2);
        std::cout << "point = " << point << std::endl;
    }

    // compute extrinsic between road ground and camera
    Eigen::Matrix3d R_cam_g_global;
    Eigen::Matrix3d R_cam_g_local;
    cv::Mat rvec;
    cv::Mat tvec;
    Eigen::Vector3d t_cam_g_global;
    Eigen::Matrix4d T_cam_ground_global;
    Eigen::Matrix4d T_cam_ground_local;
    {
        cv::Mat R0;
        // compute intrinsic camera parameters and extrinsic parameters for each of the views
        // note: rvec--rotation vector, in cv:Mat format(3 x 1)
        std::cout << "m_scenePoints.size() = " << m_scenePoints.size() << std::endl;
        std::cout << "m_imagePoints_undist.size() = " << m_imagePoints_undist.size() << std::endl;
        camera_ideal->estimateExtrinsics(m_scenePoints_final,
                                         m_imagePoints_undist,
                                         rvec,
                                         tvec);

        cv::Rodrigues(rvec, R0);
        cv::cv2eigen(R0, R_cam_g_global);

        R_cam_g_local = R_cam_g_global * R_global_local_real;

        t_cam_g_global << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

        T_cam_ground_global.setIdentity();
        T_cam_ground_global.block<3,3>(0,0) = R_cam_g_global;
        T_cam_ground_global.block<3,1>(0,3) = t_cam_g_global;

        T_cam_ground_local.setIdentity();
        T_cam_ground_local.block<3,3>(0,0) = R_cam_g_local;
        T_cam_ground_local.block<3,1>(0,3) = t_cam_g_global;
    }

    Eigen::Matrix3d R_inverse;
    R_inverse = R_cam_g_local.inverse();
    Eigen::Matrix4d T, T_out, T_ground_global_cam, T_ground_local_cam;
    T.setIdentity();
    T.block<3,3>(0,0) = R_inverse;
    T_out = T * T_cam_ground_local;
    T_ground_local_cam = T_cam_ground_local.inverse();
    T_ground_global_cam = T_cam_ground_global.inverse();

    // save extrinsic calib result
    cv::Mat extrinsic_calib_mat;
    cv::eigen2cv(T_cam_ground_local, extrinsic_calib_mat);
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
        oss << "image_undistorted.bmp";
        imgFilePath /= oss.str();
        cv::Mat image = cv::imread(imgFilePath.string());

        std::vector<cv::Point2f> estImagePoints;
        camera_ideal->projectPoints(m_scenePoints_final,
                                    rvec,
                                    tvec,
                                    estImagePoints);

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
        for (size_t i = 0; i < m_imagePoints_undist.size(); ++i)
        {
            cv::Point2f pObs = m_imagePoints_undist.at(i);
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
            // std::cout << "pixel_err_p" << i << " = " << error << std::endl;
        }

        // plot origin point
        std::vector<cv::Point2f> estImagePoint_origin_vec;
        std::vector<cv::Point3f> targetPoint_origin_vec;
        cv::Point3f targetPoint_origin(0);
        targetPoint_origin_vec.push_back(targetPoint_origin);
        camera_ideal->projectPoints(targetPoint_origin_vec,
                                    rvec,
                                    tvec,
                                    estImagePoint_origin_vec);
        cv::circle(image,
                    cv::Point(cvRound(estImagePoint_origin_vec.at(0).x * drawMultiplier),
                              cvRound(estImagePoint_origin_vec.at(0).y * drawMultiplier)),
                    200,
                    blue,
                    2,
                    CV_AA,
                    drawShiftBits);

        erravg = errorSum / m_imagePoints_undist.size();
        std::ostringstream oss_text;
        oss_text << "Reprojection error: avg = " << erravg
                 << "   max = " << errorMax << ", at " << errorMax_pos;

        std::cout << "Reprojection error: avg = " << erravg << ", max = " << errorMax << ", at " << errorMax_pos << std::endl;

        cv::Point2f pObs_max = m_imagePoints_undist.at(errorMax_pos);
        cv::Point2f pEst_max = estImagePoints.at(errorMax_pos);
        std::cout << "pObs_max = " << pObs_max << std::endl;
        std::cout << "pEst_max = " << pEst_max << std::endl;
        cv::circle(image,
                    cv::Point(cvRound(pObs_max.x * drawMultiplier),
                                cvRound(pObs_max.y * drawMultiplier)),
                    10, 
                    green,
                    2,
                    CV_AA,
                    drawShiftBits);

        cv::circle(image,
                    cv::Point(cvRound(pEst_max.x * drawMultiplier),
                                cvRound(pEst_max.y * drawMultiplier)),
                    10,
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
        cv::resize(image, output_show_img, cv::Size(1280, 720));
        cv::imshow("output_show_img", output_show_img);
        cv::waitKey(10);
    }

    // get 3d points pos on the ground from the 2d points
    {
        // 1.get the ground plane func under camera's coordinates
        Eigen::Vector3d plane_normal_vec, t_cam_g;
        Eigen::Matrix3d R_cam_g = T_cam_ground_local.block<3,3>(0,0);      // 3rows and 3 cols matrix started from (0,0)
        t_cam_g = T_cam_ground_local.block<3,1>(0,3);
        plane_normal_vec = R_cam_g.block<3,1>(0,2);
        double d = -plane_normal_vec.transpose() * t_cam_g;
        std::cout << "plane func:\n" << "n = " << plane_normal_vec.transpose() << "\nd = " << d << std::endl;

        // call python to convert 3d points to gps points
        {
            // Py_Initialize();
            // PyRun_SimpleString("print('hello world')\n");
            // Py_Finalize();
            // system("pwd");
            // system("python3.8 ./utm2gps.py");
        }

        // conpute camera's pos under ground frame
        Eigen::Vector3d camera_point_global;
        Eigen::Vector3d camera_point = T_ground_global_cam.block<3,1>(0,3);
        camera_point /= 1000;   // mm ==> m
        camera_point_global.x() = camera_point.x() + start_point.x;
        camera_point_global.y() = camera_point.y() + start_point.y;
        camera_point_global.z() = camera_point.z() + start_point.z;
        std::cout << "camera_point = " << camera_point.transpose() << std::endl;
        std::cout << "camera_point_global = " << camera_point_global.transpose() << std::endl;

        // get 200m distance image line
        {
            Eigen::Vector3d point_left_c, point_mid_c, point_right_c;
        }

        // get all 3d points pos on the ground from the 2d points
        {
            cv::Size Mat_size(camera_ori->getParameters().imageWidth(),
                              camera_ori->getParameters().imageHeight());
            std::cout << "Mat_size = " << Mat_size << std::endl;
            cv::Mat Mat_points;
            if (flag_720p) {
                Mat_points = cv::Mat::zeros(Mat_size.height/3, Mat_size.width/3, CV_64FC(3));
            } else {
                Mat_points = cv::Mat::zeros(Mat_size.height, Mat_size.width, CV_64FC(3));
            }

            int drawShiftBits = 4;
            int drawMultiplier = 1 << drawShiftBits;
            cv::Scalar yellow(0, 255, 255);
            cv::Scalar yellow_low(0, 125, 125);

            for (int v = 0; v < Mat_size.height; ++v) {
                for (int u = 0; u < Mat_size.width; ++u) {
                    Eigen::Vector3d tagpoint_measured_c;
                    Eigen::Vector3d tagpoint_measured_g, tagpoint_measured_g_global;
                    bool plane_point_flag;
                    plane_point_flag = get3dPos_onroad_single(camera_ori,          // camera with distortion
                                                              cv::Point2f(u, v),   // distorted image point
                                                              tagpoint_measured_c,
                                                              tagpoint_measured_g,
                                                              plane_normal_vec,
                                                              d,
                                                              T_cam_ground_global);

                    if (plane_point_flag) {     // have Intersection between ray and road plane
                        tagpoint_measured_g /= 1000;    // mm ==> m
                        tagpoint_measured_g_global.x() = tagpoint_measured_g.x() + start_point.x;
                        tagpoint_measured_g_global.y() = tagpoint_measured_g.y() + start_point.y;
                        tagpoint_measured_g_global.z() = tagpoint_measured_g.z() + start_point.z;
                        Eigen::Vector2d delt_2d(tagpoint_measured_g_global.x() - camera_point_global.x(),
                                                tagpoint_measured_g_global.y() - camera_point_global.y());
                        double distance = sqrt(delt_2d.x() * delt_2d.x() +  delt_2d.y() * delt_2d.y());
                        if (distance > 200) {
                            tagpoint_measured_g_global.x() = DBL_MAX;
                            tagpoint_measured_g_global.y() = DBL_MAX;
                            tagpoint_measured_g_global.z() = DBL_MAX; // numeric_limits<double>::max()
                            cv::circle(image_ori,
                                        cv::Point(cvRound(u * drawMultiplier),
                                                    cvRound(v * drawMultiplier)),
                                        10, 
                                        yellow_low,
                                        2,
                                        CV_AA,
                                        drawShiftBits);
                        }
                    } else {    // have no Intersection between ray and road plane
                        cv::circle(image_ori,
                                    cv::Point(cvRound(u * drawMultiplier),
                                                cvRound(v * drawMultiplier)),
                                    10, 
                                    yellow,
                                    2,
                                    CV_AA,
                                    drawShiftBits);
                        tagpoint_measured_g_global.x() = DBL_MAX;
                        tagpoint_measured_g_global.y() = DBL_MAX;
                        tagpoint_measured_g_global.z() = DBL_MAX;
                    }

                    if (flag_720p) {
                        if (v % 3 == 0 && u % 3 == 0) {
                            int new_v = v / 3;
                            int new_u = u / 3;
                            for (int c = 0; c < Mat_points.channels(); ++c) {              
                                Mat_points.at<cv::Vec<double, 3> >(new_v,new_u)[c] = tagpoint_measured_g_global[c];
                            }
                        }
                    } else {
                        for (int c = 0; c < Mat_points.channels(); ++c) {              
                            Mat_points.at<cv::Vec<double, 3> >(v,u)[c] = tagpoint_measured_g_global[c];
                        }
                    }

                    if (v < 5 && u < 5) {
                        std::cout << cv::Point2f(u, v) << std::endl;
                        std::cout << "tagpoint_measured_g = " << tagpoint_measured_g.transpose() << std::endl;
                        std::cout << "tagpoint_measured_g_global = " << tagpoint_measured_g_global.transpose() << std::endl;
                        std::cout << "start_point = " << start_point << std::endl;
                    }

                    // for show
                    if (v % 100 == 0 && u == 0) {
                        std::cout << "v = " << v << std::endl;
                    }

                    // test
                    // if (u == 668 && v == 38) {
                    //     std::cout << cv::Point2f(u, v) << std::endl;
                    //     std::cout << "tagpoint_measured_g_global = " << tagpoint_measured_g_global.transpose() << std::endl;
                    // }
                }
            }
            // save result image
            boost::filesystem::path resultimgFilePath(calibDir);
            std::string result_file = resultimgFilePath.string() + "/labeled_result.bmp";
            cv::imwrite(result_file, image_ori);

            cv::Mat Mat_campoint_cv;
            Eigen::MatrixXd Mat_campoint_eigen = camera_point_global.transpose();
            eigen2cv(Mat_campoint_eigen, Mat_campoint_cv);

            // save to csv file
            std::cout << "@@@save Mat_points to csv" << std::endl;
            boost::filesystem::path imgFilePath(calibDir);
            std::ostringstream oss;
            oss << "road_cloundpoints.csv";
            imgFilePath /= oss.str();
            std::ofstream myfile;
            myfile << std::setprecision(16);
            myfile.open(imgFilePath.string().c_str());
            myfile << cv::format(Mat_campoint_cv, cv::Formatter::FMT_CSV) << std::endl;
            myfile << cv::format(Mat_points, cv::Formatter::FMT_CSV) << std::endl;
            myfile.close();
        }

        // get all 3d points pos on the ground from the pointed 2d points(test points)
        {
            std::cout << "=======================================3d-reconstruction test bein=========================================" << std::endl;
            // compute pos err
            double d_test_err_max = 0;
            double d_test_err_sum = 0;
            double d_test_err_avg = 0;
            for (int i = 0; i < m_imagePoints_test_dist.size(); ++i) {
                Eigen::Vector3d tagpoint_test_measured_c;
                Eigen::Vector3d tagpoint_test_measured_g;
                get3dPos_onroad_single(camera_ori,          // camera with distortion
                                       m_imagePoints_test_dist.at(i),   // distorted image point
                                       tagpoint_test_measured_c,
                                       tagpoint_test_measured_g,
                                       plane_normal_vec,
                                       d,
                                       T_cam_ground_global);     // only translation from cloundpoints frame

                // undist
                Eigen::Vector2d pixel_dist(m_imagePoints_test_dist.at(i).x,
                                           m_imagePoints_test_dist.at(i).y);
                Eigen::Vector2d pixel_undist;
                camera_ori->pixel_undistortion(cameramatrix_input,
                                               cameramatrix_output,
                                               pixel_dist,
                                               pixel_undist);
                // std::cout << "pixel_undist = " << pixel_undist.transpose() << std::endl;

                // compute pos err
                Eigen::Vector3d tagpoint_test_truth(m_scenePoints_test_final.at(i).x,
                                                    m_scenePoints_test_final.at(i).y,
                                                    m_scenePoints_test_final.at(i).z);
                Eigen::Vector3d tagpoint_test_truth_global(m_scenePoints_test_final.at(i).x / 1000 + start_point.x,
                                                           m_scenePoints_test_final.at(i).y / 1000 + start_point.y,
                                                           m_scenePoints_test_final.at(i).z / 1000 + start_point.z);
                double d_test_err = get_distance_2points(tagpoint_test_measured_g, tagpoint_test_truth);
                double distance_road = get_distance_2points(camera_point_global, tagpoint_test_truth_global);
                // double distance_test_z_axis = abs(tagpoint_test_measured_c.z());
                d_test_err /= 1000;
                double d_test_err_percent = 0.0;
                d_test_err_percent = 100 * d_test_err / distance_road;
                if (d_test_err_max < d_test_err) {
                    d_test_err_max = d_test_err;
                }
                d_test_err_sum += d_test_err;

                std::cout.setf(std::ios::fixed);
                std::cout << "d_test_err_p" << i << " = " << std::setprecision(3) << d_test_err
                        << ", " << std::setprecision(2) << d_test_err_percent << "%"
                        << ", distance_road = " << std::setprecision(3) << distance_road << "(m)" << std::endl;
            }
            if (m_imagePoints_test_dist.size() > 0) {
                d_test_err_avg = d_test_err_sum / m_imagePoints_test_dist.size();
                std::cout << "d_test_err_max = " << d_test_err_max << ", d_test_err_avg = " << d_test_err_avg << std::endl;
            }
            std::cout << "=======================================3d-reconstruction test end=========================================" << std::endl;

            // read image
            boost::filesystem::path imgFilePath(calibDir);
            std::ostringstream oss;
            oss << "reprojection_result.bmp";
            imgFilePath /= oss.str();
            cv::Mat image = cv::imread(imgFilePath.string());
            std::vector<cv::Point2f> estImagePoints_test;
            camera_ideal->projectPoints(m_scenePoints_test_final,
                                        rvec,
                                        tvec,
                                        estImagePoints_test);

            int drawShiftBits = 4;
            int drawMultiplier = 1 << drawShiftBits;
            cv::Scalar green(0, 255, 0);
            cv::Scalar red(0, 0, 255);
            cv::Scalar blue(255, 0, 0);
            for (size_t i = 0; i < estImagePoints_test.size(); ++i)
            {
                cv::Point2f pEst = estImagePoints_test.at(i);

                cv::circle(image,
                        cv::Point(cvRound(pEst.x * drawMultiplier),
                                    cvRound(pEst.y * drawMultiplier)),
                        2,
                        blue,
                        2,
                        CV_AA,
                        drawShiftBits);
            }
            // save result image
            boost::filesystem::path resultimgFilePath(calibDir);
            std::string result_file = resultimgFilePath.string() + "/reprojection_result.bmp";
            cv::imwrite(result_file, image);
        }

        // just for test
        {
            // 2.get 3d points pos on the ground
            // cv::Point2f m_imagePoint = m_imagePoints.at(0);
            std::vector<Eigen::Vector3d> tagpoints_measured_c;
            std::vector<Eigen::Vector3d> tagpoints_measured_g;
            get3dPos_onroad_list(camera_ideal,
                                 m_imagePoints_undist,
                                 tagpoints_measured_c,
                                 tagpoints_measured_g,
                                 plane_normal_vec,
                                 d,
                                 T_cam_ground_global);
            // 3. compare distance
            size_t f_points_length = tagpoints_measured_g.size();
            double d_0_half_meaure = get_distance_2points(tagpoints_measured_g.at(0), tagpoints_measured_g.at(f_points_length / 2 - 1));
            double d_0_last_mesure = get_distance_2points(tagpoints_measured_g.at(0), tagpoints_measured_g.at(f_points_length - 1));

            double d_0_half_truth = get_distance_2points(m_scenePoints_final.at(0), m_scenePoints_final.at(f_points_length / 2 - 1));
            double d_0_last_truth = get_distance_2points(m_scenePoints_final.at(0), m_scenePoints_final.at(f_points_length - 1));

            std::cout << "d_0_half_meaure = " << d_0_half_meaure << ", d_0_last_mesure = " << d_0_last_mesure << std::endl;
            std::cout << "d_0_half_truth = " << d_0_half_truth << ", d_0_last_truth = " << d_0_last_truth << std::endl;

            // 4. compute pos err
            std::vector<Eigen::Vector3d> tagpoints_truth;
            for (auto it = m_scenePoints_final.begin(); it != m_scenePoints_final.end(); it++) {
                tagpoints_truth.push_back(Eigen::Vector3d((*it).x, (*it).y, (*it).z));
            }
            // compute pos err
            double d_err_max = 0;
            double d_err_sum = 0;
            double d_err_avg = 0;
            for (int i = 0; i < tagpoints_truth.size(); i++) {
                Eigen::Vector3d tagpoint_measured, tagpoint_truth;
                tagpoint_measured = tagpoints_measured_g.at(i);
                tagpoint_truth = tagpoints_truth.at(i);
                Eigen::Vector3d tagpoint_test_truth_global(m_scenePoints_final.at(i).x / 1000 + start_point.x,
                                                           m_scenePoints_final.at(i).y / 1000 + start_point.y,
                                                           m_scenePoints_final.at(i).z / 1000 + start_point.z);
                double d_err = get_distance_2points(tagpoint_measured, tagpoint_truth);
                double distance_road = get_distance_2points(camera_point_global, tagpoint_test_truth_global);
                // double distance_z_axis = abs(tagpoints_measured_c.at(i).z());
                d_err /= 1000;
                double d_err_percent = 0.0;
                d_err_percent = 100 * d_err / distance_road;
                if (d_err_max < d_err) {
                    d_err_max = d_err;
                }
                d_err_sum += d_err;

                std::cout.setf(std::ios::fixed);
                std::cout << "d_err_p" << i << " = " << std::setprecision(3) << d_err
                          << ", " << std::setprecision(2) << d_err_percent << "%"
                          << ", distance_road = " << std::setprecision(3) << distance_road << "(m)" << std::endl;
            }
            d_err_avg = d_err_sum / tagpoints_truth.size();
            std::cout << "d_err_max = " << d_err_max << ", d_err_avg = " << d_err_avg << std::endl;
        }
    }

    // extrinsic result print
    {
        Eigen::Matrix3d R_g_hcam, R_cam_hcam;
        R_g_hcam << 0, 0, 1,
                    -1, 0, 0,
                    0, -1, 0;
        R_cam_hcam = R_cam_g_local * R_global_local * R_g_hcam;
        std::cout << "R_cam_g_local = \n" << R_cam_g_local << std::endl;
        std::cout << "t_cam_g_local = " << t_cam_g_global.transpose() << std::endl;
        Eigen::Vector3d eulerAngle = R_cam_hcam.eulerAngles(2,1,0);
        eulerAngle *= 180 / CV_PI;
        Eigen::Quaterniond q_cam_g(R_cam_hcam);
        std::cout << "q_cam_g: (x, y, z, w) = " << "(" << q_cam_g.x() << ", " << q_cam_g.y() << ", "
                  << q_cam_g.z() << ", " << q_cam_g.w() << ")\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "eulerAngle(ZYX,RYP) =" << eulerAngle.transpose() << "\n";

        std::cout << "T_out = \n" << T_out << std::endl;
        std::cout << "T_cam_ground_local = \n" << T_cam_ground_local << std::endl;
        std::cout << "T_ground_local_cam = \n" << T_ground_local_cam << std::endl;
        std::cout << "T_ground_global_cam = \n" << T_ground_global_cam << std::endl;
    }

    {
        Eigen::Vector3d Vz_pixel(2989, 638, 1);
        Eigen::Vector3d Vz_mm = cameramatrix_output.inverse() * Vz_pixel;
        Vz_mm.normalize();
        std::cout << "Vz_mm = " << Vz_mm.transpose() << std::endl;
        double angle_ry = atan(Vz_mm(0) / Vz_mm(2));
        double angle_rx = asin(-Vz_mm(1));
        angle_ry = angle_ry * 180 / CV_PI;
        angle_rx = angle_rx * 180 / CV_PI;
        std::cout << "angle_rx = " << angle_rx << std::endl;
        std::cout << "angle_ry = " << angle_ry << std::endl;
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

bool get3dPos_onroad_single(camodocal::CameraPtr camera,
                            const cv::Point2f &imagepoint,
                            Eigen::Vector3d &tagpoint_c,
                            Eigen::Vector3d &tagpoint_g,
                            Eigen::Vector3d plane_func_normal,
                            double plane_fund_d,
                            Eigen::Matrix4d T_cam_ground)
{
    Eigen::Vector3d P_ray;
    camera->liftProjective(Eigen::Vector2d(imagepoint.x, imagepoint.y), P_ray);

    // P_ray must be nor    milized before
    P_ray.normalize();

    // get Intersection of plane and ray
    double cos_theta = plane_func_normal.transpose() * P_ray;
    if (cos_theta >= 0) {
        return false;   // no effective Intersection point
    }
    double t = -plane_fund_d / cos_theta;
    tagpoint_c = P_ray * t;

    if (imagepoint.x == 362 && imagepoint.y == 96) {
        std::cout << "P_ray = " << P_ray.transpose() << ", t = " << t << ", cos_theta = " << cos_theta << std::endl;
    }

    if (imagepoint.x == 595 && imagepoint.y == 252) {
        std::cout << "P_ray = " << P_ray.transpose() << ", t = " << t << ", cos_theta = " << cos_theta << std::endl;
    }

    // transform to ground frame
    Eigen::Vector4d hp_c(tagpoint_c(0), tagpoint_c(1), tagpoint_c(2), 1);
    Eigen::Vector4d hp_g = T_cam_ground.inverse() * hp_c;
    hp_g = hp_g / hp_g(3);
    tagpoint_g.x() = hp_g(0);
    tagpoint_g.y() = hp_g(1);
    tagpoint_g.z() = hp_g(2);
}

void get3dPos_onroad_list(camodocal::CameraPtr camera,
                     const std::vector<cv::Point2f> &imagepoints,
                     std::vector<Eigen::Vector3d> &tagpoints_c,
                     std::vector<Eigen::Vector3d> &tagpoints_g,
                     Eigen::Vector3d plane_func_normal,
                     double plane_fund_d,
                     Eigen::Matrix4d T_cam_ground)
{
    for (auto it = imagepoints.begin(); it != imagepoints.end(); it++) {
        cv::Point2f img_p = *it;
        Eigen::Vector3d P_ray;
        camera->liftProjective(Eigen::Vector2d(img_p.x, img_p.y), P_ray);
        // get Intersection of plane and ray
        double t = -plane_fund_d / (plane_func_normal.transpose() * P_ray);
        Eigen::Vector3d f_point_c = P_ray * t;
        tagpoints_c.push_back(f_point_c);

        // transform to ground frame
        Eigen::Vector4d hp_c(f_point_c(0), f_point_c(1), f_point_c(2), 1);
        Eigen::Vector4d hp_g = T_cam_ground.inverse() * hp_c;
        hp_g = hp_g / hp_g(3);
        Eigen::Vector3d f_point_g(hp_g(0), hp_g(1), hp_g(2));
        tagpoints_g.push_back(f_point_g);
    }
}

void imagetargetpoints_read(std::ifstream &imgtagpoints_file,
                            std::vector<imgPoint> &imgpoint_vec,
                            std::vector<tagPoint> &tagpoint_vec,
                            Eigen::Matrix3d &R_global_local)
{
    std::vector<std::string> vec_target;
    std::vector<std::string> vec_imgtarget;
    std::string temp;

    while (getline(imgtagpoints_file, temp)) {
        if (temp.size() != 0) {
            if (vec_target.size() < 2) {
                vec_target.push_back(temp);
            } else {
                vec_imgtarget.push_back(temp);
            }
        }
    }

    // get angle between global x_axis and local(road)
    R_global_local = Eigen::Matrix3d::Identity();
    std::vector<tagPoint> road_x_axis_points;
    for (auto it = vec_target.begin(); it != vec_target.end(); it++) {
        std::istringstream str_row(*it);
        std::string str_data;
        int data_index = 0;
        tagPoint tagpoint;
        while (str_row >> str_data) {
            switch (data_index)
            {
            case 0:
                tagpoint.x = atof(str_data.c_str());
                break;
            case 1:
                tagpoint.y = atof(str_data.c_str());
                break;
            case 2:
                tagpoint.z = atof(str_data.c_str());
                break;
            default:
                break;
                std::cout << "err: invalid" << std::endl;
            }

            data_index++;
        }
        road_x_axis_points.push_back(tagpoint);
    }
    std::cout << "road_x_axis_points.size() = " << road_x_axis_points.size() << std::endl;
    if (road_x_axis_points.size() == 2) {
        tagPoint pt1, pt2;
        pt1 = road_x_axis_points.at(0);
        pt2 = road_x_axis_points.at(1);
        float theta;
        if (pt1.x == pt2.x) {
            theta = 90;
        } else {
            theta = atan((pt1.y - pt2.y) / (pt1.x - pt2.x));
            std::cout << "pt1.y - pt2.y = " << pt1.y - pt2.y << std::endl;
            std::cout << "pt1.x - pt2.x = " << pt1.x - pt2.x << std::endl;
            std::cout << "theta = " << theta << std::endl;
            if (theta > CV_PI) {
                theta -= 2 * CV_PI;
            } else if (theta < -CV_PI) {
                theta += 2 * CV_PI;
            }
            theta = theta * 180.0 / CV_PI;
        }
        std::cout << "angle between global x_axis and local(road) is " << theta << std::endl;
        theta = theta * CV_PI / 180;
        R_global_local << cos(theta), -sin(theta), 0,
                          sin(theta), cos(theta),  0,
                          0,          0,           1;
    }

    for (auto it = vec_imgtarget.begin(); it != vec_imgtarget.end(); it++) {
        std::istringstream str_row(*it);
        std::string str_data;
        int data_index = 0;
        imgPoint imgpoint;
        tagPoint tagpoint;
        while (str_row >> str_data) {
            switch (data_index)
            {
            case 0:
                imgpoint.u = atof(str_data.c_str());
                break;
            case 1:
                imgpoint.v = atof(str_data.c_str());
                break;
            case 2:
                tagpoint.x = atof(str_data.c_str());
                break;
            case 3:
                tagpoint.y = atof(str_data.c_str());
                break;
            case 4:
                tagpoint.z = atof(str_data.c_str());
                break;
            default:
                break;
                std::cout << "err: invalid" << std::endl;
            }

            data_index++;
        }

        imgpoint_vec.push_back(imgpoint);
        tagpoint_vec.push_back(tagpoint);
    }
}

void imagetargetpoints_only_read(std::ifstream &imgtagpoints_file,
                            std::vector<imgPoint> &imgpoint_vec,
                            std::vector<tagPoint> &tagpoint_vec)
{
    std::vector<std::string> vec;
    std::string temp;
    while (getline(imgtagpoints_file, temp)) {
        if (temp.size() != 0) {
            vec.push_back(temp);
        }
    }

    for (auto it = vec.begin(); it != vec.end(); it++)
    {
        std::istringstream str_row(*it);
        std::string str_data;
        int data_index = 0;
        imgPoint imgpoint;
        tagPoint tagpoint;
        while (str_row >> str_data) {
            switch (data_index)
            {
            case 0:
                imgpoint.u = atof(str_data.c_str());
                break;
            case 1:
                imgpoint.v = atof(str_data.c_str());
                break;
            case 2:
                tagpoint.x = atof(str_data.c_str());
                break;
            case 3:
                tagpoint.y = atof(str_data.c_str());
                break;
            case 4:
                tagpoint.z = atof(str_data.c_str());
                break;
            default:
                break;
                std::cout << "err: invalid" << std::endl;
            }

            data_index++;
        }

        imgpoint_vec.push_back(imgpoint);
        tagpoint_vec.push_back(tagpoint);
    }
}
