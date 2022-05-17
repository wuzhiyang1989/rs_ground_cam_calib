#include "Camera.h"
#include "ScaramuzzaCamera.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/syscall.h>  
#define gettidv1() syscall(__NR_gettid) 

#if (0)
#define debug_enter()   (printf("[pid=%ld, main]debug: [%s, %d][begin]\n", (long int)gettidv1(), __func__, __LINE__))
#define debug_out()     (printf("[pid=%ld, main]debug: [%s, %d][end]\n", (long int)gettidv1(), __func__, __LINE__))
#define debug_mid()     (printf("[pid=%ld, main]debug: [%s, %d][middle]\n", (long int)gettidv1(), __func__, __LINE__))
#else
#define debug_enter()
#define debug_out() 
#define debug_mid() 
#endif

namespace camodocal
{

Camera::Parameters::Parameters(ModelType modelType)
 : m_modelType(modelType)
 , m_imageWidth(0)
 , m_imageHeight(0)
{
    switch (modelType)
    {
    case KANNALA_BRANDT:
        m_nIntrinsics = 8;
        break;
    case PINHOLE:
        m_nIntrinsics = 8;
        break;
    case SCARAMUZZA:
        m_nIntrinsics = SCARAMUZZA_CAMERA_NUM_PARAMS;
        break;
    case MEI:
    default:
        m_nIntrinsics = 9;
    }
}

Camera::Parameters::Parameters(ModelType modelType,
                               const std::string& cameraName,
                               int w, int h)
 : m_modelType(modelType)
 , m_cameraName(cameraName)
 , m_imageWidth(w)
 , m_imageHeight(h)
{
    switch (modelType)
    {
    case KANNALA_BRANDT:
        m_nIntrinsics = 8;
        break;
    case PINHOLE:
        m_nIntrinsics = 8;
        break;
    case SCARAMUZZA:
        m_nIntrinsics = SCARAMUZZA_CAMERA_NUM_PARAMS;
        break;
    case MEI:
    default:
        m_nIntrinsics = 9;
    }
}

Camera::ModelType&
Camera::Parameters::modelType(void)
{
    return m_modelType;
}

std::string&
Camera::Parameters::cameraName(void)
{
    return m_cameraName;
}

int&
Camera::Parameters::imageWidth(void)
{
    return m_imageWidth;
}

int&
Camera::Parameters::imageHeight(void)
{
    return m_imageHeight;
}

Camera::ModelType
Camera::Parameters::modelType(void) const
{
    return m_modelType;
}

const std::string&
Camera::Parameters::cameraName(void) const
{
    return m_cameraName;
}

int
Camera::Parameters::imageWidth(void) const
{
    return m_imageWidth;
}

int
Camera::Parameters::imageHeight(void) const
{
    return m_imageHeight;
}

int
Camera::Parameters::nIntrinsics(void) const
{
    return m_nIntrinsics;
}

cv::Mat&
Camera::mask(void)
{
    return m_mask;
}

const cv::Mat&
Camera::mask(void) const
{
    return m_mask;
}

void
Camera::estimateExtrinsics(const std::vector<cv::Point3f>& objectPoints,
                           const std::vector<cv::Point2f>& imagePoints,
                           cv::Mat& rvec, cv::Mat& tvec) const
{
    std::vector<cv::Point2f> Ms(imagePoints.size());
    for (size_t i = 0; i < Ms.size(); ++i)
    {
        Eigen::Vector3d P;
        liftProjective(Eigen::Vector2d(imagePoints.at(i).x, imagePoints.at(i).y), P);

        P /= P(2);

        Ms.at(i).x = P(0);
        Ms.at(i).y = P(1);
    }

    // assume unit focal length, zero principal point, and zero distortion
    std::vector<double> distCoeffs(4, 0.0);
    cv::solvePnP(objectPoints, Ms, cv::Mat::eye(3, 3, CV_64F), distCoeffs, rvec, tvec);
    std::cout << "rvec = " << rvec.t() << ", tvec = " << tvec.t() << std::endl;
    cv::solvePnPRansac(objectPoints, Ms, cv::Mat::eye(3, 3, CV_64F), distCoeffs, rvec, tvec, false, 100, 2);
    std::cout << "rvec = " << rvec.t() << ", tvec = " << tvec.t() << std::endl;
}

bool Camera::pixel_distortion(Eigen::Matrix3d &cameramatrix_dist,
                              Eigen::Matrix3d &cameramatrix_ideal,
                              const Eigen::Vector2d &pixel_undist,
                              Eigen::Vector2d &pixel_dist) const
{
    return true;
}

bool Camera::pixel_undistortion(Eigen::Matrix3d &cameramatrix_dist,
                                Eigen::Matrix3d &cameramatrix_ideal,
                                const Eigen::Vector2d &pixel_dist,
                                Eigen::Vector2d &pixel_undist) const
{
    return true;
}

void Camera::undistortImage(const cv::Mat& image,
                            cv::Mat &undistorted_image,
                            const cv::Mat &map_x,
                            const cv::Mat &map_y,
                            double empty_pixels) const
{
  if (empty_pixels) {
    cv::remap(image, undistorted_image, map_x, map_y, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);
  } else {
    // replicate is more efficient for gpus to calculate
    cv::remap(image, undistorted_image, map_x, map_y, cv::INTER_LINEAR,
              cv::BORDER_REPLICATE);
  }


}

double
Camera::reprojectionDist(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2) const
{
    Eigen::Vector2d p1, p2;

    spaceToPlane(P1, p1);
    spaceToPlane(P2, p2);

    return (p1 - p2).norm();
}

double
Camera::reprojectionError(const std::vector< std::vector<cv::Point3f> >& objectPoints,
                          const std::vector< std::vector<cv::Point2f> >& imagePoints,
                          const std::vector<cv::Mat>& rvecs,
                          const std::vector<cv::Mat>& tvecs,
                          cv::OutputArray _perViewErrors) const
{
    int imageCount = objectPoints.size();
    size_t pointsSoFar = 0;
    double totalErr = 0.0;

    bool computePerViewErrors = _perViewErrors.needed();
    cv::Mat perViewErrors;
    if (computePerViewErrors)
    {
        _perViewErrors.create(imageCount, 1, CV_64F);
        perViewErrors = _perViewErrors.getMat();
    }

    for (int i = 0; i < imageCount; ++i)
    {
        size_t pointCount = imagePoints.at(i).size();

        pointsSoFar += pointCount;

        std::vector<cv::Point2f> estImagePoints;
        projectPoints(objectPoints.at(i), rvecs.at(i), tvecs.at(i),
                      estImagePoints);

        double err = 0.0;
        for (size_t j = 0; j < imagePoints.at(i).size(); ++j)
        {
            err += cv::norm(imagePoints.at(i).at(j) - estImagePoints.at(j));
        }

        if (computePerViewErrors)
        {
            perViewErrors.at<double>(i) = err / pointCount;
        }

        totalErr += err;
    }

    return totalErr / pointsSoFar;
}

double
Camera::reprojectionError(const Eigen::Vector3d& P,
                          const Eigen::Quaterniond& camera_q,
                          const Eigen::Vector3d& camera_t,
                          const Eigen::Vector2d& observed_p) const
{
    Eigen::Vector3d P_cam = camera_q.toRotationMatrix() * P + camera_t;

    Eigen::Vector2d p;
    spaceToPlane(P_cam, p);

    return (p - observed_p).norm();
}

void
Camera::projectPoints(const std::vector<cv::Point3f>& objectPoints,
                      const cv::Mat& rvec,
                      const cv::Mat& tvec,
                      std::vector<cv::Point2f>& imagePoints) const
{
    //debug_enter();

    // project 3D object points to the image plane
    imagePoints.reserve(objectPoints.size());

    //double
    cv::Mat R0;
    //debug_mid();
    //std::cout << "call cv::Rodrigues() to get Rotation" << std::endl;
    cv::Rodrigues(rvec, R0);
    //std::cout << "R0 = " << R0 << std::endl;

    Eigen::MatrixXd R(3,3);
    R << R0.at<double>(0,0), R0.at<double>(0,1), R0.at<double>(0,2),
         R0.at<double>(1,0), R0.at<double>(1,1), R0.at<double>(1,2),
         R0.at<double>(2,0), R0.at<double>(2,1), R0.at<double>(2,2);

    Eigen::Vector3d t;
    t << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
    // std::cout << "R = " << R << std::endl;
    // std::cout << "t = " << t << std::endl;

    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
        const cv::Point3f& objectPoint = objectPoints.at(i);

        // Rotate and translate
        Eigen::Vector3d P;
        P << objectPoint.x, objectPoint.y, objectPoint.z;

        P = R * P + t;

        Eigen::Vector2d p;
        spaceToPlane(P, p);

        imagePoints.push_back(cv::Point2f(p(0), p(1)));
    }

    //debug_out();
}

}
