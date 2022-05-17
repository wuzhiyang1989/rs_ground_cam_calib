#include "EquidistantCamera.h"

#include <cmath>
#include <cstdio>
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "gpl.h"


#include <algorithm>

namespace camodocal
{

EquidistantCamera::Parameters::Parameters()
 : Camera::Parameters(KANNALA_BRANDT)
 , m_k2(0.0)
 , m_k3(0.0)
 , m_k4(0.0)
 , m_k5(0.0)
 , m_mu(0.0)
 , m_mv(0.0)
 , m_u0(0.0)
 , m_v0(0.0)
{

}

EquidistantCamera::Parameters::Parameters(const std::string& cameraName,
                                          int w, int h,
                                          double k2, double k3, double k4, double k5,
                                          double mu, double mv,
                                          double u0, double v0)
 : Camera::Parameters(KANNALA_BRANDT, cameraName, w, h)
 , m_k2(k2)
 , m_k3(k3)
 , m_k4(k4)
 , m_k5(k5)
 , m_mu(mu)
 , m_mv(mv)
 , m_u0(u0)
 , m_v0(v0)
{

}

double&
EquidistantCamera::Parameters::k2(void)
{
    return m_k2;
}

double&
EquidistantCamera::Parameters::k3(void)
{
    return m_k3;
}

double&
EquidistantCamera::Parameters::k4(void)
{
    return m_k4;
}

double&
EquidistantCamera::Parameters::k5(void)
{
    return m_k5;
}

double&
EquidistantCamera::Parameters::mu(void)
{
    return m_mu;
}

double&
EquidistantCamera::Parameters::mv(void)
{
    return m_mv;
}

double&
EquidistantCamera::Parameters::u0(void)
{
    return m_u0;
}

double&
EquidistantCamera::Parameters::v0(void)
{
    return m_v0;
}

double
EquidistantCamera::Parameters::k2(void) const
{
    return m_k2;
}

double
EquidistantCamera::Parameters::k3(void) const
{
    return m_k3;
}

double
EquidistantCamera::Parameters::k4(void) const
{
    return m_k4;
}

double
EquidistantCamera::Parameters::k5(void) const
{
    return m_k5;
}

double
EquidistantCamera::Parameters::mu(void) const
{
    return m_mu;
}

double
EquidistantCamera::Parameters::mv(void) const
{
    return m_mv;
}

double
EquidistantCamera::Parameters::u0(void) const
{
    return m_u0;
}

double
EquidistantCamera::Parameters::v0(void) const
{
    return m_v0;
}

bool
EquidistantCamera::Parameters::readFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (sModelType.compare("KANNALA_BRANDT") != 0)
        {
            return false;
        }
    }

    m_modelType = KANNALA_BRANDT;
    fs["camera_name"] >> m_cameraName;
    m_imageWidth = static_cast<int>(fs["image_width"]);
    m_imageHeight = static_cast<int>(fs["image_height"]);

    cv::FileNode n = fs["projection_parameters"];
    m_k2 = static_cast<double>(n["k2"]);
    m_k3 = static_cast<double>(n["k3"]);
    m_k4 = static_cast<double>(n["k4"]);
    m_k5 = static_cast<double>(n["k5"]);
    m_mu = static_cast<double>(n["mu"]);
    m_mv = static_cast<double>(n["mv"]);
    m_u0 = static_cast<double>(n["u0"]);
    m_v0 = static_cast<double>(n["v0"]);

    return true;
}

void
EquidistantCamera::Parameters::writeToYamlFile(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "model_type" << "KANNALA_BRANDT";
    fs << "camera_name" << m_cameraName;
    fs << "image_width" << m_imageWidth;
    fs << "image_height" << m_imageHeight;

    // projection: k2, k3, k4, k5, mu, mv, u0, v0
    fs << "projection_parameters";
    fs << "{" << "k2" << m_k2
              << "k3" << m_k3
              << "k4" << m_k4
              << "k5" << m_k5
              << "mu" << m_mu
              << "mv" << m_mv
              << "u0" << m_u0
              << "v0" << m_v0 << "}";

    fs.release();
}

EquidistantCamera::Parameters&
EquidistantCamera::Parameters::operator=(const EquidistantCamera::Parameters& other)
{
    if (this != &other)
    {
        m_modelType = other.m_modelType;
        m_cameraName = other.m_cameraName;
        m_imageWidth = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;
        m_k2 = other.m_k2;
        m_k3 = other.m_k3;
        m_k4 = other.m_k4;
        m_k5 = other.m_k5;
        m_mu = other.m_mu;
        m_mv = other.m_mv;
        m_u0 = other.m_u0;
        m_v0 = other.m_v0;
    }

    return *this;
}

std::ostream&
operator<< (std::ostream& out, const EquidistantCamera::Parameters& params)
{
    out << "Camera Parameters:" << std::endl;
    out << "    model_type " << "KANNALA_BRANDT" << std::endl;
    out << "   camera_name " << params.m_cameraName << std::endl;
    out << "   image_width " << params.m_imageWidth << std::endl;
    out << "  image_height " << params.m_imageHeight << std::endl;

    // projection: k2, k3, k4, k5, mu, mv, u0, v0
    out << "Projection Parameters" << std::endl;
    out << "            k2 " << params.m_k2 << std::endl
        << "            k3 " << params.m_k3 << std::endl
        << "            k4 " << params.m_k4 << std::endl
        << "            k5 " << params.m_k5 << std::endl
        << "            mu " << params.m_mu << std::endl
        << "            mv " << params.m_mv << std::endl
        << "            u0 " << params.m_u0 << std::endl
        << "            v0 " << params.m_v0 << std::endl;

    return out;
}

EquidistantCamera::EquidistantCamera()
 : m_inv_K11(1.0)
 , m_inv_K13(0.0)
 , m_inv_K22(1.0)
 , m_inv_K23(0.0)
{

}

EquidistantCamera::EquidistantCamera(const std::string& cameraName,
                                     int imageWidth, int imageHeight,
                                     double k2, double k3, double k4, double k5,
                                     double mu, double mv,
                                     double u0, double v0)
 : mParameters(cameraName, imageWidth, imageHeight,
               k2, k3, k4, k5, mu, mv, u0, v0)
{
    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.mu();
    m_inv_K13 = -mParameters.u0() / mParameters.mu();
    m_inv_K22 = 1.0 / mParameters.mv();
    m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

EquidistantCamera::EquidistantCamera(const EquidistantCamera::Parameters& params)
 : mParameters(params)
{
    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.mu();
    m_inv_K13 = -mParameters.u0() / mParameters.mu();
    m_inv_K22 = 1.0 / mParameters.mv();
    m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

Camera::ModelType
EquidistantCamera::modelType(void) const
{
    return mParameters.modelType();
}

const std::string&
EquidistantCamera::cameraName(void) const
{
    return mParameters.cameraName();
}

int
EquidistantCamera::imageWidth(void) const
{
    return mParameters.imageWidth();
}

int
EquidistantCamera::imageHeight(void) const
{
    return mParameters.imageHeight();
}

void
EquidistantCamera::estimateIntrinsics(const cv::Size& boardSize,
                                      const std::vector< std::vector<cv::Point3f> >& objectPoints,
                                      const std::vector< std::vector<cv::Point2f> >& imagePoints)
{
    Parameters params = getParameters();

    double u0 = params.imageWidth() / 2.0;
    double v0 = params.imageHeight() / 2.0;

    double minReprojErr = std::numeric_limits<double>::max();

    std::vector<cv::Mat> rvecs, tvecs;
    rvecs.assign(objectPoints.size(), cv::Mat());
    tvecs.assign(objectPoints.size(), cv::Mat());

    params.k2() = 0.0;
    params.k3() = 0.0;
    params.k4() = 0.0;
    params.k5() = 0.0;
    params.u0() = u0;
    params.v0() = v0;

    // Initialize focal length
    // C. Hughes, P. Denny, M. Glavin, and E. Jones,
    // Equidistant Fish-Eye Calibration and Rectification by Vanishing Point
    // Extraction, PAMI 2010
    // Find circles from rows of chessboard corners, and for each pair
    // of circles, find vanishing points: v1 and v2.
    // f = ||v1 - v2|| / PI;
    double f0 = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i)
    {
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > center(boardSize.height);

        double radius[boardSize.height];
        for (int r = 0; r < boardSize.height; ++r)
        {
            std::vector<cv::Point2d> circle;
            for (int c = 0; c < boardSize.width; ++c)
            {
                circle.push_back(imagePoints.at(i).at(r * boardSize.width + c));
            }

            fitCircle(circle, center[r](0), center[r](1), radius[r]);
        }

        for (int j = 0; j < boardSize.height; ++j)
        {
            for (int k = j + 1; k < boardSize.height; ++k)
            {
                // find distance between pair of vanishing points which
                // correspond to intersection points of 2 circles
                std::vector<cv::Point2d> ipts;
                ipts = intersectCircles(center[j](0), center[j](1), radius[j],
                                        center[k](0), center[k](1), radius[k]);

                if (ipts.size() < 2)
                {
                    continue;
                }

                double f = cv::norm(ipts.at(0) - ipts.at(1)) / M_PI;

                params.mu() = f;
                params.mv() = f;

                setParameters(params);

                for (size_t l = 0; l < objectPoints.size(); ++l)
                {
                    estimateExtrinsics(objectPoints.at(l), imagePoints.at(l), rvecs.at(l), tvecs.at(l));
                }

                double reprojErr = reprojectionError(objectPoints, imagePoints, rvecs, tvecs, cv::noArray());

                if (reprojErr < minReprojErr)
                {
                    minReprojErr = reprojErr;
                    f0 = f;
                }
            }
        }
    }

    if (f0 <= 0.0 && minReprojErr >= std::numeric_limits<double>::max())
    {
        std::cout << "[" << params.cameraName() << "] "
                  << "# INFO: kannala-Brandt model fails with given data. " << std::endl;

        return;
    }

    params.mu() = f0;
    params.mv() = f0;

    setParameters(params);
}

/**
 * \brief Lifts a point from the image plane to the unit sphere
 *
 * \param p image coordinates
 * \param P coordinates of the point on the sphere
 */
void
EquidistantCamera::liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    liftProjective(p, P);
}

/** 
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void
EquidistantCamera::liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) const
{
    // Lift points to normalised plane
    #if 0
    Eigen::Vector2d p_u;
    p_u << m_inv_K11 * p(0) + m_inv_K13,
           m_inv_K22 * p(1) + m_inv_K23;

    // Obtain a projective ray
    double theta, phi;
    backprojectSymmetric(p_u, theta, phi);

    P(0) = sin(theta) * cos(phi);
    P(1) = sin(theta) * sin(phi);
    P(2) = cos(theta);
    #endif

    double fx, fy, cx, cy;
    fx = mParameters.mu();
    fy = mParameters.mv();
    cx = mParameters.u0();
    cy = mParameters.v0();

    //normalize
    Eigen::Vector2d norm_dist;
    norm_dist(0) = (p(0) - cx) / fx;
    norm_dist(1) = (p(1) - cy) / fy;

    //undistort
    Eigen::Vector2d norm_undist;
    bool ret = undistort(norm_dist, norm_undist);

    P.x() = norm_undist.x();
    P.y() = norm_undist.y();
    P.z() = 1;
}

/** 
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void
EquidistantCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const
{
    double theta = acos(P(2) / P.norm());
    double phi = atan2(P(1), P(0));

    Eigen::Vector2d p_u = r(mParameters.k2(), mParameters.k3(), mParameters.k4(), mParameters.k5(), theta) * Eigen::Vector2d(cos(phi), sin(phi));

    // Apply generalised projection matrix
    p << mParameters.mu() * p_u(0) + mParameters.u0(),
         mParameters.mv() * p_u(1) + mParameters.v0();
}

/** 
 * \brief Project a 3D point to the image plane and calculate Jacobian
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void
EquidistantCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                                Eigen::Matrix<double,2,3>& J) const
{
    double theta = acos(P(2) / P.norm());
    double phi = atan2(P(1), P(0));

    Eigen::Vector2d p_u = r(mParameters.k2(), mParameters.k3(), mParameters.k4(), mParameters.k5(), theta) * Eigen::Vector2d(cos(phi), sin(phi));

    // Apply generalised projection matrix
    p << mParameters.mu() * p_u(0) + mParameters.u0(),
         mParameters.mv() * p_u(1) + mParameters.v0();
}

/** 
 * \brief Projects an undistorted 2D point p_u to the image plane
 *
 * \param p_u 2D point coordinates
 * \return image point coordinates
 */
void
EquidistantCamera::undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const
{
//    Eigen::Vector2d p_d;
//
//    if (m_noDistortion)
//    {
//        p_d = p_u;
//    }
//    else
//    {
//        // Apply distortion
//        Eigen::Vector2d d_u;
//        distortion(p_u, d_u);
//        p_d = p_u + d_u;
//    }
//
//    // Apply generalised projection matrix
//    p << mParameters.gamma1() * p_d(0) + mParameters.u0(),
//         mParameters.gamma2() * p_d(1) + mParameters.v0();
}

void EquidistantCamera::distort(const Eigen::Vector2d &p_norm_undist,
                                   Eigen::Vector2d &p_norm_dist) const
{
    Eigen::Vector2d y = p_norm_undist;
    double r, theta, theta2, theta4, theta6, theta8, thetad, scaling;
    r = sqrt(y[0] * y[0] + y[1] * y[1]);
    theta = atan(r);
    theta2 = theta * theta;
    theta4 = theta2 * theta2;
    theta6 = theta4 * theta2;
    theta8 = theta4 * theta4;
    thetad = theta
        * (1 + mParameters.k2() * theta2 + mParameters.k3() * theta4
             + mParameters.k4() * theta6 + mParameters.k5() * theta8);
    scaling = (r > 1e-8) ? thetad / r : 1.0;
    p_norm_dist[0] = y[0] * scaling;
    p_norm_dist[1] = y[1] * scaling;
}

bool EquidistantCamera::pixel_distortion(Eigen::Matrix3d &cameramatrix_dist,
                                         Eigen::Matrix3d &cameramatrix_ideal,
                                         const Eigen::Vector2d &pixel_undist,
                                         Eigen::Vector2d &pixel_dist) const
{
    double fx, fy, cx, cy;
    fx = cameramatrix_ideal(0, 0);
    fy = cameramatrix_ideal(1, 1);
    cx = cameramatrix_ideal(0, 2);
    cy = cameramatrix_ideal(1, 2);

    // 1. get normilized image plane point
    Eigen::Vector2d imgpoint_normilized_undist;
    imgpoint_normilized_undist.x() = (pixel_undist.x() - cx) / fx;
    imgpoint_normilized_undist.y() = (pixel_undist.y() - cy) / fy;

    // 2. add distortion at normilization image plane
    Eigen::Vector2d imgpoint_normilized_dist;
    distort(imgpoint_normilized_undist, imgpoint_normilized_dist);

    // 3. do projection to get pixel point
    // use real cameramatrix
    fx = cameramatrix_dist(0, 0);
    fy = cameramatrix_dist(1, 1);
    cx = cameramatrix_dist(0, 2);
    cy = cameramatrix_dist(1, 2);
    pixel_dist[0] = fx * imgpoint_normilized_dist[0] + cx;
    pixel_dist[1] = fy * imgpoint_normilized_dist[1] + cy;

    if (pixel_dist[0] > 0 && pixel_dist[0] < mParameters.imageWidth() && pixel_dist[1] > 0
            && pixel_dist[1] < mParameters.imageHeight()) {
        return true;
    } else {
        return false;
    }
}

bool EquidistantCamera::pixel_undistortion(Eigen::Matrix3d &cameramatrix_dist,
                                Eigen::Matrix3d &cameramatrix_ideal,
                                const Eigen::Vector2d &pixel_dist,
                                Eigen::Vector2d &pixel_undist) const
{
    double fx, fy, cx, cy;
    fx = cameramatrix_dist(0, 0);
    fy = cameramatrix_dist(1, 1);
    cx = cameramatrix_dist(0, 2);
    cy = cameramatrix_dist(1, 2);

    // 1. get normilized image plane point
    Eigen::Vector2d imgpoint_normilized_dist;
    imgpoint_normilized_dist.x() = (pixel_dist.x() - cx) / fx;
    imgpoint_normilized_dist.y() = (pixel_dist.y() - cy) / fy;

    // 2. done undistortion at normilization image plane
    Eigen::Vector2d imgpoint_normilized_undist;
    undistort(imgpoint_normilized_dist, imgpoint_normilized_undist);

    // 3. do projection to get pixel point
    // use new ideal cameramatrix
    fx = cameramatrix_ideal(0, 0);
    fy = cameramatrix_ideal(1, 1);
    cx = cameramatrix_ideal(0, 2);
    cy = cameramatrix_ideal(1, 2);
    pixel_undist[0] = fx * imgpoint_normilized_undist[0] + cx;
    pixel_undist[1] = fy * imgpoint_normilized_undist[1] + cy;

    return true;
}

bool EquidistantCamera::undistort(const Eigen::Vector2d &p_norm_dist,
                                     Eigen::Vector2d &p_norm_undist) const
{
    Eigen::Vector2d y = p_norm_dist;
    double theta, theta2, theta4, theta6, theta8, thetad, scaling;

    thetad = sqrt(y[0] * y[0] + y[1] * y[1]);
    theta = thetad;  // initial guess
    for (int i = 20; i > 0; i--) {
        theta2 = theta * theta;
        theta4 = theta2 * theta2;
        theta6 = theta4 * theta2;
        theta8 = theta4 * theta4;
        theta = thetad
            / (1 + mParameters.k2() * theta2 + mParameters.k3() * theta4
                 + mParameters.k4() * theta6 + mParameters.k5() * theta8);
    }
    scaling = tan(theta) / thetad;
    p_norm_undist[0] = y[0] * scaling;
    p_norm_undist[1] = y[1] * scaling;

    return true;
}

/* porting from kalibr: aslam_cv/aslam_imgproc/include/aslam/implementation/aslamcv_helper.hpp */
void EquidistantCamera::icvGetRectangles1(Eigen::Matrix3d &cameramatrix_ori,
                                          CvSize imgSize,
                                          cv::Rect_<float>& inner,
                                          cv::Rect_<float>& outer) const
{
    const int N = 9 * 2;
    int x, y, k;
    cv::Ptr<CvMat> _pts(cvCreateMat(1, N * N, CV_32FC2));
    CvPoint2D32f *pts = (CvPoint2D32f*) (_pts->data.ptr);
    std::vector<bool> pts_success(N * N, false);

    //normalize
    double cu = cameramatrix_ori(0, 2), cv = cameramatrix_ori(1, 2);
    double fu = cameramatrix_ori(0, 0), fv = cameramatrix_ori(1, 1);

    std::cout << "cu = " << cu << ", cv = " << cv << std::endl;
    std::cout << "fu = " << fu << ", fv = " << fv << std::endl;

    for (y = k = 0; y < N; y++) {
        for (x = 0; x < N; x++) {
            // std::cout << "y = " << y << ", x = " << x << std::endl;
            Eigen::Vector2d point(x * imgSize.width / (N - 1),
                                y * imgSize.height / (N - 1));

            point(0) = (point(0) - cu) / fu;
            point(1) = (point(1) - cv) / fv;

            // std::cout << "1 point = " << point.transpose() << std::endl;
            //undistort
            Eigen::Vector2d point_undist;
            bool ret = undistort(point, point_undist);
            // std::cout << "2 point_undist = " << point_undist.transpose() << std::endl;

            if (ret) {
                pts_success[k] = true;
            }

            pts[k++] = cvPoint2D32f((float) point_undist[0], (float) point_undist[1]);
        }
    }

    float iX0 = -FLT_MAX, iX1 = FLT_MAX, iY0 = -FLT_MAX, iY1 = FLT_MAX;
    float oX0 = FLT_MAX, oX1 = -FLT_MAX, oY0 = FLT_MAX, oY1 = -FLT_MAX;
    // find the inscribed rectangle.
    // the code will likely not work with extreme rotation matrices (R) (>45%)
    for (y = 0; y < N; y++) {
        for (x = 0; x < N; x++) {
            if (pts_success[y * N + x] == false) {
            continue;
            }
            CvPoint2D32f p = pts[y * N + x];
            oX0 = MIN(oX0, p.x);
            oX1 = MAX(oX1, p.x);
            oY0 = MIN(oY0, p.y);
            oY1 = MAX(oY1, p.y);

            // if (x == 0)
            //   iX0 = MAX(iX0, p.x);
            // if (x == N - 1)
            //   iX1 = MIN(iX1, p.x);
            // if (y == 0)
            //   iY0 = MAX(iY0, p.y);
            // if (y == N - 1)
            //   iY1 = MIN(iY1, p.y);
        }
    }

    for (y = 0; y < N; y++) {
        for (x = 0; x < N / 2; x++) {
            if (pts_success[y * N + x] == true) {
                CvPoint2D32f p = pts[y * N + x];
                iX0 = MAX(iX0, p.x);
                break;
            }
        }

        for (x = N - 1; x > N / 2; x--) {
            if (pts_success[y * N + x] == true) {
                CvPoint2D32f p = pts[y * N + x];
                iX1 = MIN(iX1, p.x);
                break;
            }
        }
    }

    for (x = 0; x < N; x++) {
        for (y = 0; y < N / 2; y++) {
            if (pts_success[y * N + x] == true) {
                CvPoint2D32f p = pts[y * N + x];
                iY0 = MAX(iY0, p.y);
                break;
            }
        }

        for (y = N - 1; y > N / 2; y--) {
            if (pts_success[y * N + x] == true) {
                CvPoint2D32f p = pts[y * N + x];
                iY1 = MIN(iY1, p.y);
                break;
            }
        }
    }

    std::cout << "[oX0, oX1] = [" << oX0 << ", " << oX1 << "], [oY0, oY1] = [" << oY0 << ", " << oY1 << "]" << std::endl;
    std::cout << "[iX0, iX1] = [" << iX0 << ", " << iX1 << "], [iY0, iY1] = [" << iY0 << ", " << iY1 << "]" << std::endl;
    inner = cv::Rect_<float>(iX0, iY0, iX1 - iX0, iY1 - iY0);
    outer = cv::Rect_<float>(oX0, oY0, oX1 - oX0, oY1 - oY0);
}

void EquidistantCamera::icvGetRectangles2(Eigen::Matrix3d &cameramatrix_ori,
                                          CvSize imgSize,
                                          cv::Rect_<float>& inner,
                                          cv::Rect_<float>& outer) const
{
    //normalize
    double cu = cameramatrix_ori(0, 2), cv = cameramatrix_ori(1, 2);
    double fu = cameramatrix_ori(0, 0), fv = cameramatrix_ori(1, 1);

    std::cout << "cu = " << cu << ", cv = " << cv << std::endl;
    std::cout << "fu = " << fu << ", fv = " << fv << std::endl;

    std::cout << "imgSize.width = " << imgSize.width << std::endl;
    std::cout << "imgSize.height = " << imgSize.height << std::endl;

    // find extreme points of the normilized image plane
    // as can compute the normilized image point directly, co compute it directly
    double x_min, y_min, x_max, y_max;
    x_min = y_min = 10000;
    x_max = y_max = -10000;
    int counter = 0;
    for (size_t v = 0; v < imgSize.height - 1; ++v) {
        for (size_t u = 0; u < imgSize.width - 1; ++u) {
            // get normilized image plane point
            Eigen::Vector2d pixel_dist(u, v);
            Eigen::Vector2d imgpoint_normilized_dist;
            imgpoint_normilized_dist.x() = (pixel_dist.x() - cu) / fu;
            imgpoint_normilized_dist.y() = (pixel_dist.y() - cv) / fv;
            // do undistortion at normilization image plane
            Eigen::Vector2d imgpoint_normilized_undist;
            undistort(imgpoint_normilized_dist, imgpoint_normilized_undist);
            x_min = std::min(imgpoint_normilized_undist.x(), x_min);
            x_max = std::max(imgpoint_normilized_undist.x(), x_max);
            y_min = std::min(imgpoint_normilized_undist.y(), y_min);
            y_max = std::max(imgpoint_normilized_undist.y(), y_max);

            if (counter++ < 10 || (v == 700 && u == 1000)) {
                std::cout << "pixel_dist = " << pixel_dist << std::endl;
                std::cout << "imgpoint_normilized_dist = " << imgpoint_normilized_dist.transpose() << std::endl;
                std::cout << "imgpoint_normilized_undist = " << imgpoint_normilized_undist.transpose() << std::endl;
                std::cout << "x_min = " << x_min << ", x_max = " << x_max << std::endl;
                std::cout << "y_min = " << y_min << ", y_max = " << y_max << std::endl;

                distort(imgpoint_normilized_undist, imgpoint_normilized_dist);
                std::cout << "imgpoint_normilized_dist = " << imgpoint_normilized_dist.transpose() << std::endl;
                std::cout << "imgpoint_normilized_undist = " << imgpoint_normilized_undist.transpose() << std::endl;
                std::cout << std::endl;
            }
        }
    }
    std::cout << "x_min = " << x_min << ", x_max = " << x_max << std::endl;
    std::cout << "y_min = " << y_min << ", y_max = " << y_max << std::endl;

    // get Outer Rectangle Normalized Image Plane
    outer.x = x_min;
    outer.y = y_min;
    outer.width = x_max - x_min;
    outer.height = y_max - y_min;
    std::cout << "normal_cam outer = " << outer << std::endl;

    inner = outer;
}

bool EquidistantCamera::setOptimalOutputCameraParameters(const double scale,
                                                         cv::Size &resolution_estimat,
                                                         Eigen::Matrix3d &cameramatrix_ori,
                                                         Eigen::Matrix3d &cameramatrix_ideal) const
{
    // get original distorted image size
    cv::Size IdealPinholImageSize(
        std::ceil(mParameters.imageWidth() * scale),
        std::ceil(mParameters.imageHeight() * scale));

    cv::Size imgSize_ori(mParameters.imageWidth(), mParameters.imageHeight());

    double fx, fy, cx, cy;
    fx = mParameters.mu();
    fy = mParameters.mv();
    cx = mParameters.u0();
    cy = mParameters.v0();

    cameramatrix_ori = Eigen::Matrix3d::Identity();
    cameramatrix_ori(0, 0) = fx;
    cameramatrix_ori(1, 1) = fy;
    cameramatrix_ori(0, 2) = cx;
    cameramatrix_ori(1, 2) = cy;
    std::cout << "IdealPinholImageSize = " << IdealPinholImageSize << std::endl;

    cv::Rect_<float> inner1, outer1;
    icvGetRectangles1(cameramatrix_ori,
                      imgSize_ori,
                      inner1,
                      outer1);
    std::cout << "inner1 = " << inner1 << std::endl;
    std::cout << "outer1 = " << outer1 << std::endl;

    cv::Rect_<float> inner2, outer2;
    icvGetRectangles2(cameramatrix_ori,
                      imgSize_ori,
                      inner2,
                      outer2);
    std::cout << "inner2 = " << inner2 << std::endl;
    std::cout << "outer2 = " << outer2 << std::endl;

    cv::Size newImgSize_adj;
    newImgSize_adj.height = IdealPinholImageSize.height;
    newImgSize_adj.width =  (double)(outer2.width / outer2.height) * IdealPinholImageSize.height;
    std::cout << "IdealPinholImageSize = [" << IdealPinholImageSize.width << ", " << IdealPinholImageSize.height << "]" << std::endl;
    std::cout << "newImgSize_adj = [" << newImgSize_adj.width << ", " << newImgSize_adj.height << "]" << std::endl;
    IdealPinholImageSize = newImgSize_adj;

    double fx_idaeal = (IdealPinholImageSize.width - 1) / outer2.width;      // in fact, fx = newImgSize.width / outer.width, "-1" used to get new image width larger
    double fy_idaeal = (IdealPinholImageSize.height - 1) / outer2.height;
    double cx_idaeal = -fx_idaeal * outer2.x;
    double cy_idaeal = -fy_idaeal * outer2.y;

    cameramatrix_ideal = Eigen::Matrix3d::Identity();
    cameramatrix_ideal(0, 0) = fx_idaeal;
    cameramatrix_ideal(1, 1) = fy_idaeal;
    cameramatrix_ideal(0, 2) = cx_idaeal;
    cameramatrix_ideal(1, 2) = cy_idaeal;

    resolution_estimat.width = newImgSize_adj.width;
    resolution_estimat.height = newImgSize_adj.height;
}

#if 0
void EquidistantCamera::initUndistortMap(cv::Mat& map1, cv::Mat& map2,
                                         double &empty_pixels,
                                         double fScale) const
{
    std::cout << "@@@EquidistantCamera::initUndistortMap begin" << std::endl;
    cv::Size imageSize(mParameters.imageWidth(), mParameters.imageHeight());

    std::cout << "imageSize = " << imageSize << std::endl;

    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

            double theta, phi;
            backprojectSymmetric(Eigen::Vector2d(mx_u, my_u), theta, phi);

            Eigen::Vector3d P_3d;
            P_3d << sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta);

            Eigen::Vector2d pixel_distortion;
            spaceToPlane(P_3d, pixel_distortion);     // add distortion

            mapX.at<float>(v,u) = pixel_distortion(0);
            mapY.at<float>(v,u) = pixel_distortion(1);

            if ((pixel_distortion.x() < 0) ||
                (pixel_distortion.y() < 0) ||
                (pixel_distortion.x() >= imageSize.width) ||
                (pixel_distortion.y() >= imageSize.height)) {
                empty_pixels = true;
            }
        }
    }
    std::cout << "@@@ call cv::convertMaps" << std::endl;
    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
    std::cout << "@@@EquidistantCamera::initUndistortMap end" << std::endl;
}
#endif

void EquidistantCamera::initUndistortMap(cv::Mat& map1, cv::Mat& map2,
                                         Eigen::Matrix3d &cameramatrix_intput,
                                         Eigen::Matrix3d &cameramatrix_output,
                                         double &empty_pixels,
                                         double fScale) const
{
    Eigen::Matrix3d cameramatrix_ori, cameramatrix_ideal;
    cv::Size resolution_out;
    setOptimalOutputCameraParameters(fScale,
                                     resolution_out,
                                     cameramatrix_ori,
                                     cameramatrix_ideal);

    cameramatrix_intput = cameramatrix_ori;
    cameramatrix_output = cameramatrix_ideal;

    std::cout << "cameramatrix_ori = \n" << cameramatrix_ori << std::endl;
    std::cout << "cameramatrix_ideal = \n" << cameramatrix_ideal << std::endl;

    std::cout << "resolution_out = " << resolution_out << std::endl;

    // Initialize maps
    cv::Mat map_x_float(resolution_out, CV_32FC1);
    cv::Mat map_y_float(resolution_out, CV_32FC1);

    // Compute the remap maps
    std::cout << "========================Compute the remap maps begin=================================" << std::endl;
    for (size_t v = 0; v < resolution_out.height; ++v) {
        for (size_t u = 0; u < resolution_out.width; ++u) {
            Eigen::Vector2d pixel_location(u, v);
            Eigen::Vector2d pixel_location_dist;
            bool success = pixel_distortion(cameramatrix_ori,
                                            cameramatrix_ideal,
                                            pixel_location,
                                            pixel_location_dist);

            if (!success) {
                pixel_location_dist[0] = -1000;
                pixel_location_dist[1] = -1000;
                empty_pixels = true;
            }

            // Insert in map
            map_x_float.at<float>(v, u) =
                static_cast<float>(pixel_location_dist.x());
            map_y_float.at<float>(v, u) =
                static_cast<float>(pixel_location_dist.y());
        }
    }

    std::cout << "========================Compute the remap maps end=================================" << std::endl;

    // convert to fixed point maps for increased speed
    cv::convertMaps(map_x_float, map_y_float, map1, map2, CV_16SC2);
}


cv::Mat
EquidistantCamera::initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                           float fx, float fy,
                                           cv::Size imageSize,
                                           float cx, float cy,
                                           cv::Mat rmat) const
{
    if (imageSize == cv::Size(0, 0))
    {
        imageSize = cv::Size(mParameters.imageWidth(), mParameters.imageHeight());
    }

    cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

    Eigen::Matrix3f K_rect;

    if (cx == -1.0f && cy == -1.0f)
    {
        K_rect << fx, 0, imageSize.width / 2,
                  0, fy, imageSize.height / 2,
                  0, 0, 1;
    }
    else
    {
        K_rect << fx, 0, cx,
                  0, fy, cy,
                  0, 0, 1;
    }

    if (fx == -1.0f || fy == -1.0f)
    {
        K_rect(0,0) = mParameters.mu();
        K_rect(1,1) = mParameters.mv();
    }

    Eigen::Matrix3f K_rect_inv = K_rect.inverse();

    Eigen::Matrix3f R, R_inv;
    cv::cv2eigen(rmat, R);
    R_inv = R.inverse();

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            Eigen::Vector3f xo;
            xo << u, v, 1;

            Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

            Eigen::Vector2d p;
            spaceToPlane(uo.cast<double>(), p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);

    cv::Mat K_rect_cv;
    cv::eigen2cv(K_rect, K_rect_cv);
    return K_rect_cv;
}

int
EquidistantCamera::parameterCount(void) const
{
    return 8;
}

const EquidistantCamera::Parameters&
EquidistantCamera::getParameters(void) const
{
    return mParameters;
}

void
EquidistantCamera::setParameters(const EquidistantCamera::Parameters& parameters)
{
    mParameters = parameters;

    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.mu();
    m_inv_K13 = -mParameters.u0() / mParameters.mu();
    m_inv_K22 = 1.0 / mParameters.mv();
    m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

void
EquidistantCamera::readParameters(const std::vector<double>& parameterVec)
{
    if (parameterVec.size() != parameterCount())
    {
        return;
    }

    Parameters params = getParameters();

    params.k2() = parameterVec.at(0);
    params.k3() = parameterVec.at(1);
    params.k4() = parameterVec.at(2);
    params.k5() = parameterVec.at(3);
    params.mu() = parameterVec.at(4);
    params.mv() = parameterVec.at(5);
    params.u0() = parameterVec.at(6);
    params.v0() = parameterVec.at(7);

    setParameters(params);
}

void
EquidistantCamera::writeParameters(std::vector<double>& parameterVec) const
{
    parameterVec.resize(parameterCount());
    parameterVec.at(0) = mParameters.k2();
    parameterVec.at(1) = mParameters.k3();
    parameterVec.at(2) = mParameters.k4();
    parameterVec.at(3) = mParameters.k5();
    parameterVec.at(4) = mParameters.mu();
    parameterVec.at(5) = mParameters.mv();
    parameterVec.at(6) = mParameters.u0();
    parameterVec.at(7) = mParameters.v0();
}

void
EquidistantCamera::writeParametersToYamlFile(const std::string& filename) const
{
    mParameters.writeToYamlFile(filename);
}

std::string
EquidistantCamera::parametersToString(void) const
{
    std::ostringstream oss;
    oss << mParameters;

    return oss.str();
}

void
EquidistantCamera::fitOddPoly(const std::vector<double>& x, const std::vector<double>& y,
                              int n, std::vector<double>& coeffs) const
{
    std::vector<int> pows;
    for (int i = 1; i <= n; i += 2)
    {
        pows.push_back(i);
    }

    Eigen::MatrixXd X(x.size(), pows.size());
    Eigen::MatrixXd Y(y.size(), 1);
    for (size_t i = 0; i < x.size(); ++i)
    {
        for (size_t j = 0; j < pows.size(); ++j)
        {
            X(i,j) = pow(x.at(i), pows.at(j));
        }
        Y(i,0) = y.at(i);
    }

    Eigen::MatrixXd A = (X.transpose() * X).inverse() * X.transpose() * Y;

    coeffs.resize(A.rows());
    for (int i = 0; i < A.rows(); ++i)
    {
        coeffs.at(i) = A(i,0);
    }
}

void
EquidistantCamera::backprojectSymmetric(const Eigen::Vector2d& p_u,
                                        double& theta, double& phi) const
{
    double tol = 1e-10;
    double p_u_norm = p_u.norm();

    if (p_u_norm < 1e-10)
    {
        phi = 0.0;
    }
    else
    {
        phi = atan2(p_u(1), p_u(0));
    }

    int npow = 9;
    if (mParameters.k5() == 0.0)
    {
        npow -= 2;
    }
    if (mParameters.k4() == 0.0)
    {
        npow -= 2;
    }
    if (mParameters.k3() == 0.0)
    {
        npow -= 2;
    }
    if (mParameters.k2() == 0.0)
    {
        npow -= 2;
    }

    Eigen::MatrixXd coeffs(npow + 1, 1);
    coeffs.setZero();
    coeffs(0) = -p_u_norm;
    coeffs(1) = 1.0;

    if (npow >= 3)
    {
        coeffs(3) = mParameters.k2();
    }
    if (npow >= 5)
    {
        coeffs(5) = mParameters.k3();
    }
    if (npow >= 7)
    {
        coeffs(7) = mParameters.k4();
    }
    if (npow >= 9)
    {
        coeffs(9) = mParameters.k5();
    }

    if (npow == 1)
    {
        theta = p_u_norm;
    }
    else
    {
        // Get eigenvalues of companion matrix corresponding to polynomial.
        // Eigenvalues correspond to roots of polynomial.
        Eigen::MatrixXd A(npow, npow);
        A.setZero();
        A.block(1, 0, npow - 1, npow - 1).setIdentity();
        A.col(npow - 1) = - coeffs.block(0, 0, npow, 1) / coeffs(npow);

        Eigen::EigenSolver<Eigen::MatrixXd> es(A);
        Eigen::MatrixXcd eigval = es.eigenvalues();

        std::vector<double> thetas;
        for (int i = 0; i < eigval.rows(); ++i)
        {
            if (fabs(eigval(i).imag()) > tol)
            {
                continue;
            }

            double t = eigval(i).real();

            if (t < -tol)
            {
                continue;
            }
            else if (t < 0.0)
            {
                t = 0.0;
            }

            thetas.push_back(t);
        }

        if (thetas.empty())
        {
            theta = p_u_norm;
        }
        else
        {
            theta = *std::min_element(thetas.begin(), thetas.end());
        }
    }
}

}
