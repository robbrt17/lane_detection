#include <CameraCalibration.hpp>

void CameraCalibration::addChessboardPoints(cv::Size& chessboardSize, bool showPoints)
{
    // Chessboard points
    std::vector<cv::Point3f> objp;

    // Defining world coordinates
    for (int i = 0; i < chessboardSize.height; i++) 
    {
        for (int j = 0; j < chessboardSize.width; j++) 
        {
            objp.push_back(cv::Point3f((float)j, (float)i, 0));
        }
    }

    std::vector<cv::String> images;
    std::string path = "./images/*.jpg";

    cv::glob(path, images);

    cv::Mat frame, gray;

    std::vector<cv::Point2f> corner_pts;
    bool success;

    for (int i = 0; i < images.size(); i++) 
    {
        frame = cv::imread(images[i]);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        success = cv::findChessboardCorners(gray, chessboardSize, corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
    
        if (success)
        {
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

            cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            objectPoints.push_back(objp);
            imagePoints.push_back(corner_pts);
        }

        if (showPoints == true)
        {
            cv::drawChessboardCorners(gray, chessboardSize, corner_pts, success);
            cv::imshow("Chessboard corners", frame);
            cv::waitKey(0);
        }
    }
    
    return;
}

/**
 * Returns calibration error
*/
double CameraCalibration::calibrate(cv::Size& imageSize) 
{
    std::vector<cv::Mat> rvecs, tvecs;

    return cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distortionCoefficients, rvecs, tvecs, calibrationDoneFlag);
}

cv::Mat CameraCalibration::undistort(cv::Mat& image)
{
    cv::Mat undistortedImage;

    cv::undistort(image, undistortedImage, cameraMatrix, distortionCoefficients);

    return undistortedImage;
}

void CameraCalibration::displayUndistortedImages() 
{
    cv::Mat image, undistortedImage;
    std::vector<cv::String> images;
    std::string path = "./images/*.jpg";

    cv::glob(path, images);

    for (int i = 0; i < images.size(); i++)
    {
        image = cv::imread(images[i]);
        undistortedImage = undistort(image);
        cv::imshow("Undistorted Image", undistortedImage);
        cv::waitKey(0);
    }
}

void CameraCalibration::saveCalibration(std::string const& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distortionCoefficients;

    fs.release();
}

void CameraCalibration::start(cv::Size chessboardSize) 
{
    std::vector<cv::String> images;
    std::string path = "./images/*.jpg";
    cv::glob(path, images);   

    cv::Mat image = cv::imread(images[0]);
    addChessboardPoints(chessboardSize, true);
    double calibrationError = calibrate(image.size());
}