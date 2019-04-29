/** ****************************************************************************
 *  @file    FaceAlignmentPrl19.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2019/04
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_ALIGNMENT_PRL19_HPP
#define FACE_ALIGNMENT_PRL19_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignment.hpp>
#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"

namespace upm {

/** ****************************************************************************
 * @class FaceAlignmentPrl19
 * @brief Class used for facial feature point detection.
 ******************************************************************************/
class FaceAlignmentPrl19: public FaceAlignment
{
public:
  FaceAlignmentPrl19(std::string path) : _path(path) {};

  ~FaceAlignmentPrl19() {};

  void
  parseOptions
    (
    int argc,
    char **argv
    );

  void
  train
    (
    const std::vector<FaceAnnotation> &anns_train,
    const std::vector<FaceAnnotation> &anns_valid
    );

  void
  load();

  tensorflow::Status
  imageToTensor
    (
    const cv::Mat &img,
    std::vector<tensorflow::Tensor>* output_tensors
    );

  std::vector<cv::Point2f>
  tensorToLnds
    (
    const tensorflow::Tensor &img_tensor
    );

  void
  process
    (
    cv::Mat frame,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );

private:
  std::string _path;
  std::vector<unsigned int> _cnn_landmarks;
  std::unique_ptr<tensorflow::Session> _session;
};

} // namespace upm

#endif /* FACE_ALIGNMENT_PRL19_HPP */
