/** ****************************************************************************
 *  @file    FaceAlignmentPrl19.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2019/04
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignmentPrl19.hpp>
#include <trace.hpp>
#include <utils.hpp>
#include <ModernPosit.h>
#include <boost/program_options.hpp>
#include "tensorflow/cc/ops/standard_ops.h"

namespace upm {

const float BBOX_SCALE = 0.3f;
const cv::Size FACE_SIZE = cv::Size(256,256);

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceAlignmentPrl19::parseOptions
  (
  int argc,
  char **argv
  )
{
  /// Declare the supported program options
  FaceAlignment::parseOptions(argc, argv);
  namespace po = boost::program_options;
  po::options_description desc("FaceAlignmentPrl19 options");
  UPM_PRINT(desc);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceAlignmentPrl19::train
  (
  const std::vector<FaceAnnotation> &anns_train,
  const std::vector<FaceAnnotation> &anns_valid
  )
{
  /// Training CNN model
  UPM_PRINT("Training PRL19 model");
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceAlignmentPrl19::load()
{
  /// Loading CNN model
  UPM_PRINT("Loading PRL19 model");
  std::string trained_model = _path + _database + ".pb";
  tensorflow::GraphDef graph;
  tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), trained_model, &graph);
  if (not load_graph_status.ok())
    UPM_ERROR("Failed to load graph: " << trained_model);
  _session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status session_create_status = _session->Create(graph);
  if (not session_create_status.ok())
    UPM_ERROR("Failed to create session");
  if ((_database == "300w_public") or (_database == "300w_private") or (_database == "menpo") or (_database == "3dmenpo"))
  {
    DB_PARTS[FacePartLabel::leyebrow] = {1, 119, 2, 121, 3};
    DB_PARTS[FacePartLabel::reyebrow] = {4, 124, 5, 126, 6};
    DB_PARTS[FacePartLabel::leye] = {7, 138, 139, 8, 141, 142};
    DB_PARTS[FacePartLabel::reye] = {11, 144, 145, 12, 147, 148};
    DB_PARTS[FacePartLabel::nose] = {128, 129, 130, 17, 16, 133, 134, 135, 18};
    DB_PARTS[FacePartLabel::tmouth] = {20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161};
    DB_PARTS[FacePartLabel::bmouth] = {156, 157, 23, 159, 160, 168, 167, 166};
    DB_PARTS[FacePartLabel::lear] = {101, 102, 103, 104, 105, 106};
    DB_PARTS[FacePartLabel::rear] = {112, 113, 114, 115, 116, 117};
    DB_PARTS[FacePartLabel::chin] = {107, 108, 24, 110, 111};
    _cnn_landmarks = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168};
  }
  else if (_database == "cofw")
  {
    DB_PARTS[FacePartLabel::leyebrow] = {1, 101, 3, 102};
    DB_PARTS[FacePartLabel::reyebrow] = {4, 103, 6, 104};
    DB_PARTS[FacePartLabel::leye] = {7, 9, 8, 10, 105};
    DB_PARTS[FacePartLabel::reye] = {11, 13, 12, 14, 106};
    DB_PARTS[FacePartLabel::nose] = {16, 17, 18, 107};
    DB_PARTS[FacePartLabel::tmouth] = {20, 22, 21, 108};
    DB_PARTS[FacePartLabel::bmouth] = {109, 23};
    DB_PARTS[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 101, 3, 102, 4, 103, 6, 104, 7, 9, 8, 10, 105, 11, 13, 12, 14, 106, 16, 17, 18, 107, 20, 22, 21, 108, 109, 23, 24};
  }
  else if (_database == "aflw")
  {
    DB_PARTS[FacePartLabel::leyebrow] = {1, 2, 3};
    DB_PARTS[FacePartLabel::reyebrow] = {4, 5, 6};
    DB_PARTS[FacePartLabel::leye] = {7, 101, 8};
    DB_PARTS[FacePartLabel::reye] = {11, 102, 12};
    DB_PARTS[FacePartLabel::nose] = {16, 17, 18};
    DB_PARTS[FacePartLabel::tmouth] = {20, 103, 21};
    DB_PARTS[FacePartLabel::lear] = {15};
    DB_PARTS[FacePartLabel::rear] = {19};
    DB_PARTS[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 2, 3, 4, 5, 6, 7, 101, 8, 11, 102, 12, 15, 16, 17, 18, 19, 20, 103, 21, 24};
  }
  else if (_database == "wflw")
  {
    DB_PARTS[upm::FacePartLabel::leyebrow] = {1, 134, 2, 136, 3, 138, 139, 140, 141};
    DB_PARTS[upm::FacePartLabel::reyebrow] = {6, 147, 148, 149, 150, 4, 143, 5, 145};
    DB_PARTS[upm::FacePartLabel::leye] = {7, 161, 9, 163, 8, 165, 10, 167, 196};
    DB_PARTS[upm::FacePartLabel::reye] = {11, 169, 13, 171, 12, 173, 14, 175, 197};
    DB_PARTS[upm::FacePartLabel::nose] = {151, 152, 153, 17, 16, 156, 157, 158, 18};
    DB_PARTS[upm::FacePartLabel::tmouth] = {20, 177, 178, 22, 180, 181, 21, 192, 191, 190, 189, 188};
    DB_PARTS[upm::FacePartLabel::bmouth] = {187, 186, 23, 184, 183, 193, 194, 195};
    DB_PARTS[upm::FacePartLabel::lear] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110};
    DB_PARTS[upm::FacePartLabel::rear] = {122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132};
    DB_PARTS[upm::FacePartLabel::chin] = {111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121};
    _cnn_landmarks = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 138, 139, 140, 141, 4, 143, 5, 145, 6, 147, 148, 149, 150, 151, 152, 153, 17, 16, 156, 157, 158, 18, 7, 161, 9, 163, 8, 165, 10, 167, 11, 169, 13, 171, 12, 173, 14, 175, 20, 177, 178, 22, 180, 181, 21, 183, 184, 23, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197};
  }
  else if (_database == "ls3dw")
  {
    DB_PARTS[upm::FacePartLabel::leyebrow] = {1, 119, 2, 121, 3};
    DB_PARTS[upm::FacePartLabel::reyebrow] = {4, 124, 5, 126, 6};
    DB_PARTS[upm::FacePartLabel::leye] = {7, 138, 139, 8, 141, 142};
    DB_PARTS[upm::FacePartLabel::reye] = {11, 144, 145, 12, 147, 148};
    DB_PARTS[upm::FacePartLabel::nose] = {128, 129, 130, 17, 16, 133, 134, 135, 18};
    DB_PARTS[upm::FacePartLabel::tmouth] = {20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161};
    DB_PARTS[upm::FacePartLabel::bmouth] = {156, 157, 23, 159, 160, 168, 167, 166};
    DB_PARTS[upm::FacePartLabel::lear] = {101, 102, 103, 104, 105, 106};
    DB_PARTS[upm::FacePartLabel::rear] = {112, 113, 114, 115, 116, 117};
    DB_PARTS[upm::FacePartLabel::chin] = {107, 108, 24, 110, 111};
    _cnn_landmarks = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168};
  }
  else
  {
    DB_PARTS[upm::FacePartLabel::leyebrow] = {1, 2, 3};
    DB_PARTS[upm::FacePartLabel::reyebrow] = {4, 5, 6};
    DB_PARTS[upm::FacePartLabel::leye] = {7, 9, 8, 10};
    DB_PARTS[upm::FacePartLabel::reye] = {11, 13, 12, 14};
    DB_PARTS[upm::FacePartLabel::nose] = {16, 17, 18};
    DB_PARTS[upm::FacePartLabel::tmouth] = {20, 22, 21};
    DB_PARTS[upm::FacePartLabel::bmouth] = {23};
    DB_PARTS[upm::FacePartLabel::lear] = {15};
    DB_PARTS[upm::FacePartLabel::rear] = {19};
    DB_PARTS[upm::FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  }
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
tensorflow::Status
FaceAlignmentPrl19::imageToTensor
  (
  const cv::Mat &img,
  std::vector<tensorflow::Tensor>* output_tensors
  )
{
  /// Copy mat into a tensor
  tensorflow::Tensor img_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({img.rows,img.cols,img.channels()}));
  auto img_tensor_mapped = img_tensor.tensor<float,3>();
  const uchar *pixel_coordinates = img.ptr<uchar>();
  for (unsigned int i=0; i < img.rows; i++)
    for (unsigned int j=0; j < img.cols; j++)
      for (unsigned int k=0; k < img.channels(); k++)
        img_tensor_mapped(i,j,k) = pixel_coordinates[i*img.cols*img.channels() + j*img.channels() + k];

  /// The convention for image ops in TensorFlow is that all images are expected
  /// to be in batches, so that they're four-dimensional arrays with indices of
  /// [batch, height, width, channel]. Because we only have a single image, we
  /// have to add a batch dimension of 1 to the start with ExpandDims()
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto holder = tensorflow::ops::Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_FLOAT);
  auto expander = tensorflow::ops::ExpandDims(root.WithOpName("expander"), holder, 0);
  auto divider = tensorflow::ops::Div(root.WithOpName("normalized"), expander, {255.0f});

  /// This runs the GraphDef network definition that we've just constructed
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<std::pair<std::string,tensorflow::Tensor>> input_tensors = {{"input", img_tensor},};
  TF_RETURN_IF_ERROR(session->Run({input_tensors}, {"normalized"}, {}, output_tensors));
  return tensorflow::Status::OK();
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
std::vector<cv::Point2f>
FaceAlignmentPrl19::tensorToLnds
  (
  const tensorflow::Tensor &img_tensor
  )
{
  tensorflow::TTypes<float>::ConstFlat data = img_tensor.flat<float>();
  std::vector<cv::Point2f> output(static_cast<unsigned int>(img_tensor.dim_size(1)));
  for (unsigned int i=0; i < output.size(); i++)
    output[i] = cv::Point2f(data(2*i), data((2*i)+1));
  return output;
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceAlignmentPrl19::process
  (
  cv::Mat frame,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  /// Analyze each detected face
  for (FaceAnnotation &face : faces)
  {
    /// Enlarge square bounding box
    cv::Point2f shift(face.bbox.pos.width*BBOX_SCALE, face.bbox.pos.height*BBOX_SCALE);
    cv::Rect_<float> bbox_enlarged = cv::Rect_<float>(face.bbox.pos.x-shift.x, face.bbox.pos.y-shift.y, face.bbox.pos.width+(shift.x*2), face.bbox.pos.height+(shift.y*2));
    /// Squared bbox required by neural networks
    bbox_enlarged.x = bbox_enlarged.x+(bbox_enlarged.width*0.5f)-(bbox_enlarged.height*0.5f);
    bbox_enlarged.width = bbox_enlarged.height;
    /// Scale image
    cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -bbox_enlarged.x, 0, 1, -bbox_enlarged.y);
    cv::warpAffine(frame, face_translated, T, bbox_enlarged.size());
    cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << FACE_SIZE.width/bbox_enlarged.width, 0, 0, 0, FACE_SIZE.height/bbox_enlarged.height, 0);
    cv::warpAffine(face_translated, face_scaled, S, FACE_SIZE);

    /// Testing CNN model
    std::vector<tensorflow::Tensor> input_tensors;
    tensorflow::Status read_tensor_status = imageToTensor(face_scaled, &input_tensors);
    if (not read_tensor_status.ok())
      UPM_ERROR(read_tensor_status);
    const tensorflow::Tensor &input_tensor = input_tensors[0];
//    UPM_PRINT("Input size:" << input_tensor.shape().DebugString() << ", tensor type:" << input_tensor.dtype()); // 1 x 256 x 256 x 3

    std::string input_layer = "input_1:0";
    std::vector<std::string> output_layers = {"k2tfout_0:0", "k2tfout_1:0", "k2tfout_2:0"};
    std::vector<tensorflow::Tensor> output_tensors;
    tensorflow::Status run_status = _session->Run({{input_layer, input_tensor}}, output_layers, {}, &output_tensors);
    if (not run_status.ok())
      UPM_ERROR("Running model failed: " << run_status);

    /// Convert output tensor to probability maps
//    UPM_PRINT("Output size:" << output_tensors[1].shape().DebugString() << ", tensor type:" << output_tensors[1].dtype()); // 1 x L x 2
    std::vector<cv::Point2f> landmarks = tensorToLnds(output_tensors[1]);

    face.parts = FaceAnnotation().parts;
    for (const auto &db_part: DB_PARTS)
      for (int feature_idx : db_part.second)
      {
        auto found = std::find(_cnn_landmarks.begin(),_cnn_landmarks.end(),feature_idx);
        if (found == _cnn_landmarks.end())
          break;
        int pos = static_cast<int>(std::distance(_cnn_landmarks.begin(), found));
        cv::Point pt = landmarks[pos]*FACE_SIZE.width;
        FaceLandmark landmark;
        landmark.feature_idx = static_cast<unsigned int>(feature_idx);
        landmark.pos.x = pt.x * (bbox_enlarged.width/FACE_SIZE.width) + bbox_enlarged.x;
        landmark.pos.y = pt.y * (bbox_enlarged.height/FACE_SIZE.height) + bbox_enlarged.y;
        landmark.occluded = 0.0f;
        face.parts[db_part.first].landmarks.push_back(landmark);
      }
  }
};

} // namespace upm
