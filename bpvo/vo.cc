/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#include "bpvo/vo.h"
#include "bpvo/vo_frame.h"
#include "bpvo/vo_pose_estimator.h"
#include "bpvo/trajectory.h"
#include "bpvo/opencv.h"

namespace bpvo {

class VisualOdometry::Impl
{
 public:
  inline Impl(const Matrix33&, float, ImageSize, AlgorithmParameters p);

  inline Result addFrame(const uint8_t*, const float*);


  inline const Trajectory& trajectory() const { return _trajectory; }

  inline int numPointsAtLevel(int) const;
  inline const PointVector& pointsAtLevel(int) const;

 private:

  AlgorithmParameters _params;
  ImageSize _image_size;
  UniquePointer<VisualOdometryPoseEstimator> _vo_pose;
  UniquePointer<VisualOdometryFrame> _ref_frame;
  UniquePointer<VisualOdometryFrame> _cur_frame;
  UniquePointer<VisualOdometryFrame> _prev_frame;
  Matrix44 _T_kf;
  Trajectory _trajectory;

  KeyFramingReason shouldKeyFrame(const Matrix44&) const;
}; // VisualOdometry::Impl


VisualOdometry::VisualOdometry(const Matrix33& K, float baseline,
                               ImageSize image_size, AlgorithmParameters params)
    : _impl(new Impl(K, baseline, image_size, params)) {}

VisualOdometry::~VisualOdometry() { delete _impl; }

Result VisualOdometry::addFrame(const uint8_t* image, const float* disparity)
{
  THROW_ERROR_IF( image == nullptr || disparity == nullptr,
                 "nullptr image/disparity" );

  return _impl->addFrame(image, disparity);
}

int VisualOdometry::numPointsAtLevel(int level) const
{
  return _impl->numPointsAtLevel(level);
}

const Trajectory& VisualOdometry::trajectory() const
{
  return _impl->trajectory();
}

auto VisualOdometry::pointsAtLevel(int level) const -> const PointVector&
{
  return _impl->pointsAtLevel(level);
}


//
// implementation
//

VisualOdometry::Impl::
Impl(const Matrix33& K, float b, ImageSize s, AlgorithmParameters p)
  : _params(p)
  , _image_size(s)
  , _vo_pose(make_unique<VisualOdometryPoseEstimator>(p))
  , _T_kf(Matrix44::Identity())
{
  if(_params.numPyramidLevels <= 0)
    _params.numPyramidLevels = 1 + std::round(
        std::log2(std::min(s.rows, s.cols) / (double) p.minImageDimensionForPyramid));

  _ref_frame = make_unique<VisualOdometryFrame>(K, b, _params);
  _cur_frame = make_unique<VisualOdometryFrame>(K, b, _params);
  _prev_frame = make_unique<VisualOdometryFrame>(K, b, _params);
}

static inline Result FirstFrameResult(int n_levels)
{
  Result r;
  r.pose.setIdentity();
  r.covariance.setIdentity();
  r.optimizerStatistics.resize(n_levels);
  r.isKeyFrame = true;
  r.keyFramingReason = KeyFramingReason::kFirstFrame;

  return r;
}

inline Result VisualOdometry::Impl::
addFrame(const uint8_t* I_ptr, const float* D_ptr)
{
  const auto I = ToOpenCV(I_ptr, _image_size);
  const auto D = ToOpenCV(D_ptr, _image_size);

  _cur_frame->setData(I, D);

  if(!_ref_frame->hasTemplate())
  {
    std::swap(_ref_frame, _cur_frame);
    _ref_frame->setTemplate();
    _trajectory.push_back( _T_kf );
    return FirstFrameResult(_ref_frame->numLevels());
  }

  Matrix44 T_est;

  Result ret;
  ret.optimizerStatistics = _vo_pose->estimatePose(_ref_frame.get(), _cur_frame.get(),
                                                   _T_kf, T_est);
  ret.keyFramingReason = shouldKeyFrame(T_est);
  ret.isKeyFrame = KeyFramingReason::kNoKeyFraming != ret.keyFramingReason;

  if(!ret.isKeyFrame)
  {
    // store _cur_frame in _prev_frame as a keyframe candidate for the future
    std::swap(_prev_frame, _cur_frame);
    ret.pose = T_est * _T_kf.inverse(); // return the relative motion
    _T_kf = T_est; // accumulate the intitialization
  } else
  {
    if(_prev_frame->empty())
    {
      //
      // we were unable to obtain an intermediate frame (either because
      // keyframing is turned off, or all estimated motions thus far satisfy the
      // keyframine criteria
      //
      std::swap(_cur_frame, _ref_frame);
      _ref_frame->setTemplate();

      ret.pose = T_est *  _T_kf.inverse();
      _T_kf.setIdentity(); // reset intiialization
    } else
    {
      std::swap(_prev_frame, _ref_frame);
      _prev_frame->clear(); // no longer a suitable candidate for keyframing
      _ref_frame->setTemplate();

      //
      // re-estimate the motion, because the estimate than caused keyframing is
      // most likely bogus
      //
      Matrix44 T_init(Matrix44::Identity());
      ret.optimizerStatistics = _vo_pose->estimatePose(_ref_frame.get(), _cur_frame.get(),
                                                       T_init, T_est);
      ret.pose = T_est;
      _T_kf = T_est;
    }
  }

  _trajectory.push_back(ret.pose);
  return ret;
}

inline KeyFramingReason VisualOdometry::Impl::
shouldKeyFrame(const Matrix44& pose) const
{
  auto t_norm = pose.block<3,1>(0,3).squaredNorm();
  if(t_norm > math::sq(_params.minTranslationMagToKeyFrame))
  {
    dprintf("keyFramingReason::kLargeTranslation\n");
    return KeyFramingReason::kLargeTranslation;
  }

  auto r_norm = math::RotationMatrixToEulerAngles(pose).squaredNorm();
  if(r_norm > math::sq(_params.minRotationMagToKeyFrame))
  {
    dprintf("kLargeRotation\n");
    return KeyFramingReason::kLargeRotation;
  }

  auto frac_good = _vo_pose->getFractionOfGoodPoints(_params.goodPointThreshold);
  if(frac_good < _params.maxFractionOfGoodPointsToKeyFrame)
  {
    dprintf("kSmallFracOfGoodPoints\n");
    return KeyFramingReason::kSmallFracOfGoodPoints;
  }

  return KeyFramingReason::kNoKeyFraming;
}

inline int VisualOdometry::Impl::
numPointsAtLevel(int level) const
{
  int ret = 0;
  if(_ref_frame)
    ret = _ref_frame->getTemplateDataAtLevel(level)->numPoints();

  return ret;
}

inline auto VisualOdometry::Impl::
pointsAtLevel(int level) const -> const PointVector&
{
  THROW_ERROR_IF( _ref_frame == nullptr, "no reference frame has been set");
  return _ref_frame->getTemplateDataAtLevel(level)->points();
}

}; // bpvo


