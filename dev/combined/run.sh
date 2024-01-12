python3 main.py \
  -i /dev/video0 \
  -m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml \
  -m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
  -m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml \
  -m_ed intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml \
  -m_gd intel/models/model.h5 \
  -m_pd intel/models/ssdlite_mobilenet_v2_fp16.xml
