"""
Class to process video frames and apply face detection and recognition ML/AI models.

동영상 프레임을 처리하여 얼굴 인식 머신러닝/인공지능 모델을 적용하는 클래스.
"""

import logging as log
from openvino.runtime import Core, get_version
from f_utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
import cv2


class FrameProcessor:
    """
    Class to process video frames.
    """
    QUEUE_SIZE = 16
    blur_value = 20

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show

        log.info('OpenVINO Runtime')
        log.info('\tbuild: %s', get_version())
        core = Core()

        self.face_detector = FaceDetector(core, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(core, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from %s', args.fg)
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector
                                            if args.run_detector
                                            else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered %s identities',
                 len(self.faces_database))

    def process(self, frame):
        """
        Process a video frame.
        """
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))

        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. \
                        Will be processed only %s of %s',
                        self.QUEUE_SIZE, len(rois))
            rois = rois[:self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame,
                                                                rois,
                                                                landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check prevents saving half-images in image boundaries
                if rois[i].position[0] == 0.0 or \
                    rois[i].position[1] == 0.0 or \
                        (rois[i].position[0] + rois[i].size[0] >
                         orig_image.shape[1]) or \
                        (rois[i].position[1] + rois[i].size[1] >
                         orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    ident = self.faces_database.dump_faces(crop_image,
                                                           ace_identities[i].
                                                           descriptor,
                                                           name)
                    face_identities[i].id = ident

        return [rois, landmarks, face_identities]
