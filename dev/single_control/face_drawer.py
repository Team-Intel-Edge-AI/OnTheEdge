"""
Face drawer class.
"""

import cv2


class FaceDrawer:
    """
    Draw over a given image of a human face.
    """
    @staticmethod
    def draw_face(frame, xmin, ymin, xmax, ymax, text,
                  landmarks, output_transform, emotion_image):
        """
        Draw emoticons over a detected human face.
        인식한 얼굴 위에 이모티콘 이미지를 붙인다.
        """
        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale((xmax - xmin) * point[0])
            y = ymin + output_transform.scale((ymax - ymin) * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)

        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        #cv2.rectangle(frame, (xmin, ymin),
         #             (xmin + textsize[0], ymin - textsize[1]),
          #            (255, 255, 255), cv2.FILLED)
        #cv2.putText(frame, text, (xmin, ymin),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # 감정 이미지를 얼굴 영역의 높이에 맞추어 크기 조절
        emotion_image_resized = cv2.resize(emotion_image,
                                           (int(xmax - xmin),
                                            int(ymax - ymin)))

        # 이미지 합성
        frame = FaceDrawer.overlay_transparent(frame, emotion_image_resized,
                                               x=xmin, y=ymin)

        # 이모티콘을 붙인 얼굴 프레임
        return frame

    # 투명한 배경을 그대로 이용하기 위해서 알파값을 살려 투명한 부분에 오버레이되게 한다.
    @staticmethod
    def overlay_transparent(background, overlay, x, y):
        """
        Overlay a transparent image onto a background.

        Parameters:
        - background: Background image.
        - overlay: Overlay image with transparency (RGBA format).
        - x, y: Coordinates to place the overlay image.

        Returns:
        - Resulting image after overlay.
        """
        background = background.copy()

        # Extract the alpha channel from the overlay image
        overlay_alpha = overlay[:, :, 3] / 255.0

        # Calculate the inverse of the alpha channel
        background_alpha = 1.0 - overlay_alpha

        # Calculate the weighted sum of the background and overlay images
        for c in range(0, 3):
            background[y:y + overlay.shape[0], x:x + overlay.shape[1], c] = \
                (background_alpha * background[y:y + overlay.shape[0],
                                               x:x + overlay.shape[1], c] +
                 overlay_alpha * overlay[:, :, c])

        return background
