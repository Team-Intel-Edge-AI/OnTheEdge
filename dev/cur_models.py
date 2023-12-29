""" Common model """


class CommonModel:
    """ Common model which is the parent class of the utilized AI models. """
    def __init__(self):
        pass
    
    def run(self):
        """ Function docstring """
        print("running model...")


class FaceRecognitionModel(CommonModel):
    """ Face recognition AI model """
    def __init__(self):
        pass


class GestureRecognitionModel(CommonModel):
    """ Gesture recognition AI model """
    def __init__(self):
        pass


class EmotionRecognitionModel(CommonModel):
    """ Emotion recognition AI model """
    def __init__(self):
        pass
