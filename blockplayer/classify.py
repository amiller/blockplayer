from kontort import DepthPredict
import cPickle as pickle


def load_classifier():
    """
    Load the trained classifier model from the data directory
    """
    global classifier
    with open('data/rfc_model.pkl','r') as f:
        model_ser = pickle.load(f)
    classifier = DepthPredict(model_ser)

if not 'classifier' in globals():
    classifier = None
    load_classifier();


def predict(depth, mask):
    """
    Apply the classifier to the depth image. The classifier returns:
        0 for block,
        1 for hand.
    Note that there's no need for the classifier to distinguish between
    the background and the hand - we already have a mask we can
    use to ignore the background pixels.
    """

    # This is a bit wasteful because it classifies all the pixels, including
    # the ones we'll discard anyway (i.e., ~mask)
    label_image = classifier.predict(depth)
    return label_image
