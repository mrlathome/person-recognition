"""
Acquires data from multiple sources
"""
from matplotlib import image


def load_img(file):
    """
    Reads image from disk
    :param file: image file path
    :return: RGB image as numpy array
    """
    # Creating an empty array only to see the test fail
    # img = np.array([])
    # Load image file from disk
    img = image.imread(file)
    return img
