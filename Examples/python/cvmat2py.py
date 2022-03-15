import cv2


def load_image(image):
    cv2.imshow("image in python", image)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    return "result"


def load_image_name(name):
    image = cv2.imread(name)
    cv2.imshow("image python", image)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    return "result"
