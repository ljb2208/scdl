import cv2

def convert_to_colormap(imagefile):
    im_gray = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_TURBO)

    return im_color

