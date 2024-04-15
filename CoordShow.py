import cv2 as cv

def mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        xy = '%d, %d' %(x, y)
        cv.circle(img, (x,y), 1, (255, 255, 255), thickness=-1)
        cv.putText(img, xy, (x, y), cv.FONT_HERSHEY_PLAIN,
                   1.0, (255, 255, 255))
        cv.imshow('image', img)

img = cv.imread('20210421170605000Z.JPG')
cv.namedWindow('image', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
cv.imshow('image', img)
cv.setMouseCallback('image', mouse)
if cv.waitKey(0) & 0xff == 227:
    cv.destroyAllWindows()

