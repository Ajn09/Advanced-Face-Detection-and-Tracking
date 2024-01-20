import cv2
import imutils

cam = cv2.VideoCapture(0)  # cam id
# time.sleep(1)  # You can remove this line if not needed

firstframe = None
area = 500

while True:
    _, img = cam.read()  # read from the camera
    text = "Normal"
    resizeimg = imutils.resize(img, width=1000)  # resize img
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # GRAY IMAGE
    gaussianimg = cv2.GaussianBlur(grayimg, (21, 21), 0)  # smoothed img

    if firstframe is None:
        firstframe = gaussianimg  # capturing first frame
        continue

    imgdiff = cv2.absdiff(firstframe, gaussianimg)
    thresholdimg = cv2.threshold(gaussianimg, 180, 255, cv2.THRESH_BINARY)[1]  # Fixed typo guassianimg
    thresholdimg = cv2.dilate(thresholdimg, None, iterations=2)
    cnts = cv2.findContours(thresholdimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Fixed typo RETR_EXTERNAL, cv2.CHAIN.APPROX_SIMPLE
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:  # Fixed typo contoursArea
            continue

        (X, Y, W, H) = cv2.boundingRect(c)
        cv2.rectangle(img, (X, Y), (X + W, Y + H), (0, 0, 255), 2)
        text = "Moving Object Detected"

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Fixed typo FONT_SIMPLEX
    cv2.imshow("cameraFeed", img)

    key = cv2.waitKey(10)  # Fixed typo waitKet
    print(key)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
