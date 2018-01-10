def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


import cv2
def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

if __name__ == '__main__':
    #Here initialize the Histogram of Oriented Gradients detector and sets the Support
    #Vector Machine detector to be the default pedestrian detector included with OpenCV.
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    #face dection xml file
    faceCascade = cv2.CascadeClassifier("face.xml")

    #sample video
    #cap = cv2.VideoCapture("Trump1.3gp")
    #stream live video
    cap = cv2.VideoCapture(0)

    #resize video size
    width = 640
    height = 320
    cap.set(3, width)
    cap.set(4, height)

    while True:
        #read stream live video
        ret, img = cap.read()
        #Convert BGR to GRAY
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #BONUS: face detection
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )
        #draw_rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #detect pedestrians in our video using the detectMultiScale  function
        found, w = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        print('%d (%d) found' % (len(found_filtered), len(found)))

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
