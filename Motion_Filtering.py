import cv2 as cv

# Chargement de la vidéo
video = cv.VideoCapture('C:\\Users\\hotsa\\Downloads\\people.mp4')
# Création du soustracteur de fond
subtractor = cv.createBackgroundSubtractorMOG2(300, varThreshold=30)

# Initialize HOG people detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

while True:

    ret, frame = video.read()
    print(ret)
    if ret: 
        # Application du soustracteur de fond
        mask = subtractor.apply(frame)# learningRate

        # Nettoyage du masque avec un filtre morphologique
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Detect people in the frame (static or moving)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
        # Count people
        people_count = len(boxes)
        # Draw rectangles and count
        for (x, y, w, h) in boxes:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display count on the frame
        cv.putText(frame, f'People count: {people_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('people_detected', frame)

        # Sortie avec touche 'x' ou 'X'
        if cv.waitKey(5) == ord('x'):
            break
    else:
        # Rejouer la vidéo depuis le début
        video = cv.VideoCapture('C:\\Users\\hotsa\\Downloads\\people.mp4')

# Libération des ressources
cv.destroyAllWindows()
video.release()
