import cv2
import cv2.data

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

cap= cv2.VideoCapture(0)


while True:
    face_count=0
    ret, frame=cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    for (x, y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        
    count_text = f'rostros: {len(faces)}'
    cv2.putText(frame, count_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)    
    cv2.imshow('Detector de Rostros',frame)
    
      

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('mobilenet_filtro_resultado.jpg', frame)
        break