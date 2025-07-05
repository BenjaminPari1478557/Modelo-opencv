import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

webcam=cv2.VideoCapture(0)
while True:
    status, frame=webcam.read()

    if not status:
        break
    bbox, label, conf= cv.detect_common_objects(frame,confidence=0.8)

    filtered_bbox = []
    filtered_label = []
    filtered_conf = []

    #contador de personas
    person_count = 0
    tv_count = 0

    for i in range(len(label)):
        #filtrar solo 2 objetos
        if label[i] == 'person':
            person_count += 1
            filtered_bbox.append(bbox[i])
            filtered_label.append(label[i])
            filtered_conf.append(conf[i])
        elif label[i] == 'tv':
            tv_count += 1
            filtered_bbox.append(bbox[i])
            filtered_label.append(label[i])
            filtered_conf.append(conf[i])

    output= draw_bbox(frame,filtered_bbox,filtered_label,filtered_conf)

    count_text = f'person: {person_count} | tv: {tv_count}'
    cv2.putText(output, count_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)        

    cv2.imshow("Object Detection", output)

    
    
    key= cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == 32:
        if person_count >= 2:
            cv2.imwrite('mobilenet_filtro_resultado.jpg', output)

webcam.release()
cv2.destroyAllWindows()