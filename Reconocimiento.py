import cv2
import face_recognition

# Cargar las imágenes con los rostros conocidos
imagen_con_rostro1 = face_recognition.load_image_file("Nayeli.jpg")
imagen_con_rostro2 = face_recognition.load_image_file("Jungkook.jpg")
imagen_con_rostro3 = face_recognition.load_image_file("MariaBecerra.jpg")

# Codificar las imágenes de los rostros conocidos
encodings_rostro1 = face_recognition.face_encodings(imagen_con_rostro1)
encodings_rostro2 = face_recognition.face_encodings(imagen_con_rostro2)
encodings_rostro3 = face_recognition.face_encodings(imagen_con_rostro3)

rostro_conocido1 = encodings_rostro1[0] if len(encodings_rostro1) > 0 else None
rostro_conocido2 = encodings_rostro2[0] if len(encodings_rostro2) > 0 else None

if rostro_conocido1 is None or rostro_conocido2 is None:
    print("No se encontraron todos los rostros conocidos. El programa se cerrará.")
    exit()

# Inicializar la cámara
captura = cv2.VideoCapture(0)  # Usar la cámara predeterminada

# Si utilizan la cámara del celular
# ip = "http://172.29.22.62:4747/video"
# captura.open(ip)

# Cargar el clasificador Haar Cascade para la detección de rostros
clasificadorCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = captura.read()  # Captura un fotograma del flujo de video que proviene de una cámara 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el fotograma a escala de grises

    # Detección de rostros utilizando el clasificador Haar Cascade
    rostros = clasificadorCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Dibuja un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in rostros:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Encuentra las coordenadas en una imagen donde se han detectado rostros
    coordenadas_rostro = face_recognition.face_locations(frame)

    for coordenada in coordenadas_rostro:
        # Codificar el rostro detectado 
        encodings_detectados = face_recognition.face_encodings(frame, [coordenada])
        if len(encodings_detectados) > 0:
            rostroDetectado = encodings_detectados[0]

            # Compara el rostro actual con los rostros conocidos
            resultado1 = face_recognition.compare_faces([rostro_conocido1], rostroDetectado)
            resultado2 = face_recognition.compare_faces([rostro_conocido2], rostroDetectado)
        
            # Añadir el nombre sobre la imagen del rostro detectado
            if resultado1[0]:
                cv2.putText(frame, "Nayeli", (coordenada[3], coordenada[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif resultado2[0]:
                cv2.putText(frame, "Jungkook", (coordenada[3], coordenada[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Desconocido", (coordenada[3], coordenada[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Detección y reconocimiento de rostros', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
