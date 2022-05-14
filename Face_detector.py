import PySimpleGUI as sg
import cv2

layout = [
    # GUI text
    [sg.Text('Default view', key='-TEXT2-'),
     sg.Text('People in picture: 0', key='-TEXT-',
             expand_x=True, justification='center'),
     sg.Text('AI view', key='-TEXT2-')],

    # GUI image
    [sg.Image(key='-IMAGE-'), sg.Image(key='-IMAGE2-')]
]

window = sg.Window('Face Detector', layout)

# image_path = 'eyes.png'...

# Camera
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    event, values = window.read(timeout=0)
    if event == sg.WIN_CLOSED:
        break

    _,  frame = video.read()
    # AI view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=7,
        minSize=(50, 50)
    )
    # print(faces)

    # Draw a rectangle around the faces Default view
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert to img Default view
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

    # Draw a rectangle around the faces AI view
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Convert to img AI view
    imgbytes = cv2.imencode('.png', gray)[1].tobytes()
    window['-IMAGE2-'].update(data=imgbytes)

    # Count people in picture Text
    window['-TEXT-'].update(f'People in picture: {len(faces)}')


window.close()
