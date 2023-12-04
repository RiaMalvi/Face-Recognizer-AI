import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
import time
from utils import append_df_to_excel

video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_roll_no = []
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
attendance_record = set([])
roll_record = {}

name_col = []
roll_no_col = []
time_col = []

df = pd.read_excel("student_db" + os.sep + "people_db.xlsx")

for key, row in df.iterrows():
    roll_no = row['roll_no']
    name = row['name']
    image_path = row['image']
    roll_record[roll_no] = name
    student_image = face_recognition.load_image_file(
        "student_db" + os.sep + image_path)
    student_face_encoding = face_recognition.face_encodings(student_image)[0]
    known_face_encodings.append(student_face_encoding)
    known_face_roll_no.append(roll_no)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                roll_no = known_face_roll_no[best_match_index]
                name = roll_record[roll_no]
                if roll_no not in attendance_record:
                    attendance_record.add(roll_no)
                    print(name, roll_no)
                    name_col.append(name)
                    roll_no_col.append(roll_no)
                    curr_time = time.localtime()
                    curr_clock = time.strftime("%H:%M:%S", curr_time)
                    time_col.append(curr_clock)

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

data = {'Name': name_col, 'Roll No.': roll_no_col, 'Time': time_col}

curr_time = time.localtime()
curr_clock = time.strftime("%c", curr_time)
# log_file_name = "demo " + curr_clock + ".xlsx"

# append_df_to_excel(log_file_name, data)

log_file_name = 'D:\\ai\\Face Recognizer AI\\Face Recognizer\\demo.xlsx'
append_df_to_excel(log_file_name, data)

video_capture.release()
cv2.destroyAllWindows()