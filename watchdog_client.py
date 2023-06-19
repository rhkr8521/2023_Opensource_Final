### 영상 전송
import socket
import sys
import os
import glob
from datetime import datetime

import numpy as np
from keras.models import load_model
from PIL import Image
from faceidentify.SVMclassifier import model as svm

import argparse
import cv2
import os.path as osp
from detectheadposition import headpose
from gaze_tracking import GazeTracking

import time # For sleep

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	#print(face_pixels.shape)
    # transform face into one sample
    #expand dims adds a new dimension to the tensor
	samples = np.expand_dims(face_pixels, axis=0)
	#print(samples.shape)
    # make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# Print Result 
def PrintResult(x, y):
    print("###############--RESULT--#################")
    print("yellocard:", x, "/ redcard", y)
    print("###########################################")

# point can't get negative
def notnegative(x):
    if x  < 0:
        return 0
    else:
        return x

def send_image(client_socket, image_path):
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()

    # Send image size
    img_size_str = f"{len(img_data):010}"
    client_socket.sendall(img_size_str.encode())

    # Send image data
    client_socket.sendall(img_data)

# main function
def main(args):

    HOST = 'localhost'  # 와치독 서버 주소
    PORT = 9999  # 포트 번호

    # 소켓 생성
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 서버에 연결
    try:
        client_socket.connect((HOST, PORT))
    except ConnectionRefusedError:
        print("[WatchDog] Can't connect server")
        sys.exit()
    else:
        print("[WatchDog] Connected to server successfully")

    filename = args["input_file"]
    faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    model = load_model('models/facenet_keras.h5')

    if filename is None:
        isVideo = False
        webcam = cv2.VideoCapture(0)
        webcam.set(3, args['wh'][0])
        webcam.set(4, args['wh'][1])
    else:
        isVideo = True
        webcam = cv2.VideoCapture(filename)
        fps = webcam.get(cv2.webcam_PROP_FPS)
        width = int(webcam.get(cv2.webcam_PROP_FRAME_WIDTH))
        height = int(webcam.get(cv2.webcam_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"], fourcc, fps, (width, height))

    # Variable Setting
    hpd = headpose.HeadposeDetection(args["landmark_type"], args["landmark_predictor"]) #import headpose
    gaze = GazeTracking() # import gazetracking
    yellocard = 0
    redcard = 0
    tempval = 0
    image_number = 0

    img_folder = "capture_img"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    student_id = input("[WatchDog] 학번을 입력해주세요 : ")
    timee = int(input("[WatchDog] 시험 시간을 입력하세요(Minute): ")) # Input time for limit test time
    max_time_end = time.time() + (60 * timee)

    student_img_folder = img_folder+"/"+student_id
    if not os.path.exists(student_img_folder):
        os.makedirs(student_img_folder)

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d")

    while(webcam.isOpened()):

        ret, frame = webcam.read() # Read wabcam
        gaze.refresh(frame)
        frame = gaze.annotated_frame() # Mark pupil for frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE) # face structure

        # Get point from pupil
        if gaze.is_blinking():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_right():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_left():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_center():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        else:
            yellocard = yellocard + 2

        # Get redcard optiom    
        if yellocard > 50:
            yellocard = 0
            tempval = tempval + 1
            redcard = redcard + 1
            img_name = now_str + "_" + student_id + "_" + str(image_number)
            img_save = os.path.join(student_img_folder, img_name + ".jpg")
            cv2.imwrite(img_save, frame)
            image_number = image_number + 1

        # Get log consistently
        print("<< *의심수준:" , yellocard," || ", "*경고횟수:", redcard, " >>")
        
        #Detect head position
        if isVideo:
            frame, angles = hpd.process_image(frame)
            if frame is None: 
                break
            else:
                out.write(frame)
        else:
            frame, angles = hpd.process_image(frame)

            if angles is None : 
                pass
            #angles[0]>15 or angles[0] <-15 or
            else : #angles = [x,y,z] , get point from headposition
                if angles[0]>15 or angles[0] <-15 or angles[1]>15 or angles[1] <-15 or angles[2]>15 or angles[2] <-15:
                    yellocard = yellocard + 2
                else:
                    yellocard = yellocard - 1
                    yellocard = notnegative(yellocard)

        yellocard = yellocard + hpd.yello(frame)
        if yellocard <0:
            yellocard = notnegative(yellocard)
       

       # Draw a rectangle around the faces and predict the face name
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # take the face pixels from the frame
            crop_frame = frame[y:y+h, x:x+w] # turn the face pixels back into an image
            new_crop = Image.fromarray(crop_frame) # resize the image to meet the size requirment of facenet
            new_crop = new_crop.resize((160, 160)) # turn the image back into a tensor
            crop_frame = np.asarray(new_crop) # get the face embedding using the face net model
            face_embed = get_embedding(model, crop_frame) # it is a 1d array need to reshape it as a 2d tensor for svm
            face_embed = face_embed.reshape(-1, face_embed.shape[0]) # predict using our SVM model
            pred = svm.predict(face_embed) # get the prediction probabiltiy
            pred_prob = svm.predict_proba(face_embed) # pred_prob has probabilities of each class

            class_index = pred[0]
            class_probability = pred_prob[0,class_index] * 100
            text = student_id
            
            #add the name to frame but only if the pred is above a certain threshold
            if (class_probability > 70):
                cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
            cv2.imshow('WatchDog', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("관리자에 의해 시험이 강제 종료 되었습니다.")
            PrintResult(yellocard, redcard)
            break
        elif time.time() > max_time_end:
            print(timee, "분의 시험이 종료되었습니다.")
            PrintResult(yellocard, redcard)
            break

    redcard_result = str(redcard)

    client_socket.sendall(student_id.encode())
    client_socket.sendall(redcard_result.encode())

    # Get image paths
    image_paths = glob.glob(os.path.join(student_img_folder, "*.[jJ][pP]*[gG]"))
    num_images = len(image_paths)

    # Send number of images
    client_socket.sendall(str(num_images).encode())

    # Send images
    for image_path in image_paths:
        send_image(client_socket, image_path)

    client_socket.close()
    # When everything done, release the webcam
    webcam.release()
    if isVideo: 
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', default='gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())
    main(args)
