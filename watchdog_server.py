import socket
import threading
from datetime import datetime
import os

# WatchDog 서버 정보
HOST = 'localhost'  # 호스트 이름
PORT = 9999         # 포트 번호

# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 서버 바인딩
server_socket.bind((HOST, PORT))

# 클라이언트 접속 대기
server_socket.listen()

img_folder = "receive_img"
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

print('[WatchDog] Server Start')

# 클라이언트로부터 데이터(학번, 등등)를 받기 위한 함수
def receive_data(client_socket, addr):
    print('[WatchDog] Connected by', addr)

    # 클라이언트가 보낸 데이터 받기
    r_student_id = client_socket.recv(1024)
    d_student_id = r_student_id.decode()

    c_student_id = d_student_id[0:10]

    if len(d_student_id) == 11:
        redcard_result = d_student_id[-1]
    elif len(d_student_id) == 12:
        redcard_result = d_student_id[-2:]
    else:
        redcard_result = d_student_id[-3:]

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d")

    if d_student_id == "":
        print('[WatchDog] Suddenly Shutdown Client IP : {}'.format(addr))
    else:
        LogFile = open("log.txt", "a")
        print('[WatchDog] Finish student ID : {} / RedCard : {} / IP : {}'.format(c_student_id, redcard_result, addr))
        print('\n[WatchDog] Finish student ID : {} / RedCard : {} / IP : {}\n'.format(c_student_id, redcard_result, addr), file=LogFile)
        LogFile.close()

        student_foldername = img_folder+"/"+c_student_id
        os.makedirs(student_foldername)

        num_images = int(client_socket.recv(1024).decode())

        for i in range(num_images):
            # Receive image size
            img_size_data = client_socket.recv(10)
            img_size = int(img_size_data.decode().strip())

            # Receive image data
            img_data = b''
            while len(img_data) < img_size:
                packet = client_socket.recv(img_size - len(img_data))
                if not packet:
                    break
                img_data += packet

            # Save image
            img_path = os.path.join(student_foldername, f'{now_str}_{c_student_id}_{i}.jpg')
            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)

    # 연결 종료
    client_socket.close()
    print('[WatchDog] Disconnected by', addr)


# 클라이언트 연결 처리 루프
while True:
    # 클라이언트 접속 수락
    client_socket, addr = server_socket.accept()

    # 클라이언트 데이터 처리를 위한 새로운 스레드 시작
    client_thread = threading.Thread(target=receive_data, args=(client_socket, addr))
    client_thread.start()
