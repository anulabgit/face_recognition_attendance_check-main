import subprocess
import os
from tkinter import *

root = Tk()
root.title("출석 체크 프로그램")
root.geometry("320x480")

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(6, weight=1)

script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(script_dir, "excel", "출석표.xlsx")

btn1 = Button(root, text="사진 촬영", height = 3, width = 30,command = lambda:subprocess.call(["python", "face_download.py"]))
btn2 = Button(root, text="이름 설정", height = 3, width = 30,command = lambda:os.startfile("Faces"))
btn3 = Button(root, text="얼굴 학습", height = 3, width = 30,command = lambda:subprocess.call(["python", "train_v2.py"]))
btn4 = Button(root, text="엑셀 출력", height = 3, width = 30,command = lambda:subprocess.Popen([filepath], shell=True))
btn5 = Button(root, text="출석 체크", height = 3, width = 30,command = lambda:subprocess.call(["python", "detect.py"]))

btn1.grid(row=0, column=1, pady=10)
btn2.grid(row=1, column=1, pady=10)
btn3.grid(row=2, column=1, pady=10)
btn4.grid(row=4, column=1, pady=10)
btn5.grid(row=3, column=1, pady=10)

root.mainloop()
