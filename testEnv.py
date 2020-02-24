import numpy as np
import cv2
import win32gui
import win32con
import win32ui

import re
import time
from time import sleep

import control as c
import random
import threading
import pickle

class Env:
    def __init__(self, height, width, frame_time):
        self.height = height
        self.width = width
        self.frame_time = frame_time
        self.lower_red = np.array([0, 200, 120])
        self.upper_red = np.array([10, 255, 150])
        self.lower_yellow = np.array([20, 120, 100])
        self.upper_yellow = np.array([30, 255, 255])

        self.topCropRatio = 0.16
        self.bottomCropRatio = 0.90
        
        self.round = 0
        self.isWin = False
        
        # Keyboard
        self.key_ground = [c.right,
            c.left,
            c.stay,
            c.up,
            c.left_p,
            c.right_p
        ]

        self.key_action_strings = ["right", "left", "stay", "up", "left_p", "right_p"]

        self.key_action_dicts = {
            "right": c.right,
            "left": c.left,
            "stay": c.stay,
            "up": c.up,
            "left_p": c.left_p,
            "right_p": c.right_p
        }

        #self.currentImg = np.zeros((height,width), np.uint8)
        
        self.reward = 0         #get score: 1, lose score: -1, others: 0
        self.action = 0         #right: 0, left: 1, stay: 2, up: 3, left_p: 4, right_p: 5
        self.state = np.zeros((height, width), np.uint8)        #current frame
        self._state = np.zeros((height, width), np.uint8)       #next frame

        self.threshold = 0.95
        self.polepoint = cv2.imread('img/polepoint/polepoint.png', 0)


    def FindWindow_bySearch(self, pattern):
        window_list = []
        win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), window_list)
        for each in window_list:
            if re.search(pattern, win32gui.GetWindowText(each)) is not None:
                return each


    def getWindow_W_H(self, hwnd):
        # 取得目標視窗的大小
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        width = right - left - 15
        height = bot - top - 11
        return (left, top, width, height)


    def getWindow_Img(self, hwnd):
        # 將 hwnd 換成 WindowLong
        s = win32gui.GetWindowLong(hwnd,win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, s|win32con.WS_EX_LAYERED)
        # 判斷視窗是否最小化
        show = win32gui.IsIconic(hwnd)
        # 將視窗圖層屬性改變成透明    
        # 還原視窗並拉到最前方
        # 取消最大小化動畫
        # 取得視窗寬高
        if show == 1: 
            win32gui.SystemParametersInfo(win32con.SPI_SETANIMATION, 0)
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 0, win32con.LWA_ALPHA)
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)    
            x, y, width, height = self.getWindow_W_H(hwnd)        
        # 創造輸出圖層
        hwindc = win32gui.GetWindowDC(hwnd)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        # 取得視窗寬高
        x, y, width, height = self.getWindow_W_H(hwnd)
        # 如果視窗最小化，則移到Z軸最下方
        if show == 1: win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, x, y, width, height, win32con.SWP_NOACTIVATE)
        # 複製目標圖層，貼上到 bmp
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0 , 0), (width, height), srcdc, (8, 3), win32con.SRCCOPY)
        # 將 bitmap 轉換成 np
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4) #png，具有透明度的
        # 釋放device content
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        # 還原目標屬性
        if show == 1 :
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
            win32gui.SystemParametersInfo(win32con.SPI_SETANIMATION, 1)
        # 回傳圖片
        return img


    def preprocess_img(self, resize=True, save_dir=False):
        hwnd = self.FindWindow_bySearch("咑偭忤痚")
        frame = self.getWindow_Img(hwnd)
        img = cv2.resize(frame, (self.width, self.height))

        #cv2.imwrite('pika.jpg', img)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_red = cv2.inRange(img_hsv, self.lower_red, self.upper_red)
        img_yellow = cv2.inRange(img_hsv, self.lower_yellow, self.upper_yellow)
        img_mask = img_red + img_yellow
        
        self.img = img              #origin image
        self.mask = img_mask        #masked image
        self.img_red = img_red
        self.img_yellow = img_yellow

        self.crop_img = img_mask[int(self.topCropRatio * self.height) : int(self.bottomCropRatio * self.height), :]
        
        #cv2.imshow('img', self.mask)
        #cv2.waitKey(1)
        del img
        del img_hsv
        if resize:
            state = cv2.resize(self.mask, (self.width, self.height))
            if save_dir:
                cv2.imwrite(save_dir, state)
            else:
                return state


    def get_isNewSet(self):
        #crop_img = img[y:y+h, x:x+w]
        imgMaskVal = np.sum(self.mask[:, :])
        imgRedVal = np.sum(self.img_red[:, :])
        imgYellowVal = np.sum(self.img_yellow[:, :])
        #print("imgMaskVal: %d, imgRedVal: %d, imgYellowVal %d" %(imgMaskVal, imgRedVal, imgYellowVal))

        if imgMaskVal < 50000 and imgYellowVal < 50000 and imgRedVal <= 255:      #start a new set
            return True
        else:
            self.reward = 0     #no win & lose
            return False


    def compareScoreArea(self, currentPic):
        currentImg = currentPic[int(0.1 * self.height) : int(0.16 * self.height), :]
        cv2.imwrite("prev.jpg", currentImg)

        cropR = currentImg[:, int(0.5 * self.width) : int(self.width)]
        cropL = currentImg[:, 0 : int(0.5 * self.width)]

        cropRH, cropRW = cropR.shape        #height, width, channel = Img.shape
        cropLH, cropLW = cropL.shape

        cropRVal = np.sum(cropR[0:cropRH, 0:cropRW])
        cropLVal = np.sum(cropL[0:cropLH, 0:cropLW])

        #cv2.imwrite('desire.jpg', self.img_red)
        print("RVal: %d, LVal: %d" %(cropRVal, cropLVal))
        if(cropRVal > cropLVal):        #When right serve, our pikachu win this set
            self.isWin = True
            self.reward = 1         #this set is win
        else:
            self.isWin = False
            self.reward = -1        #this set is lose


    def random_keyboard(self, is_stop=True):
        if is_stop:
            # modify t to make Pika more powerful(randomly)
            t = random.uniform(0, 0.001)
            time.sleep(t)
            #random.choice(self.key_ground)(1)
            item = random.choice(self.key_action_strings)
            self.key_action_dicts[item](1)
            self.action = self.key_action_strings.index(item)


    def get_standard(self, first_=False, set_=False):
        if first_:
            self.preprocess_img(resize=False)
            gameset_match = cv2.matchTemplate(self.mask, self.polepoint, eval('cv2.TM_CCOEFF_NORMED'))
            if np.max(gameset_match) > self.threshold:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gameset_match)
                top_left = max_loc
                w, h = self.polepoint.shape[::-1]
                x, y = (top_left[0] + w // 2, top_left[1] + h // 2)
                self.pole = x
                self.ground = y
                return True
            else:
                return False

        else:
            self.preprocess_img(resize=True)
            gameset_match = cv2.matchTemplate(self.mask, self.polepoint, eval('cv2.TM_CCOEFF_NORMED'))
            if np.max(gameset_match) > self.threshold:
                return True
            else:
                return False


if __name__ == "__main__":
    height = 350
    width = 450
    frame_time = 0.2
    env = Env(height, width, frame_time)

    replay_buffer = []

    while True:
        env.preprocess_img(resize=False, save_dir=False)
        #print(env.get_standard())

        #cv2.imshow('img', env.mask)
        cv2.imshow('img', env.crop_img)
        env.state = env.crop_img

        isNewSet = env.get_isNewSet()

        if(isNewSet == True):
            sleep(1.2)
            env.preprocess_img(resize=False, save_dir=False)
            env.compareScoreArea(env.img_red)
            #cv2.imwrite('current.jpg', env.img);
            if(env.isWin):
                print("Win this set")
            else:
                print("Loss this set")
        else:
            env.random_keyboard(True)
            env.preprocess_img(resize=False, save_dir=False)
            env._state = env.crop_img
            replay_buffer.append({'state': env.state, 'action': env.action, 'reward': env.reward, '_state': env._state})
            with open('temp.pickle', 'wb') as file:       #pickle open & write file
                pickle.dump(replay_buffer, file)

            #print(env.action)
            env.currentImg = env.img_red        #from maskd red image to get which pikachu wins


        k = cv2.waitKey(30)&0xFF #64bits! need a mask
        if k == 27:   #Esc to stop
            cv2.destroyAllWindows()
            env.random_keyboard(False)
            break



















