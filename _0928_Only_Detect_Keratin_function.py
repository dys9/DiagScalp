#-*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import time

cv2.cuda_GpuMat()

# --------- Get Variables
kernel_0 = np.ones((3, 3), np.uint8)
kernel_1 = np.ones((5, 5), np.uint8)
kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))  # (4,8)
kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))  # (4,8)
kernel_4 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))  # (4,8)
kernel_5 = np.array(([[1, 1, 1], [1, -1, 1], [1, 1, 1]]))
kernel_6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
kernel_8 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel_9 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
kernel_laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
kernel_sharp = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0
lower_range = (0)
upper_range = (90)


#---------영상 히스토그램 조작
def hist_mani(img, m0=128, s0=52):
    m = np.mean(img)
    s = np.std(img)
    out = img.copy()

    # normalize
    out = s0 / s * (out - m) + m0
    out[out < 0] = 0
    out[out > 255] = 255
    out = out.astype(np.uint8)
    return out



def createMask(source, avg, std):

    print("_____Create_MASK___________")


    # ----------------------- 원본
    img = cv2.imread(source)
    print("avg = %.3f \nstd = %.3f " % (avg, std))


    # ----------------------- Gamma
    gamma = ((avg - math.sqrt(avg)) / std) * 2.0
    print("gamma_hair = %f" % gamma)
    gamma_cvt = np.zeros((256, 1), dtype='uint8')


    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    gamma_img = cv2.LUT(img, gamma_cvt)


    # ----------------------- Gray
    gray_img = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)


    # ----------------------- Sharpening
    sharp_img = cv2.filter2D(gray_img, -1, kernel_sharp)
    # sharp_img = cv2.filter2D(sharp_img, -1, kernel_sharp)


    # ----------------------- BlackHat
    blackhat_img = cv2.morphologyEx(sharp_img, cv2.MORPH_BLACKHAT, kernel_8)


    # ----------------------- Not zero statistics
    not_zero = blackhat_img[blackhat_img>0]
    not_zero_avg = np.average(not_zero)
    not_zero_std = np.std(not_zero)
    print("not_zero_avg = %.3f \nnot_zero_std = %.3f " % (not_zero_avg, not_zero_std))


    # ----------------------- Blur
    # gaussian_blur_img = cv2.GaussianBlur(blackhat_img, (7, 9), sigmaY=math.sqrt(not_zero_std),sigmaX=math.sqrt(not_zero_std))
    gaussian_blur_img = cv2.medianBlur(blackhat_img, ksize=5)################################################################################################
    # gaussian_blur_img = blackhat_img



    # --------- 원본 흑백 영상 밝기 변화량 계산, 1차 차분
    hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])

    diff_hist = np.zeros((254, 1))
    min_hist = 0
    d_tsh = 0
    for i in range(len(hist) - 2):
        diff_hist[i] = int((hist[i + 1] - hist[i]))
        if min_hist > diff_hist[i]:
            min_hist = diff_hist[i]
            d_tsh = i
    print("d_tsh = ", d_tsh)
    d_tsh = int(d_tsh - std - math.sqrt(avg))
    print("d_tsh - @ = ", d_tsh)


    # ----------------------- Normal_Thresh_Origin_img
    _, thresh_origin_img = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),d_tsh, 255,cv2.THRESH_BINARY_INV) # 90

    # ----------------------- Normal_Thresh
    ret, thresh_gamma_img = cv2.threshold(gaussian_blur_img, int(not_zero_avg), 255, cv2.THRESH_BINARY) # int(not_zero_avg) # round(not_zero_avg + 0.5) # round(not_zero_avg)
    print("thin_hair_ret = ", ret)

    # ----------------------- Thresh OR
    thresh_img = cv2.bitwise_or(thresh_gamma_img, thresh_origin_img)

    # ----------------------- Opening
    hair_mask_img = cv2.morphologyEx(thresh_img, cv2.MORPH_DILATE, kernel_0)

    # ------------------------------------------------------- 반사 마스크 생성
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv_img)


    bilateral_img = cv2.bilateralFilter(v, 5, 75, 75)
    light_mask_img = (bilateral_img - v)
    light_mask_img = cv2.morphologyEx(light_mask_img, cv2.MORPH_CLOSE, kernel_1, iterations=2)
    light_mask_img = cv2.dilate(cv2.bitwise_not(light_mask_img), kernel_2, iterations=1)
    _, light_mask_img = cv2.threshold(light_mask_img, 180, 255, cv2.THRESH_BINARY)


    # -----------------------------------------마스크합성-----------------------------------------------------#
    # --------- 모서리 마스크
    size_y = np.shape(img)[0]
    size_x = np.shape(img)[1]

    rect_mask_img = np.zeros((size_y,size_x), np.uint8)
    rect_mask_img = cv2.rectangle(rect_mask_img, (0, 0), (int(size_x * 0.2), int(size_y * 0.2)), (255, 0, 0), cv2.FILLED)
    rect_mask_img = cv2.rectangle(rect_mask_img, (size_x, 0), (int(size_x * 0.8), int(size_y * 0.2)), (255, 0, 0), cv2.FILLED)
    rect_mask_img = cv2.rectangle(rect_mask_img, (0, size_y), (int(size_x * 0.2), int(size_y * 0.8)), (255, 0, 0), cv2.FILLED)
    rect_mask_img = cv2.rectangle(rect_mask_img, (size_x, size_y), (int(size_x * 0.8), int(size_y * 0.8)), (255, 0, 0), cv2.FILLED)


    # --------- [반사 + 모발] 제거 마스크 생성
    mask_img = cv2.bitwise_or(light_mask_img, hair_mask_img)
    mask_img = cv2.bitwise_or(mask_img, rect_mask_img)


    return mask_img


#--------- Detect Keratin & Get Whole Mask
def keratinDetect(image, mask_img, avg, std):

    print("\n_____Keratin_Detection_____")

    # --------- 영상 읽기
    img = cv2.imread(image)


    # ---------------------------------------전처리-시작-----------------------------------------------------#
    # --------- 감마 처리, 영상을 아주 어둡게
    gamma = ((255. - avg) / std) * 0.1
    # gamma = ((255. - avg) / (2 * std))
    print("gamma_keratin = %f" % gamma)
    gamma_cvt = np.zeros((256, 1), dtype='uint8')


    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    gamma_img = cv2.LUT(img, gamma_cvt)


    # --------- 흑백 전환
    gamma_hsv_img = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2HSV)
    _, _, gray_gamma_img = cv2.split(gamma_hsv_img)


    # --------- 그레디언트 생성
    gradi_img = cv2.morphologyEx(gray_gamma_img, cv2.MORPH_TOPHAT, kernel_6) # kernel_0


    # --------- 마스크 처리
    mask_img = cv2.dilate(mask_img, kernel_1) # kernel_1
    mask_gradi_img = cv2.bitwise_and(cv2.bitwise_not(mask_img), gradi_img)
    # cv2.imwrite("mask_gradi_"+image, mask_gradi_img)
    # cv2.imshow("mask_gradi_img", mask_gradi_img)

    # --------- 남은 피부 이미지의 히스토그램
    hist = cv2.calcHist([mask_gradi_img], [0], None, [256], [0, 256])


    # --------- 마스크의 밝기가 0이 아닌 곳의 평균과 분산
    # not_zero = [i for i in mask_gradi_img.ravel().tolist() if i is not 0]
    not_zero = mask_gradi_img[mask_gradi_img>0]
    not_zero_avg = np.average(not_zero)
    not_zero_std = np.std(not_zero)


    # --------- 마스크의 변화량 계산, 1차 차분
    diff_hist = np.zeros((254, 1))
    for i in range(len(hist) - 2):
        diff_hist[i] = int(abs(hist[i + 1] - hist[i]))


    # --------- res 값 추출, OTSU Algorithm
    res, _ = cv2.threshold(mask_gradi_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # --------- 변화량이 낮은 밝기값 추출
    tsh = 0
    for i in range(len(diff_hist)-1):
        if diff_hist[i] < diff_hist[i + 1]:
            if i > res and diff_hist[i] <= 10:
                tsh = i
                break


    print("res = %d , avg = %.3f, std = %.3f, not_zero_avg = %.3f , not_zero_std = %.3f, tsh = %d" % (res, avg, std, not_zero_avg, not_zero_std, tsh))


    # --------- 경계값 판별 후 return 경계값, 흑백 각질 영상
    if avg > 120:   # 영상이 밝은 경우 if not_zero_avg + tsh > avg or avg > 137:

        print("밝은 경우의 경계값 = %d\n" % tsh)
        return tsh - int(not_zero_avg), mask_gradi_img

    else:
        tsh = res
        print("어두운 경우의 경계값 = %d\n" % tsh)

        return tsh, mask_gradi_img




def keratinDraw(source, tsh, mask_gradi_img):
    print("_____Keratin_Draw__________")

    # --------- 전달 받은 경계값으로 Threshold 처리
    _, thresh = cv2.threshold(mask_gradi_img, tsh, 255, cv2.THRESH_BINARY)


    # --------- 각질 영역 확장
    morph_img = cv2.dilate(thresh, kernel_6)
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel_7)

    # --------- Contour 탐색
    img = cv2.imread(source)
    contours, hierarchy = cv2.findContours(morph_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_NONE

    area_True = []
    area_False = []
    keratin = np.zeros((np.shape(img)[0], np.shape(img)[1]), np.uint8)

    for con in contours:
        if cv2.contourArea(con, False) > 0:
            area_False.append(con)
        if cv2.contourArea(con, True) > 0:
            area_True.append(con)

    for i in range(len(area_True)):
        pts = area_True[i];
        pts = pts.astype('int32')
        pts = pts.reshape(-1, 1, 2)

        cv2.drawContours(keratin, [pts], 0,128,5)

    for i in range(len(area_False)):
        cv2.drawContours(keratin, [area_False[i]], 0, 128, 5)


    # --------- 남은 모발 제거
    minLineLength = 120
    maxLineGap = 30
    temp = np.copy(img)

    lines = cv2.HoughLinesP(keratin, 1, np.pi / 180, int(tsh), minLineLength, maxLineGap)


    if lines is not None:
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(keratin, (x1, y1), (x2, y2), 0, 6)
                cv2.line(temp, (x1, y1), (x2, y2), (0,255,0), 6)
        print("제거된 직선의 수 !! : ",len(lines))
    else:
        print("제거된 직선 없음 !!")


    # --------- 검출된 각질 이미지 색칠
    img = cv2.add(img, cv2.merge([np.zeros((np.shape(img)[0], np.shape(img)[1]), np.uint8),np.zeros((np.shape(img)[0], np.shape(img)[1]), np.uint8),keratin]))


    print("[[tsh = %d] 검출된 각질의 수 : %d ]\n" %(tsh,len(area_True) + len(area_False)))


    return img # 각질 검출 결과 영상


def keratinCalc(result, mask):
    print("_____Keratin_Calculate_____")
    _, _, r = cv2.split(result)

    zero = len(mask[mask==0])
    red  = len(r[r>254])

    return str(" %.3f" % ((red/zero)* 100))+"%"



os.chdir("D:/dendong")
os.chdir("D:/홍조")
source = 'test13.bmp'
    # _______________________________________________________
img = cv2.imread(source)
cv2.imshow("Origin_img", img)
img = cv2.medianBlur(img, ksize=13)
cv2.imshow("Preprocessed_img", img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
avg = np.average(gray_img)
std = np.std(gray_img)

    # Create Mask
imported_mask = createMask(source, avg, std)

cv2.imshow("imported_mask", imported_mask)
cv2.waitKey()
cv2.destroyAllWindows()
