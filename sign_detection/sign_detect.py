import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

resize_ratio = 1

search_N = 7    # 搜索邻域半径

show = True



# 读取图片

# 检测圆形 作为ROI

# 在圆形中计算白色像素重心

'''
轮廓检测
'''

# 对向量中每一个点，判断其值与邻域点值的大小，若大部分邻域点均大于其值，则为峰谷点
# 得到两个峰谷点的索引，比较大小
def find_min(x_vector):
    min_idx = []
    for idx in range(search_N,len(x_vector) - search_N):
        x_cur = x_vector[idx]
        greater_cnt_l = 0
        greater_cnt_r = 0
        for j1 in range(-search_N, 0):
            greater_cnt_l += int(x_vector[idx + j1] > x_cur)
        for j2 in range(1, search_N + 1):
            greater_cnt_r += int(x_vector[idx + j2] > x_cur)
        # print(greater_cnt)
        if (((greater_cnt_l > search_N-2) and (greater_cnt_r > search_N-2))):
            # print(idx)
            min_idx.append(x_vector[idx])
    # print(min_idx)
    if (min_idx[0]>min_idx[-1]):
        print("left")
    if (min_idx[0]<min_idx[-1]):
        print("right")

    
            
        

if __name__ == "__main__":
    #加载图像img
    img = cv2.imread('./images/signs/signs/IMG_3134 Medium.jpeg',cv2.IMREAD_COLOR)
    # img = cv2.imread('./images/signs/1.jpeg',cv2.IMREAD_COLOR)
    img = cv2.resize(img, (int(img.shape[1]*resize_ratio), int(img.shape[0]*resize_ratio)), interpolation = cv2.INTER_AREA)
    if (show):
        cv2.imshow('img',img)

    # # 直方图均衡
    # img = cv2.equalizeHist(img)
    # cv2.imshow('equalized', img)
    # cv2.waitKey(0)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 180，255，255
    low_hsv = np.array([100, 88, 0])
    high_hsv = np.array([130, 255, 255])

    img_blue = cv2.inRange(hsv_img, low_hsv, high_hsv)
    if (show):
        cv2.imshow("result", img_blue) # 显示图片
    # cv2.waitKey(0)

    # canny边缘检测
    # img_canny=cv2.Canny(img_blue,600,100,3)
    # cv2.imshow("canny", img_canny)

    contours, hierarchy = cv2.findContours(img_blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#contours为轮廓集，可以计算轮廓的长度、面积等

    #### 不能用
    # mu = [None] * len(contours)
    # for i in range(len(contours)):
    #     mu[i] = cv2.moments(contours[i],True)
    # # Get the mass centers
    # mc = [None]*len(contours)
    # for i in range(len(contours)):
    #     # add 1e-5 to avoid division by zero
    #     mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

    # 找椭圆
    for cnt in contours:
        if len(cnt)>50:
            # 围合区域的面积
            S1 = cv2.contourArea(cnt)
            # bounding box
            (bb_x, bb_y, bb_w, bb_h) = cv2.boundingRect(cnt)
            ell = cv2.fitEllipse(cnt)
            # print(ell.points())
            # 拟合椭圆的面积
            S2 = (0.25) * math.pi * ell[1][0] * ell[1][1]     # 外接最小矩形 质心（圆心），边长，角度
            if (S1/S2) > 0.85 :#面积比例，可以更改，根据数据集。。。
                img_to_show = cv2.ellipse(img, ell, (0, 255, 0), 2)
                img_to_show  = cv2.circle(img_to_show, (int(ell[0][0]),int(ell[0][1])), 2, (0, 0, 255), 2)
                img_to_show  = cv2.circle(img_to_show, (bb_x, bb_y), 2, (0, 0, 255), 2)
                img_to_show  = cv2.rectangle(img_to_show , pt1=(bb_x, bb_y), pt2=(bb_x+bb_w, bb_y+bb_h),color=(255, 255, 255), thickness=3)
                # print(str(S1) + "    " + str(S2) + "   " + str(ell[0][0]) + "   " + str(ell[0][1]))
    if (show):
        cv2.imshow("ellipse",img_to_show)
    # 对img_blue进行投影变换
        
    # img = cv2.circle(img, (int(mc[0][0]),int(mc[0][1])), 3, (255, 255, 255))
    # cv2.imshow("ellipse",img)

    # 求质心
    # mm = cv2.moments(img_blue, True)
    #     m01 = m10 = m00 = 0
    # dx = bb_x
    # while(dx < (bb_x + bb_w)):
    #     dy = bb_y
    #     while(dy < (bb_y + bb_h)):
    #         if(dx < img.shape[0] and dy < img.shape[1]):
    #             pixel = img_blue.item(dx, dy)
    #         m01 += dx * pixel
    #         m10 += dy * pixel
    #         m00 += pixel
    #         # print(pixel)
    #         dy += 1
    #     dx += 1

    # mm = (int(m10/m00), int(m01/m00))
    # print(mm)

    # img = cv2.circle(img, mm, 2, (0, 255, 255), 2)

    # 统计水平方向的像素个数
    img_mask = np.zeros_like(img_blue)
    img_mask[bb_y:bb_y+bb_h,bb_x:bb_x+bb_w] = 255
    # img_blue[bb_y:bb_y+bb_h,bb_x:bb_x+bb_w] = 255
    # img_res = img_mask*img_blue
    img_res = cv2.bitwise_and(img_mask,img_blue)
    cv2.imshow("img_res",img_res)
    
    sum_for_x = np.sum(img_res, 0)
    if show:
        plt.plot(sum_for_x)
        plt.show()
    find_min(sum_for_x)
    # print(sum_for_x)
    # print(len(sum_for_x))
    # idx = 0
    # for x in sum_for_x:
    #     print(sum_for_x[idx])
    #     print(x)
    #     idx += 1
    if (show):
        cv2.waitKey(0) # 等待键盘触发事件，释放窗口
        cv2.destroyAllWindows()











