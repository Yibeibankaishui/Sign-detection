#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace std;
using namespace cv;

double down_scale = 0.4;
int H_min;
int H_max;
int S_min;
int S_max;
int V_min;
int V_max;

int main(int argc, char *argv[])
{
    //加载图像
    Mat srcImage=imread("../images/signs/IMG_3093 Large.jpeg");
    
    int down_height = srcImage.size().height * down_scale;
    int down_width = srcImage.size().width * down_scale;
    Mat downImage;
    resize(srcImage, downImage, Size(down_width, down_height), INTER_LINEAR);
    imshow("原始图",downImage);

    //变换成hsv通道
    Mat hsvImage;
    cvtColor(downImage,hsvImage,COLOR_BGR2HSV);
    // imshow("未增强色调的hsv图片",hsvImage);

    //分割split 与合并merge
    vector<Mat> hsvsplit;//hsv的分离通道
    split(hsvImage,hsvsplit);
    // equalizeHist(hsvsplit[2],hsvsplit[2]);//直方图均衡化，增强对比度，hsvsplit[2]为返回的h
    // merge(hsvsplit,hsvImage);//在色调调节后，重新合并
    // imshow("增强色调对比度后的hsv图片",hsvImage);

    // 取得标志蓝色色块
    Mat binImage;
    threshold(hsvsplit[0],binImage,(int)(256 * 0.5),(int)(256 * 0.7),THRESH_BINARY);
    imshow("二值化后图片",binImage);

    // 检测圆
    // Mat edges;
    // Canny(thresHold, edges, 100, 200, 3, false);
    // imshow("edges", edges);

    // 投影变换

    // 计算圆心和质心




   while(1)
   {
       int key = waitKey(0);
   if (key==27)
   {
       break;
   }
   }
   return(0);


}
