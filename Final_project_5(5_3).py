import argparse
import sys
import os, sys
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import math
from PIL import Image
import scipy.ndimage as nd
from scipy.ndimage import map_coordinates as interp2
import random
import glob

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def ReadCameraModel(models_dir):
    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"
    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]
    # 4x4 matrix that transforms x-forward coordinate frame at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5,0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()
    return fx, fy, cx, cy, G_camera_image, LUT

def UndistortImage(image,LUT):
    reshaped_lut = LUT[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)
    return undistorted.astype(image.dtype)

def ImagePreprocessor(img,LUT):
    image=cv2.imread(img,0)
    color_image = cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
    undistort_image=UndistortImage(color_image,LUT)
    return undistort_image

def fetch_ORBKeypoints(img1,img2):
    orb=cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    pts = np.asarray([[p.pt[0], p.pt[1]] for p in kp1])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    list_kp1 = np.mat([0,0])
    list_kp2 = np.mat([0,0])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        x1,y1 = kp1[img1_idx].pt
        x2,y2 = kp2[img2_idx].pt
        append_1=np.mat([x1, y1])
        append_2=np.mat([x2, y2])
        list_kp1=np.vstack((list_kp1,append_1))
        list_kp2=np.vstack((list_kp2,append_2))
    list_kp1=np.delete(list_kp1,(0),axis=0)
    list_kp2=np.delete(list_kp2,(0),axis=0)
    return list_kp1,list_kp2

def F_calculator(X1,X2):
    A= np.zeros((8,9))
    for i in range(len(A)):
        # equation x1x2,x1y2,x1,y1x2,y1y2,y1,x2,y2,1
        A[i,:]=[X1[i,0]*X2[i,0],X1[i,0]*X2[i,1],X1[i,0],X1[i,1]*X2[i,0],X1[i,1]*X2[i,1],X1[i,1],X2[i,0],X2[i,1],1]
    u, s, vh = LA.svd(A, full_matrices=True)
    v=vh.T
    F=np.reshape(v[:,8],(3,3))
    return F

def Normalize(X1):
    M1=max(X1[:,0])
    m1=min(X1[:,0])
    M2=max(X1[:,1])
    m2=min(X1[:,1])
    T=np.mat([[float(1/(M1-m1)),0,float(m1/(M1-m1))],[0,float(1/(M2-m2)),float(m2/(M2-m2))],[0,0,1]])
    X1[:,0]=(X1[:,0]-m1)/(M1-m1)
    X1[:,1]=(X1[:,1]-m2)/(M2-m2)
    return X1,T

def random_points(X1,X2,size):
    Comp_X1=np.mat(np.zeros((8,2)))
    Comp_X2=np.mat(np.zeros((8,2)))
    i=1
    a=random.randint(0,len(X1)-1)
    Comp_X1[0,:]=X1[a,:]
    Comp_X2[0,:]=X2[a,:]
    while(i<8):
        correction=0    
        # 960 X 1280
        a=random.randint(0,len(X1)-1)
        for j in range(i):  
            if abs(float(Comp_X1[j,0])-float(X1[a,0]))>size[1]/32 or abs(float(Comp_X2[j,1])-float(X1[a,1]))>size[0]/32:
                correction+=1
        if correction==i:
            Comp_X1[i,:]=X1[a,:]
            Comp_X2[i,:]=X2[a,:]
            i+=1
    return Comp_X1,Comp_X2
    
def vector_to_skew(A):
    matrix=np.mat([[0,-A[0,2],A[0,1]],[A[0,2],0,-A[0,0]],[-A[0,1],A[0,0],0]])
    return matrix    


def Ransac(X1,X2,iteration,size):
    error=0.005
    best=0
    F_best=[]
    T1_best=[]
    T2_best=[]
    points_1=[]
    points_2=[]
    for i in range(iteration):
        Org_X1,Org_X2=random_points(X1,X2,size)
        Save_x1=Org_X1.copy()
        Save_x2=Org_X2.copy()
        
        Comp_X1,T1=Normalize(Org_X1)
        Comp_X2,T2=Normalize(Org_X2)
        F=F_calculator(Comp_X1,Comp_X2)
        sum=0
        for k in range(len(X2)):
            a=np.zeros((3,1))
            a[0,0],a[1,0],a[2,0]=X1[k,0],X1[k,1],1
            #print(np.shape(a))
            b=np.zeros((3,1))
            b[0,0],b[1,0],b[2,0]=X2[k,0],X2[k,1],1
            #print(np.shape(b))
            if(np.dot(np.dot(a.T,F),b))<error:
                sum+=1
        if best<sum:
            best=sum
            F_best=F
            T1_best=T1
            T2_best=T2
            points_1=Save_x1
            points_2=Save_x2
    F_best=F_best/(LA.norm(F_best))
    
    F_denorm=np.dot(np.dot(LA.inv(T1_best).T,F_best),LA.inv(T2_best))
    u,s,vt=LA.svd(F_denorm)
    S=np.mat([[float(s[0]),0,0],[0,float(s[1]),0],[0,0,0]])
    F_final=np.dot(np.dot(u,S),vt)
    #print(F)
    print(LA.det(F),'det of F')
    return F_final,points_1,points_2

def E_calculator(K,F):
    answer=np.dot(np.dot(K.transpose(),F),K)
    u,s,vt=LA.svd(answer)
    sigma=np.array([[1,0,0],[0,1,0],[0,0,0]])
    answer=np.dot(np.dot(u,sigma),vt)
    #print(LA.det(answer))
    #print(answer)
    return answer

def Rotation_translation(K,E,point_X1,point_X2):
    C=np.zeros((3,1,4))
    R=np.zeros((3,3,4))
    U,S,Vt =LA.svd(E)
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    #print(U,'U')
    C[:,:,0]=U[:,2]
    C[:,:,1]=-U[:,2]
    C[:,:,2]=U[:,2]
    C[:,:,3]=-U[:,2]
    R[:,:,0]=R[:,:,1]=np.dot(np.dot(U,W),Vt)
    R[:,:,2]=R[:,:,3]=np.dot(np.dot(U,W.transpose()),Vt)
    for i in range(4):
        if LA.det(R[:,:,i])<0:
            R[:,:,i]=-R[:,:,i]
            C[:,:,i]=-C[:,:,3]
            print(LA.det(R[:,:,i]))
    best_C=np.mat(np.zeros((3,1)))
    best_R=np.mat(np.eye(3))
    best=0
    point_X1=np.hstack((point_X1,np.ones((len(point_X1),1))))
    point_X2=np.hstack((point_X2,np.ones((len(point_X2),1))))
    print(point_X1)
    X1_3D=np.ones((4,8,4))
    X2_3D=np.ones((4,8,4))
    for i in range(4):
        sum=0
        C_current=np.mat(C[:,:,i])
        R_current=np.mat(R[:,:,i])
        R_mat=np.mat(R[2,:,i])
        P=np.dot(K,np.eye(3,4))
        P_2=np.hstack((R_current,C_current))
        P_2=np.dot(K,P_2)
        #base=np.mat([0,0,0,1])
        #P_2=np.vstack((P_2,base))
        X1=point_X1
        X2=point_X2
        for k in range(len(point_X1)):
            #A=[X1[0,k]*P[2,:]-P[0,:]]
            #two=X1[1,k]*P[2,:]-P[1,:]
            #three=X2[0,k]*P_2[2,:]-P_2[0,:]
            #four=X2[1,k]*P_2[2,:]-P_2[1,:]
            #A=np.vstack((A,two))
            #A=np.vstack((A,three))
            #A=np.vstack((A,four))
            A1=np.dot(vector_to_skew(X1[k,:]),P)
            A2=np.dot(vector_to_skew(X2[k,:]),P)
            A=np.vstack((A1,A2))
            u,s,vt=LA.svd(A)
            v=vt.transpose()
            X=v[:,3]
            X=X/X[3]
            for h in range(4):
                X1_3D[h,k,i]=X[h,0]

            point_1=np.mat(X1_3D[0:3,k,i])
            point_1=point_1.reshape(3,1)
            if(np.dot(R_mat,(point_1-C_current)))>0:
                sum+=1
        if best<sum:
            best=sum
            best_C=C_current
            best_R=R_current
        print(sum,'sum')
    #print(best_C,'best C')
    #print(best_R,'best R')
    #print(LA.det(best_R))
    return best_C,best_R

def plot_image(R,T,P,plot_img):
    new=np.dot(R,P)+T
    plot_img[-int(new[2])+500,int(new[0])+500]=0
    cv2.imshow('plot',plot_img)
    cv2.waitKey(10)
    return new

def Pipeline():
    #vidObj = cv2.VideoCapture()
    count=0
    img_array=[]
    i=0
    plot_img=255*np.ones((1000,1000))
    PathtoRead="Oxford_dataset/stereo/centre/*.png"
    fx, fy, cx, cy, G_camera_image, LUT=ReadCameraModel("Oxford_dataset/model")
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    P=np.mat(np.zeros((3,1)))
    for img in sorted(glob.glob(PathtoRead)):
        if count==0:
            Image_1=ImagePreprocessor(img,LUT)
            height,width,layers=Image_1.shape
            size = (width,height)
            Image_2=Image_1
        else:
            Image_2=ImagePreprocessor(img,LUT)
            X1,X2=fetch_ORBKeypoints(Image_1,Image_2)
            F,point_X1,point_X2=Ransac(X1,X2,100,size)
            F_function=cv2.findFundamentalMat(point_X1,point_X2,method=CV_FM_8POINT)
            print(np.shape(F_function))
            
            E=E_calculator(K,F)
            best_C,best_R=Rotation_translation(K,E,point_X1,point_X2)
            #print(best_C,'C')
            #print(best_R,'R')
            #print(LA.det(best_R))
            print(P,'Path value')
            P=plot_image(best_R,best_C,P,plot_img)
            print(count)
        cv2.imshow('video',Image_2)
        cv2.imshow('video_1',Image_1)
        count += 1
        Image_1=Image_2
        print('Frame processing index')
        print(i)
        #cv2.imwrite('%d.jpg' %count,Final)
        #img_array.append(Final)
        #success, image = vidObj.read()
    cv2.imwrite('%d.jpg' %count,plot_img)
    return img_array,size

def video(img_array,size):
    video=cv2.VideoWriter('%s.avi' %Thing,cv2.VideoWriter_fourcc(*'DIVX'), 10.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()
# main
if __name__ == '__main__':

    # Calling the function
    Image,size=Pipeline()
    #video(Image,size)
