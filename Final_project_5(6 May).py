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
    i=0
    while i<len(list_kp1):
        if list_kp1[i,1]>750:
            list_kp1=np.delete(list_kp1,(i),axis=0)
            list_kp2=np.delete(list_kp2,(i),axis=0)
            i-=1
        i+=1
            
    Swipe_1=np.hstack((list_kp1[:,0],960-list_kp1[:,1]))
    Swipe_2=np.hstack((list_kp2[:,0],960-list_kp2[:,1]))
    return Swipe_1,Swipe_2

def F_calculator(X1,X2):
    A= np.zeros((8,9))
    for i in range(len(A)):
        # Equation x1x2,x1y2,x1,y1x2,y1y2,y1,x2,y2,1
        A[i,:]=[X1[i,0]*X2[i,0],X1[i,0]*X2[i,1],X1[i,0],X1[i,1]*X2[i,0],X1[i,1]*X2[i,1],X1[i,1],X2[i,0],X2[i,1],1]
    u, s, vh = LA.svd(A, full_matrices=True)
    v=vh.transpose()
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
    error=0.01
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
            if abs(np.dot(np.dot(a.transpose(),F),b))<error:
                sum+=1
        if best<=sum:
            best=sum
            F_best=F
            T1_best=T1
            T2_best=T2
            points_1=Save_x1
            points_2=Save_x2
    F_best=F_best/(LA.norm(F_best))
    u,s,vt=LA.svd(F_best)
    S=np.mat([[float(s[0]),0,0],[0,float(s[1]),0],[0,0,0]])
    F_final=np.dot(np.dot(u,S),vt)
    #F_denorm=np.dot(np.dot(LA.inv(T1_best).T,F_final),LA.inv(T2_best))
    F_denorm=F_final
    #F_denorm=F_denorm/F_denorm[2,2]
    #print(F)
    # print(LA.det(F),'det of F')

    return F_denorm,points_1,points_2

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
            C[:,:,i]=-C[:,:,i]
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
    if best_C[2,0]:
        best_C=-best_C
    return best_C,best_R

def RotToEuler(R):
    def isRotationMatrix(R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return x,y,z


def EulerToMat(alpha,beta,gamma):
    Rz=np.mat([[math.cos(gamma),-math.sin(gamma),0],[math.sin(gamma),math.cos(gamma),0],[0,0,1]])
    Ry=np.mat([[math.cos(beta),0,math.sin(beta)],[0,1,0],[-math.sin(beta),0,math.cos(beta)]])
    Rx=np.mat([[1,0,0],[0,math.cos(alpha),-math.sin(alpha)],[0,math.sin(alpha),math.cos(alpha)]])
    R=np.dot(np.dot(Rz,Ry),Rx)
    return R

def plot_image(R,T,P,plot_img,Green,Blue):
    d_X,d_Y,d_Z=RotToEuler(R)
    print(d_X,d_Y,d_Z)

    if d_Y<0 :
        d_Y=0

    if d_Y >0.05 and d_Y < 2 :
        d_Y=0

    R=EulerToMat(0,d_Y,0)

    if T[2,0]<0:
        T[2,0]=-T[2,0]
    if T[0,0]>3:
        T[0,0]=3
    if T[0,0]<-3:
        T[0,0]=-3

    A=np.hstack((R,T))
    P_new=np.vstack((A,[0,0,0,1]))
    answer=np.dot(P_new,P)

    plot_x=int(-answer[2,3])+1200
    plot_y=int(answer[0,3])+200
    #plot_img[plot_x,plot_y]=0

    for i in range(-7,7):
        plot_img[plot_x+i,plot_y+i,:]=[0,Green,Blue]
        plot_img[plot_x-i,plot_y-i,:]=[0,Green,Blue]
        plot_img[plot_x,plot_y-i,:]=[0,Green,Blue]
        plot_img[plot_x-i,plot_y,:]=[0,Green,Blue]

    resized = cv2.resize(plot_img, (750,750))
    cv2.imshow('plot',resized)
    cv2.waitKey(2)
    return answer

def Pipeline():
    #vidObj = cv2.VideoCapture()
    count=0
    img_array=[]
    i=0
    plot_img=255*np.ones((2500,2500,3),dtype=np.uint8)
    PathtoRead="Oxford_dataset/stereo/centre/*.png"
    fx, fy, cx, cy, G_camera_image, LUT=ReadCameraModel("Oxford_dataset/model")
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    A=np.hstack((np.eye(3),np.zeros((3,1))))
    P=np.vstack((A,[0,0,0,1]))
    P_func=np.vstack((A,[0,0,0,1]))
    
    for img in sorted(glob.glob(PathtoRead)):
        if count==0:
            Image_1=ImagePreprocessor(img,LUT)
            height,width,layers=Image_1.shape
            size = (width,height)
            Image_2=Image_1
        else:
            Image_2=ImagePreprocessor(img,LUT)
            X1,X2=fetch_ORBKeypoints(Image_1,Image_2)
            if len(X1)>=8:
                F,point_X1,point_X2=Ransac(X2,X1,100,size)
                #print(np.shape(F_function))
                E=E_calculator(K,F)
                best_C,best_R=Rotation_translation(K,E,point_X1,point_X2)
                E_func, mask = cv2.findEssentialMat(X1, X2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.95, threshold=1)
                points, R_func, t_func, mask = cv2.recoverPose(E_func, X1, X2)
                #print(best_C,'C')
                #print(best_R,'R')
                #print(LA.det(best_R))
                print(P,'Path value')
            else :
                best_R=np.eye(3)
                best_C=np.zeros((3,1))
                R_func=np.eye(3)
                C_func=np.zeros((3,1))
            P=plot_image(best_R,best_C,P,plot_img,200,0)
            P_func=plot_image(R_func,t_func,P_func,plot_img,0,200)
            print(count)
        cv2.imshow('video',Image_2)
        cv2.imshow('video_1',Image_1)
        count += 1
        Image_1=Image_2
        print('Frame processing index')
        print(i)
        append_image=plot_img.copy()
        #cv2.imwrite('%d.jpg' %count,Final)
        img_array.append(append_image)
        #success, image = vidObj.read()
    cv2.imwrite('%d.jpg' %count,plot_img)
    return img_array,size

def video(img_array,size):
    video=cv2.VideoWriter('1.avi' ,cv2.VideoWriter_fourcc(*'DIVX'), 50.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()
# main
if __name__ == '__main__':

    # Calling the function
    Image,size=Pipeline()
    size=(2500,2500)
    video(Image,size)