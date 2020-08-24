# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:39:13 2020

@author: tycoer
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2.aruco as aruco
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation as rot
from scipy import signal,linalg 
import open3d as o3d
import json
import socket
import h5py
from sklearn.cluster import MeanShift, estimate_bandwidth
class cvtools:
    '''
    根据ylb习惯, 对函数中变量名做出如下约定(cvtools下凡出现以下变量名,均遵循且仅遵循以下描述):

        uv:像素坐标(u为横坐标,v为纵坐标)
        xyz:世界坐标(待补充), 命名上:z = depth, 函数名中采用'depth', 参数采用'z'
        HW:图像尺寸(H为高,W为宽)
        hw:长度(h为高,w为宽)
        vh,vw:为像素长度,宽度
        xh,yw:点云长度,宽度
        
        path: 附带文件名的路径 , 例'./data/test.txt'
        dir:文件夹名,例 './data'
        name: 文件名, 例 'test.txt'
        
        img:单张图片       
        imgs:tuple或list形式存在的多张图片
        
    函数名前缀说明:
        aruco相关: aruco
        手眼标定相关: handeye
        算法相关:alg
        相机标定相关: camcalib
        点云相关:pc
        测量相关:measure
    其余未有前缀之函数, 原因为ylb未想好命名或功能不多-不足以形成前缀
        
        
    tips:
        1.numpy中对图像描述采用[v,u],即第一个值为列,第二个值为行
         opencv中对图像描述采用[u,v],即第一个值为行,第二个值为列
         故函数中经常出现  ::-1  的表达, 将[u,v]转成[v,u]
        2.
         
    '''
    
    def aruco_createDict(self,dict_name):
        '''
        生成aruco字典, 对应的字典名将决定带有aruco前缀函数的作用
        例: 若对该函数传入dict_name='4x4', 则对于一张图中存在同时4x4,6x6的图片,aruco_centroid函数将只识别4x4的marker

        Parameters
        ----------
        dict_name : str 字典名
            DESCRIPTION.支持输入'4x4','5x5','6x6','7x7',默认字典长度为250的aruco字典

        Returns
        -------
        aruco_dict : obj, 输出的字典
            DESCRIPTION.

        '''
        if dict_name=='4x4':
            aruco_dict=3
        elif dict_name=='5x5':
            aruco_dict=7
        elif dict_name=='6x6':
            aruco_dict=11                   
        elif dict_name=='7x7':
            aruco_dict=15
        aruco_dict=aruco.Dictionary_get(aruco_dict)
        return aruco_dict

    def aruco_createBoard(self,aruco_dict,size=0.03,shape=(6,7),firstMarker=0,border=10):
        '''
        生成aruco板

        Parameters
        ----------
        aruco_dict :obj aruco字典, 参见函数aruco_createDict
            DESCRIPTION.
        size : float, optional, 每个aruco的实际边长,单位为m
            DESCRIPTION. The default is 0.03.
        shape : tuple, optional,板的行列数
            DESCRIPTION. The default is (6,7).
        firstMarker : 首个aruco序号, optional
            DESCRIPTION. The default is 0. 例: firstMarker=7 则aruco板会从ids=7的marker开始生成,(这里单个aruco被称为marker)
        border : int, 每个marker的边界的像素长度
            DESCRIPTION. The default is 10.

        Returns
        -------
        board : array(uint8),生成的板子
            DESCRIPTION.

        '''
        border=int(border)
        row,col,size_real=shape[0],shape[1],size*1000*3.74
        board=aruco.GridBoard_create(row,col,size_real,border,aruco_dict,firstMarker)
        row_px,col_px=int(size_real*row+border*(row+1)),int(size_real*col+border*(col+1))
        board=board.draw((row_px,col_px),marginSize = border)
        return board
    
    def aruco_createMarker(self,aruco_dict,ids=0,size=0.05,shape=(1,1),border=5,save_dir='./marker'):
        '''
        生成单个aruco

        Parameters
        ----------
        aruco_dict : obj,参见函数aruco_createDict
            DESCRIPTION.
        ids : int, optional, marker的ids
            DESCRIPTION. The default is 0.
        size : float, optional, marker的实际边长,单位为m
            DESCRIPTION. The default is 0.05.
        shape : tuple, optional, marker板的行列数
            DESCRIPTION. The default is (1,1). 生成具有相同ids的marker板
        border : int, optional, 边界像素长度
            DESCRIPTION. The default is 5.

        Returns
        -------
        marker : array(uint8) 生成的marker
            DESCRIPTION.

        '''
        # 生成一个marker, 必须用a4纸打印,用其他规格的纸计算结果会出错.
        # 第三个参数为生成marker的像素尺寸单位为px,对于a4纸, 1mm=3.78px, 故若想在a4纸上打印实际50mm*50mm的marker, 则marker的像素长度为size*1000*3.78, 其中size的单位为m
        border=int(border)
        marker=aruco.drawMarker(aruco_dict,ids,int(size*1000*3.78)) # 生成单个
        marker=cv2.copyMakeBorder(marker,border,border,border,border,cv2.BORDER_CONSTANT,value=255) # 填充边界
        marker=np.tile(marker,shape) # 根据num生成多个
        marker_size=aruco_dict.markerSize
        if os.path.exists(save_dir)==False:
            os.mkdir(save_dir)
        name=str(ids)+'_'+str(marker_size)+'x'+str(marker_size)+'x'+str(size),'m'+'.jpg'
        path=os.path.join(save_dir, name)
        cv2.imwrite(path,marker)
        return marker
                
    def aruco_centroid(self,img,aruco_dict,draw=False):
        '''
        提取图片中所有marker的中心点

        Parameters
        ----------
        img : array(uint8), 输入图片
            DESCRIPTION. 需为单张图片
        aruco_dict : obj,参见函数aruco_createDict
            DESCRIPTION.
        draw : bool, optional,是否绘制marker
            DESCRIPTION. The default is False. 直接绘制原图, 谨慎使用

        Returns
        -------
        centroid : tuple, 中心点元组, 元组长度由图中的marker数量决定
            DESCRIPTION.
        ids : tuple, ids元组
            DESCRIPTION.

        '''
        corners, ids, _ = aruco.detectMarkers(img, aruco_dict)
        centroid=tuple([np.int32([sum(i[0][:,1])*0.25, sum(i[0][:,0])*0.25]) for i in corners])
        if draw==True and ids is not None:
            aruco.drawDetectedMarkers(img,corners,ids)
        return centroid,ids
    
    def aruco_pose(self,aruco_dict,img,cam_matrix,dist_coeff,size=0.05,draw=False):
        '''
        图片中marker的姿态

        Parameters
        ----------
        aruco_dict : obj,参见函数aruco_createDict
            DESCRIPTION.
        img : array, 输入图片
            DESCRIPTION.
        cam_matrix : array, 相机内参矩阵
            DESCRIPTION.
        dist_coeff : array, 相机畸变系数
            DESCRIPTION.
        size : float, optional, 实际marker的边长,单位为m
            DESCRIPTION. The default is 0.05.
        draw : bool, optional, 绘制marker的姿态
            DESCRIPTION. The default is False. 直接绘制原图, 谨慎使用

        Returns
        -------
        rvec : list, marker的旋转矢量
            DESCRIPTION.
        tvec : list, marker的平移矢量
            DESCRIPTION.
        ids : list, marker的ids
            DESCRIPTION.

        '''
        corners, ids, _ = aruco.detectMarkers(img, aruco_dict)
        #rvec 为旋转向量,不随markerlength变化,单位为rad,而非欧拉角.tvec为平移向量,随markerlength变化,单位为m.markerlength为实际纸上marker的边长,用尺量,单位为m
        rvec, tvec,_= aruco.estimatePoseSingleMarkers(corners,size , cam_matrix, dist_coeff)
        if draw==True and ids is not None:
            #画出aruco姿态
            aruco.drawDetectedMarkers(img, corners,ids)
            #最后一个参数为绘制的轴的长度
            aruco.drawAxis(img, cam_matrix, dist_coeff,rvec,tvec, 0.03)
        return rvec,tvec,ids
    
    def aruco_ids2Dict(self,ids,src):
        '''
        将具有相同ids的marker的信息, 归入一个ids字典下, 传参时ids与src的长度必须相等
        例: 现有marker1,marker2,marker3, 三者对应中心点 centroid1, centroid2, centroid3, 但对应不同aruco
            marker1: ids=1, centroid=np.array([0,0])
            marker2: ids=0,centroid=np.array([10,10])
            marker3: ids=1, centroid=np.array([55,55])
        则,通过该函数,将得到字典:  {'0':np.array([55,55]),
                                  '1':np.array([0,0]),np.array([10,10])}
            
            
        Parameters
        ----------
        ids : tuple or list or array, marker的ids
            DESCRIPTION.
        src : tuple or list or array, 需要归类的src
            DESCRIPTION.

        Returns
        -------
        ids_dict : dict key=ids, value=src的字典
            DESCRIPTION.

        '''
        src=np.array(src)
        ids_dict={}
        for i in np.unique(ids):
            index=np.where(i==ids)[0]
            ids_dict0={str(i):src[index]}
            ids_dict.update(ids_dict0)
        return ids_dict
        

    def handeye_affineCalculation(self,cam_points,rob_points,eye_in_hand=True,th_1=0,th_2=0):
        cam_points=np.float32(cam_points)
        rob_points=np.float32(rob_points)
        # hand_in_eye 下, 首先应保证机械臂移动点相对于一个中心点对称,即机械臂应按s型路线走"田", 如此便能保证所有机械臂点关于'田'中心点对称
        # 这是因为由于相机的镜像关系,如果将机械臂点与相机点顺序成对(即rob_point1,cam_point1|rob_point2,cam_point2|...)送入公式进行仿射矩阵的计算, 如将新点带入会使机械臂向新点的镜像位置行进
        # 该镜像错误有如下解决办法:
        # 1.将机械臂点与相机点倒序成对带入(即rob_point1,cam_point9|rob_point2,cam_point8|...),直接可求出正确的仿射矩阵
        # 2.将机械臂点与相机点顺序成对带入(即rob_point1,cam_point1|rob_point2,cam_point2|...), 但需对仿射矩阵作如下处理:
        #   R-旋转部分四个元素取负(即对affine_matrix[:,:2]取负,取负的几何意义为将矩阵旋转180度)
        #   T-平移部分tx'=2*cx-tx,ty'=2*cy-ty, 其中(cx,cy)为机械臂采样点里的中心点坐标, tx,ty分别为顺序成对代入求出的仿射矩阵的affine_matrix[0,2],affine_matrix[1,2]
        #   R:affine_matrix[:,:2]=-affine_matrix[:,:2]
        #   T:affine_matrix[0,2]=2*cx-affine_matrix[0,2],affine_matrix[1,2]=2*cy-affine_matrix[0,2]
        # 此外, 每次工作,机械臂必须回到中心点拍照
        
        # hand_to_eye下, 唯一的要求为:采样阶段应尽量保持机械臂末端贴近工作平面, 并保证同一平面上取9个点(位置任意,不需保证对称),工作时也无需回到某个固定点
        
        if len(cam_points) == len(rob_points) and (len(cam_points) and len(rob_points)>=3):
            if eye_in_hand==True:
                cam_points_inverse=cam_points[::-1,:]
                affine_matrix=cv2.estimateAffine2D(cam_points_inverse,rob_points)[0]
            else:
                affine_matrix=cv2.estimateAffine2D(cam_points,rob_points)[0]
            affine_matrix[0,2]+=th_1
            affine_matrix[1,2]+=th_2
        else:
            affine_matrix=None
        if affine_matrix is not None:
            affine_matrix[0,2]+=th_1
            affine_matrix[1,2]+=th_2
        return affine_matrix
    
    def handeye_affineTransform(self,affine_matrix,cam_point):
        cam_point,affine_matrix=np.float32(cam_point).reshape(-1,2),np.float32(affine_matrix)
        if len(cam_point) > 2:
            rob_point = np.vstack((affine_matrix[0,0]*cam_point[:,0]+affine_matrix[0,1]*cam_point[:,1]+affine_matrix[0,2],
                                   affine_matrix[1,0]*cam_point[:,0]+affine_matrix[1,1]*cam_point[:,1]+affine_matrix[1,2])).T
        else:
            rob_point=None
        return rob_point
    
    def handeye_vuPredict(self,H,W,vu,z,z_calib):
        '''
        矫正由物体高度产生的像素坐标畸变, 需配合深度相机使用

        Parameters
        ----------
        H : int
            DESCRIPTION.图像的宽
        W : int
            DESCRIPTION.图像的高
        vu : array/tuple/list
            DESCRIPTION.需矫正的的点像素坐标
        z : float/int
            DESCRIPTION.需矫正的点的深度
        z_calib : float/int
            DESCRIPTION.标定板所处平面深度
        Returns
        -------
        array
            DESCRIPTION.矫正后的像素坐标

        '''
        vu=np.int64(vu).flatten() # 统一格式
        img_centroid=np.int64([H*0.5,W*0.5]) # 求图像中心点
        h,w=vu[0]-img_centroid[0],vu[1]-img_centroid[1]
        d = self.distance(vu,img_centroid)
        d_predict=d*z/z_calib
        u_predict=w*d_predict/d+img_centroid[1]
        v_predict=h*d_predict/d+img_centroid[0]
        return np.int64([v_predict,u_predict]).flatten()
    
    def handeye_createRobPoints(self,c1,c2,t):
        rob_points=np.array([[c1-t,c2+t],
                             [c1-t,c2  ],
                             [c1-t,c2-t],
                             [c1  ,c2-t],
                             [c1  ,c2  ],
                             [c1  ,c2+t],
                             [c1+t,c2+t],
                             [c1+t,c2  ],
                             [c1+t,c2-t]])
        return rob_points
    
    def handeye_getCamPoints(self,imgs,aruco_dict,show=False):
        '''
        提取多个图片中aruco的中心点像素坐标(仅识别单个aruco), 若单张图片中识别到多个aruco,或未识别到aruco
        函数将打印'无效采样',单张图片存在单个aruco则为有效采样,函数不做任何反馈
        该函数主要适用于手眼标定
        
        依赖:aruco_centroid,labelPoint
        Parameters
        ----------
        imgs : 输入图片
            DESCRIPTION.需以元组,列表的形式传入
        aruco_dict : aruco字典
            DESCRIPTION. 参考aruco_createDict
        show : TYPE, 是否显示结果
            DESCRIPTION. The default is False.

        Returns
        -------
        cam_points : 元组,中心点
            DESCRIPTION. 例: 若输入9中图片中有效采样数为8,则cam_point为长度为8的元组
        valid : 元组,有效采样的索引
            DESCRIPTION. 

        '''
        centroids,valid=(),()
        for i in range(len(imgs)):
            centroid,ids=ct.aruco_centroid(imgs[i],aruco_dict)
            if ids is not None and len(centroid)==1:
                centroids+=(centroid[0],)
                valid+=(i,)
                if show==True:
                    for j in centroids:
                        self.labelPoint(imgs[i],j)
                    cv2.polylines(imgs[i],[np.int0(centroids)[:,::-1]],False,(0,255,255),thickness=2)
                    cv2.imshow('img',imgs[i])
                    cv2.waitKey(500)     
            else:
                print(i,'无效采样')
        cv2.destroyAllWindows()
        print('有效采样共',len(valid),'个')
        cam_points=np.array(centroids)
        return cam_points,valid
    
    def handeye_tsai(self,H_grid2cam,H_base2tool,eye_in_hand=True):
        def skew(vector):
            vector=vector.flatten()
            return np.array([[0, -vector[2], vector[1]], 
                             [vector[2], 0, -vector[0]], 
                             [-vector[1], vector[0], 0]])
        
        S,b,=(),()
        A_tuple,B_tuple=(),()
        I=np.eye(3)
        
        # 计算R
        for i in range(len(H_grid2cam)-1):
            # 构造A,B 待解决, 顺序有错
            if eye_in_hand==False:
                A=np.linalg.inv(H_base2tool[i+1]).dot(H_base2tool[i])
                B=H_grid2cam[i+1].dot(np.linalg.inv(H_grid2cam[i]))
            else:
                A=np.linalg.inv(H_base2tool[i+1]).dot(H_base2tool[i])
                B=np.linalg.inv(H_grid2cam[i+1]).dot(H_grid2cam[i])
            A_tuple+=(A,)
            B_tuple+=(B,)
            
            rgij,rcij=cv2.Rodrigues(A[:3,:3])[0],cv2.Rodrigues(B[:3,:3])[0]
            theta_gij,theta_cij=np.linalg.norm(rgij),np.linalg.norm(rcij)
            rngij,rncij = rgij/theta_gij,rcij/theta_cij
            Pgij,Pcij= 2*np.sin(theta_gij/2)*rngij,2*np.sin(theta_cij/2)*rncij
            
            S+=(skew((Pgij + Pcij)),)
            b+=(Pcij - Pgij,)
        S,b=np.vstack(S),np.vstack(b)
        pcg_prime=np.linalg.pinv(S).dot(b)
        pcg=2*pcg_prime/(np.sqrt(1+np.linalg.norm(pcg_prime)**2))
        pcg0=np.linalg.norm(pcg)*np.linalg.norm(pcg)
        
        R=(1-pcg0*0.5)*I+0.5*(pcg*pcg.T+np.sqrt(4-pcg0)*skew(pcg))
        
        # 计算T
        a0,b0=(),()
        for a,b in zip(A_tuple,B_tuple):
            RA=a[:3,:3]
            TB=b[:3,3]
            TA=a[:3,3]
            a0+=(RA-I,)
            b0+=((R.dot(TB))-TA,)
        a0,b0=np.vstack(a0),np.hstack(b0)
        T=np.linalg.pinv(a0).dot(b0)   
        H=np.eye(4)
        H[:3,:3],H[:3,3]=R,T.reshape(3)
        return H
    
    def handeye_navy(self,H_grid2cam,H_base2tool,eye_to_hand=True):
        def logMatrix(H):
            R=H[:3,:3]
            fi=np.arccos((R.trace()-1)/2)
            w=fi/(2*np.sin(fi))*(R-R.T)
            return np.array([[w[2,1],w[0,2],w[1,0]]]).reshape(3,1)
        I=np.eye(3)
        M=np.zeros((3,3))
        Ra=()
        Ta=()
        Tb=()
        
        # 求R
        for i in range(len(H_grid2cam)-1):
            if eye_to_hand==True:
                A=np.linalg.inv(H_base2tool[i+1]).dot(H_base2tool[i])
                B=H_grid2cam[i+1].dot(np.linalg.inv(H_grid2cam[i]))
            else:
                A=np.linalg.inv(H_base2tool[i+1]).dot(H_base2tool[i])
                B=np.linalg.inv(H_grid2cam[i+1]).dot(H_grid2cam[i])
            Ra+=(A[:3,:3],)
            Ta+=(A[:3,3].reshape(3,1),)
            Tb+=(B[:3,3].reshape(3,1),)
            alpha,beta=logMatrix(A),logMatrix(B)
            M=M+beta*alpha.T
        R=np.linalg.inv(linalg.sqrtm(M.T.dot(M))).dot(M.T)
        
        # 求T
        C,d=(),()
        for i in range(1,len(Ra)):
            C+=(I-Ra[i],)
            d+=(Ta[i]-R.dot(Tb[i]),)
        C,d=np.vstack(C),np.vstack(d)
            
        T=np.linalg.inv(C.T.dot(C)).dot(C.T.dot(d))
        H=np.eye(4)
        H[:3,:3],H[:3,3]=R,T.reshape(3)
        return H
    
    
    # 旋转相关函数
    # 以下函数名中 各字母意义
    # R:旋转矩阵(3x3)
    # T:平移向量(3x1)
    # E:欧拉角
    # Q:四元数
    def RT2H(self,R,T):
        T=np.array(T)
        H=np.eye(4)
        H[:3,:3],H[:3,3]=R,T.reshape(3)
        return H
    def E2R(self,seq,E,degree=False):
        r = rot.from_euler(seq, E, degrees=degree)
        return r.as_matrix()
    
    def Q2R(self,Q):
        r=rot.from_quat(Q)
        return r.as_matrix()
    
    def R2E(self,seq,R,degree=False):
        r=rot.from_matrix(R)
        return r.as_euler(seq, degrees=degree)
    
    def R2Q(self,R):
        r=rot.from_matrix(R)
        return r.as_quat()
    def Q2E(self,Q,seq,degree=False):
        r=rot.from_quat(Q)
        return r.as_euler(seq,degree)

    def circle_detection(self,img):
        #该算法极度依赖调参, 不建议使用
        #增加预处理
        #增加圆数量判断
        #增加中心点返回
        if img.ndim==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,2,100,param1=100,param2=50,minRadius=0,maxRadius=0)
        #param2越小检测到的圆越多,反之越少
        circles = np.uint16(np.around(circles))
        print(len(circles[0,:]),'个圆被检测到')
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        centroid=()
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
            centroid+=(np.array([i[0],i[1]]),)
        return img,centroid
    
    def depth2Sensor(self,vu,z,H,W):
        img_centroid=np.int64([H*0.5,W*0.5])
        d=self.distance(vu,img_centroid)
        z_sensor=np.sqrt((pow(d,2)+pow(z,2)))
        return z_sensor
    
    def alg_measure(self,vu1,vu2,depth_map,depth_cam_matrix,
                    depth_dist_coeff=np.zeros(5,dtype='float32'),
                    z1=None,z2=None,method=0,depth_scale=1000):
        if method==0:
            if z1==None:
                z1=depth_map[vu1[0],vu1[1]]
            if z2==None:
                z2=depth_map[vu2[0],vu2[1]]
            if z1==0 and z2==0:
                real_distance=None 
            else:
                vu = np.vstack((np.float64(vu1).reshape(1,2),
                                np.float64(vu2).reshape(1,2)))
                vu_undistort=cv2.undistortPoints(vu,depth_cam_matrix,depth_dist_coeff).reshape(-1,2)
                z1,z2=z1/depth_scale,z2/depth_scale
                x1,y1=vu_undistort[0,:]*z1
                x2,y2=vu_undistort[1,:]*z2
                real_distance=np.sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2))
        elif method==1:
            xyz=self.pc_depth2xyz(depth_map,depth_cam_matrix,False,depth_scale)
            xyz1,xyz2=xyz[vu1[0],vu1[1],:],xyz[vu2[0],vu2[1],:]
            # 用户自定义z1,z2
            if z1 is not None:
                xyz1[2]=z1/depth_scale
            if z2 is not None:
                xyz2[2]=z2/depth_scale
            # 判断深度是否为0            
            if xyz1[2]==0 or xyz2[2]==0:
                real_distance =None
            else:
                real_distance=self.distance(xyz1,xyz2)
        return real_distance
    
    def cvMultiProcess(self):
        cv2.setUseOptimized( True );
        cv2.setNumThreads( 4 );
    
    def socket_createServer(self,port,ip='host',connect_num=5): 
        ip =socket.gethostname() if ip=='host' else ip
        server=socket.socket()
        server.bind((ip, port))
        server.listen(connect_num)
        return server
    
    def socket_createClient(self,port,ip='host'):
        ip =socket.gethostname() if ip=='host' else ip
        client=socket.socket() 
        client.connect((ip, port))
        return client
    
    def distance(self,point1,point2):
        '''
        计算两个2d点或两个3d点的距离

        Parameters
        ----------
        point1 : array/list/tuple
            DESCRIPTION.
        point2 : array/list/tuple
            DESCRIPTION.

        Returns
        -------
        distance : float
            DESCRIPTION.

        '''
        point1,point2=np.float64(point1).flatten(),np.float64(point2).flatten()
        assert len(point1)==len(point2)
        if len(point1)==2 and len(point2)==2:
            x1,x2,y1,y2=point1[0],point2[0],point1[1],point2[1]
            distance=np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
        elif len(point1)==3 and len(point2)==3:
            x1,x2,y1,y2,z1,z2=point1[0],point2[0],point1[1],point2[1],point1[2],point2[2]
            distance=np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))            
        return distance
    
    def hsvTuner(self,img,save=False,save_dir='./HSV'):
        '''
        hsv调色板, 若save=True, 则保存h_upper,h_lower,s_upper,s_lower,v_upper,v_lower
        至save_dir下,需要用户输入文件名
        Parameters
        ----------
        img : array
            DESCRIPTION.输入图片
        save : bool, optional
            DESCRIPTION. The default is True.
        save_dir : str, optional
            DESCRIPTION. The default is './HSV'.

        Returns
        -------
        hsv_range : tuple
            DESCRIPTION.按照h_upper,h_lower,s_upper,s_lower,v_upper,v_lower顺序保存为txt

        '''
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        cv2.namedWindow('tuner', cv2.WINDOW_NORMAL)
        def callback(para):
            pass
        cv2.createTrackbar('h_upper','tuner',180,180,callback)
        cv2.createTrackbar('h_lower','tuner',0,180,callback)
        cv2.createTrackbar('s_upper','tuner',255,255,callback)
        cv2.createTrackbar('s_lower','tuner',0,255,callback)
        cv2.createTrackbar('v_upper','tuner',255,255,callback)
        cv2.createTrackbar('v_lower','tuner',0,255,callback)
        while 1:
            key=cv2.waitKey(1)&0xFF
            if key==27:
                break
            h_upper=cv2.getTrackbarPos('h_upper','tuner')
            h_lower=cv2.getTrackbarPos('h_lower','tuner')
            if h_lower>h_upper:
                cv2.setTrackbarPos('h_lower','tuner',h_upper)
            s_upper=cv2.getTrackbarPos('s_upper','tuner')
            s_lower=cv2.getTrackbarPos('s_lower','tuner')
            if s_lower>s_upper:
                cv2.setTrackbarPos('s_lower','tuner',s_upper)
            v_upper=cv2.getTrackbarPos('v_upper','tuner')
            v_lower=cv2.getTrackbarPos('v_lower','tuner')
            if v_lower>v_upper:
                cv2.setTrackbarPos('v_lower','tuner',v_upper)
            mask=cv2.inRange(hsv,(h_lower,s_lower,v_lower),(h_upper,s_upper,v_upper))
            img_mask=cv2.bitwise_and(img,img,mask=mask)
            cv2.imshow('tuner',img_mask)
        cv2.destroyAllWindows()
        hsv_range=(h_upper,h_lower,s_upper,s_lower,v_upper,v_lower)
        
        if save==True:
            if os.path.exists(save_dir)==False:
                os.mkdir(save_dir)
            name=input('请输入文件名: ')
            path=os.path.join(save_dir,name)
            if path.endswith('.txt'):    
                np.savetxt(path,hsv_range)
            else:
                np.savetxt(path+'.txt',hsv_range)
        return hsv_range, img_mask

    def hsvConverter(self,img,hsv_range):
        '''
        hsv掩膜工具

        Parameters
        ----------
        img : array
            DESCRIPTION.输入图片
        hsv_range : tuple
            DESCRIPTION.应按照(h_upper,h_lower,s_upper,s_lower,v_upper,v_lower)输入

        Returns
        -------
        img_mask : array
            DESCRIPTION.掩膜后的图片

        '''
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(hsv_range[1],hsv_range[3],hsv_range[5]),
                         (hsv_range[0],hsv_range[2],hsv_range[4]))
        img_mask=cv2.bitwise_and(img,img,mask=mask)
        return img_mask
    
    def poseEstimation(self,):
        pass
        
    def poseDraw(self,img,point_zero,xyz,thickness=2):
        point_zero=np.array(point_zero).flatten()
        xyz=np.array(xyz).reshape(3,2)
        cv2.arrowedLine(img, tuple(point_zero),tuple(xyz[0]), (255,0,0), thickness)
        cv2.putText(img,'x',tuple(xyz[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),thickness)
        cv2.arrowedLine(img, tuple(point_zero), tuple(xyz[1]), (0,255,0), thickness)
        cv2.putText(img,'y',tuple(xyz[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),thickness)
        cv2.arrowedLine(img, tuple(point_zero),tuple(xyz[2]), (0,0,255),thickness)
        cv2.putText(img,'z',tuple(xyz[2]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness)
        
    def labelPoint(self,img,vu,size=10,color=(0,255,0),thickness=2):
        # 标记vu
        vu=np.int64(vu).flatten()
        cv2.drawMarker(img,(vu[1],vu[0]),color,cv2.MARKER_CROSS,size,thickness)
                
    def labelPointManully(self,img,draw_line=False,window_name='img'):
        # 手动标记vu
        # 按左键标记,按右键删除上一个点
        if img.ndim==2:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        p=[]
        cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
        cv2.imshow(window_name,img)
        def draw_point(event, x, y, flags, param):
            img_draw=img.copy()
            if event==cv2.EVENT_LBUTTONUP:
                p.append([x,y])
                for u,v in np.array(p):
                    cv2.drawMarker(img_draw,(u,v),(0,255,0),cv2.MARKER_CROSS,10,2)
                if draw_line==True:
                    cv2.polylines(img_draw,[np.array(p)],False,(0,255,255),thickness=2)
                cv2.imshow(window_name,img_draw)
            if event==cv2.EVENT_RBUTTONUP:
                if len(p)!=0:
                    p.pop()
                    if draw_line==True:
                        if len(p)!=0:
                            cv2.polylines(img_draw,[np.array(p)],False,(0,255,255),thickness=2)
                    for u,v in np.array(p):
                        cv2.drawMarker(img_draw,(u,v),(0,255,0),cv2.MARKER_CROSS,10,2)
                cv2.imshow(window_name,img_draw)
            if event==cv2.EVENT_MOUSEMOVE:
                cv2.setWindowTitle(window_name,str(y)+' , '+str(x))
        cv2.setMouseCallback(window_name,draw_point)
        cv2.waitKey()
        cv2.destroyAllWindows()
        p=np.array(p)[:,::-1] if len(p)!=0 else None # uv到vu
        return p
    
    def text(self,img,vu,text,delta_vu=(10,10),size=1,color=(255,0,0),thickness=1):
        vu=np.int64(vu).flatten()+delta_vu
        cv2.putText(img,str(text),(vu[1],vu[0]),cv2.FONT_ITALIC,size,color,thickness)
        
    def getDepthRanges(self,depth_map,th = 1000, hist_show=False):
        max_depth=int(depth_map.max())
        hist = cv2.calcHist([depth_map],[0],None,[max_depth],[1,max_depth]).flatten()
        
        hist[hist<th]=0
        if hist_show==True:
            plt.plot(hist,color="g", )
            plt.show()
        hist_nonzero=hist.nonzero()[0]
        depth_ranges,depths=(),()
        depth_range =[hist_nonzero[0]]
        for i in range(len(hist_nonzero)-1): #前景
            if hist_nonzero[i+1]-hist_nonzero[i]>2:
                depth_range.append(hist_nonzero[i])
                if depth_range[1]-depth_range[0]>=3:
                    depth_ranges+=(depth_range,)
                depth_range=[hist_nonzero[i+1]]
            if i==len(hist_nonzero)-2: # 背景
                depth_range.append(hist_nonzero[i+1])
                if depth_range[1]-depth_range[0]>=3:
                    depth_ranges+=(depth_range,)
        depths=tuple(i[0]+np.argmax(hist[i[0]:i[1]]) for i in depth_ranges)
        return depth_ranges,depths
    
    def depthRange2Mask(self,depth_map,depth_range):
        mask=cv2.inRange(depth_map,int(depth_range[0]),int(depth_range[1]))
        return mask
    
    def maskImg(self,img,mask):
        img_mask=cv2.bitwise_and(img,img,mask=mask)
        return img_mask 
        
    def sharp(self,img,method=0,iterations=1):
        if method ==0:
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]) # lap5
        elif method == 1:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) #lap9
        elif method ==2:
            kernel = np.array([[-1, -2, -1], [-2, 19, -2], [-1, -2, -1]])/7 #lap9

        img_sharp = img.copy()
        for i in range(iterations):
            img_sharp = cv2.filter2D(img_sharp,cv2.CV_8U,kernel)
        return img_sharp


    def alg_depthMaskPeak(self,color_img,depth_map,lv=0,show_hist=False,distance=20,prominence=500,width=1,height=1,threshold=1):   
        # 待优化
        # 该算法鲁棒性不高,建议使用getDephRange
        max_value=int(depth_map.max())
        hist = cv2.calcHist([depth_map],[0],None,[max_value],[1,max_value-1]).flatten()
        
        #nonzero_index=hist.nonzero()[0]
        #hist_nonzero=hist[nonzero_index]
        
        # 寻找波峰
        peaks, properties=signal.find_peaks(hist,
                                            prominence=prominence,
                                            distance=distance,
                                            width=width,
                                            height=height,
                                            threshold=threshold)
        if show_hist==True:
            plt.plot(hist,color="g", )
            plt.plot(peaks,hist[peaks],'*r')
            plt.show()
        
        
        # 计算每一个波峰的区间
        border=()
        border0=np.zeros(2,dtype='int')
        for left, right in zip(properties['left_bases'], properties['right_bases']):
            border0[0],border0[1]=left,right
            border+=(border0,)
            border0=np.zeros(2,dtype='int')
        depth=[i[0]+np.argmax(hist[i[0]:i[1]]) for i in border]
        mask=cv2.inRange(depth_map,int(border[lv][0]),int(border[lv][1]))
        img_mask=cv2.bitwise_and(color_img,color_img,mask=mask)
        return img_mask,depth[lv]
    
    def alg_statisticThreshold(self,img):
        if img.ndim==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
        kernelx,kernely=np.array([[-1,0,1]]),np.array([[-1,0,1]]).T
        dx,dy = abs(cv2.filter2D(img, -1, kernelx)),abs(cv2.filter2D(img, -1, kernely))
        dmax=np.maximum(dx,dy)
        weight=np.sum(dmax)
        total=np.sum(dmax*img)
        threshold=int(total/weight)
        img_bin=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)[1]
        return img_bin
    
    def alg_DerekBradleyThreshold(self,img,s=30,t=15):
        if img.ndim==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_bin=img.copy()
        img_integral=cv2.integral(img_bin)[1:,1:]
        kernel=np.zeros((s,s))
        kernel[0,0],kernel[-1,-1],kernel[0,-1],kernel[-1,0]=1,1,-1,-1
        img_conv=cv2.filter2D(img_integral,-1,kernel,borderType=1)
        area=s*s
        weight=img.astype('int64')*area
        brightness=img_conv*((100-t)/100)
        img_bin[weight<=brightness],img_bin[weight>brightness]=0,255
        return img_bin
    
    def meanShift(self,ps):
        # 该函数主要应用于特征点聚类,以用于确定特征点位置
        bandwidth = estimate_bandwidth(ps, quantile=0.1, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
        ms.fit(ps)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        return labels,cluster_centers
    
    def getNearestNonzero(self,img,vu):
        # 搜索最近的非零像素点, img-输入图像,必须为单通道, vu目标点
        # vu_value-长度为3的array 前两个值为 最近非零点的像素坐标, 第三个值为该坐标的至
        uv_nonzero = cv2.findNonZero(img)
        if uv_nonzero is not None:
            vu_nonzero=uv_nonzero.reshape(-1,2)[:,::-1] # uv转vu
            distances = np.sqrt((vu_nonzero[:,0]-vu[0])**2+(vu_nonzero[:,1]-vu[1])**2)
            vu_nearest=vu_nonzero[np.argmin(distances)]
            value=img[vu_nearest[0],vu_nearest[1]]
            vu_value=np.hstack((vu_nearest,value))
        else:
            vu_value=None
        return vu_value
    
    def FOV2Area(self,FOV,height):
        H,V,D=FOV
        area_W = np.tan(V*np.pi/180) * height
        area_H = np.tan(H*np.pi/180) * height
        return area_W, area_H
    
    def cnt_getChildParent(self,cnts,hierarchy):
        cnts_child,cnts_parent=(),()
        for i in range(len(cnts)):
            if hierarchy[0,i,3] == -1:
                cnts_parent+=(cnts[i],)
            else:
                cnts_child +=(cnts[i],)
        return cnts_child,cnts_parent
    
    def cnt_getBBox(self,cnt):
        rect = cv2.minAreaRect(cnt)
        bbox_uv = np.int0(cv2.boxPoints(rect))
        angle = rect[2]
        return bbox_uv, angle
    
    def cnt_getBBoxSize(self,bbox):
        d1=self.distance(bbox[0],bbox[1])
        d2=self.distance(bbox[1],bbox[2])
        if d1>d2:
            w,h = d1, d2
        else:
            w,h = d2, d1
        return w,h
    def cnt_getBBoxCentre(self,bbox_uv):
        centre = np.array([sum(bbox_uv[:,0])*0.25,sum(bbox_uv[:,1])*0.25])
        return centre

    
    def feature_match(self,img_source,img_target,detector,matcher,count=10,show=False):
        # kps : keypoints, dps: descriptors
        # s: source , t: target
        # 封装过度
        kps_s,dps_s =  detector.detectAndCompute(img_source,None)
        kps_t,dps_t =  detector.detectAndCompute(img_target,None)
        matched_kps = matcher.match(dps_s,dps_t)
        matched_kps_optimized=sorted(matched_kps,key=lambda x:x.distance)[:count]
        matched_kps_s,matched_kps_t=(),()
        for i in matched_kps_optimized:
            matched_kps_s+=(kps_s[i.queryIdx].pt,)
            matched_kps_t+=(kps_t[i.trainIdx].pt,)
        matched_kps_s=np.float32(matched_kps_s)
        matched_kps_t=np.float32(matched_kps_t)
        if show==True:
            img_draw=cv2.drawMatches(img_source,kps_s,
                                     img_target,kps_t,
                                     matched_kps_optimized,
                                     None,
                                     matchColor=(0,255,0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            self.show(img_draw)
        return matched_kps_s,matched_kps_t
    
    def feature_optimization(self,matched_kps_s, matched_kps_t):
        M, mask = cv2.findHomography(matched_kps_s, matched_kps_t, cv2.RANSAC, 5.0)
        return M,mask
    
    def feature_kps2Numpy(self,kps):
        kps_numpy = tuple(i.pt for i in kps)
        return kps_numpy
    
    def feature_labelKpsDps(self,kps,dps,labels):
        kps_labels,dps_labels=(),()
        label_count=labels.max()+1
        for i in range(label_count):
            label=np.where(labels==i)[0]
            kps_label= tuple(kps[j] for j in label)
            kps_labels+=(kps_label,)
            dps_labels+=(dps[label],) 
        return kps_labels,dps_labels
            
    def perspectImg(self,img,ps_s_uv):
        p1,p2,p3=ps_s_uv[0],ps_s_uv[1],ps_s_uv[2]
        d1,d2=self.distance(p1,p2),self.distance(p2,p3)
        if d1>d2:
            w,h=d1,d2
        else:
            w,h=d2,d1
        ps_t=np.float32([[0,0],[0,w],[h,w],[h,0]])
        M=cv2.getPerspectiveTransform(np.float32(ps_s_uv),ps_t)
        img_out=cv2.warpPerspective(img,M,(int(h),int(w)))
        return img_out,M,ps_t
    
#    def perspectImgReverse(self)
    
    def perspectPs(self,ps_s,M):
        ps_t = cv2.perspectiveTransform(ps_s.reshape(-1,1,2),M)
        return ps_t
    
    def matchKFeatures():
        pass
    
    def roiShotCut(self,img,save_dir='./data/roi'):
        # 依赖cv_roi2img
        print('按任意键输入名称,按esc键退出')
        if os.path.exists(save_dir)==False:
            os.mkdir(save_dir)
        img_dict={}
        while 1:
            roi,key=self.roiSelect(img)
            if key == 27:
                break
            else:
                img_roi=self.roi2img(img,roi)
                if roi.all()!=0:
                    name=input('请输入文件名(仅支持英文输入): ')
                    if len(name)!=0:
                        img_dict.update({name:img_roi})
                        path=os.path.join(save_dir,name) \
                        if name.endswith('.jpg') else os.path.join(save_dir,name+'.jpg')
                        print(path)
                        cv2.imwrite(path,img_roi)
                        print('已保存图片至',path)
                    else:
                        continue
                else:
                    continue
        return img_dict
    
    def roiSelect(self,img,window_name='img',thickness=2,bgr=(0,0,255)):
        # 选取矩形roi,返回(4,)向量,分别为(v1,u1,v2,u2)
        # v1,u1为矩形左上角点, v2,u2为矩形右下角点
        def draw_rect(event, x, y, flags, param):
            if event==cv2.EVENT_LBUTTONDOWN:
                roi[0]=y,x # 框的左上角点 x=u,y=v
            elif event == cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
                roi[1]=y,x # 框的右下角点 x=u,y=v
                img_draw=cv2.rectangle(img.copy(),(roi[0][1],roi[0][0]),
                                       (roi[1][1],roi[1][0]),bgr,thickness)
                cv2.imshow(window_name,img_draw)
        if img.ndim==2:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
        cv2.imshow(window_name,img)
        roi=np.zeros((2,2),dtype='int64')
        cv2.setMouseCallback(window_name, draw_rect)
        key=cv2.waitKey()
        cv2.destroyAllWindows()
        roi[roi<0]=0
        # 由于roi已定义roi[0]为左上角点,roi[1]为右下角点
        # 当用户画框时从非左上到右下的放下画时,即从右下往左上,或左下到右上等,就会导致
        # roi[0] 为非左上角点,roi[1] 为非右下角点
        # 故产生如下判断
        if roi[0,0] > roi[1,0]:
            roi=roi[::-1,:]
        if roi[0,1] > roi[1,1]:
            roi[:,1] = roi[:,1][::-1]

        roi=roi.flatten()
        return roi,key


    def roiSelectExpand(self,img,selection=True,save_read_dir='./data/roi'):
        # 依赖roiSelect, roiselect的二次封装
        # 如果选取了roi,则保存到save_read_dir下roi.txt,若已存在roi.txt,则会被覆盖
        # 如果未选取将读取save_read_dir下的roi.txt
        # 如果save_read_dir未存在roi.txt, 又未选取roi,则返回(0,0,H,W)
        if os.path.exists(save_read_dir)==False:
            os.mkdir(save_read_dir)
        name_roi=os.path.join(save_read_dir,'roi.txt')
        if os.path.exists(name_roi)==False:
            selection=True
            print('未检测到',save_read_dir,'中存在框选参数,强制进入框选')
        if selection==True:
            print('按esc键退出')
            roi,key=self.roiSelect(img)
            if key==27:
                print('已退出框选,函数将返回原图')
                roi=np.array([0,0,img.shape[0],img.shape[1]])
            else:
                if roi.all()==0:
                    if os.path.exists(name_roi)==True:
                        roi=np.int64(np.loadtxt(name_roi))
                        print('未检测到框选,将读取上次框选参数')
                    elif os.path.exists(name_roi)==False:
                        print('未检测到框选,且',save_read_dir,'中未找到框选参数,函数将返回原图')
                        roi=np.int64([0,0,img.shape[0],img.shape[1]])
                else:
                    np.savetxt(name_roi,roi)
                    print('已保存框选参数至',name_roi)
        else:
            print('将读取',name_roi,'作为此次运行的框选参数')
            roi=np.loadtxt(name_roi)        
        return roi

    def roi2img(self,img,roi,padding=False):
        # 待修改
        # 提取img的roi区域,padding=True则非roi区域填充为黑色, 即图片将保持原大小
        # padding=False则直接返回roi大小的图片
        v1,u1,v2,u2=int(roi[0]),int(roi[1]),int(roi[2]),int(roi[3])
        if padding== True:
            img_roi=np.zeros_like(img)
            if img.ndim==3:
                img_roi[v1:v2,u1:u2,:]=img[v1:v2,u1:u2,:]
            elif img.ndim==2:
                img_roi[v1:v2,u1:u2]=img[v1:v2,u1:u2]
        else:
            if img.ndim==3:
                img_roi=img[v1:v2,u1:u2,:]
            elif img.ndim==2:
                img_roi=img[v1:v2,u1:u2]
        return img_roi

        
    def CamIntr2CamMatrix(self,fx,fy,cx,cy):
        cam_matrix=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        return cam_matrix
    
    def alg_AdaptiveThreshold(self,img, val1=10,val2=2):
        if img.ndim==3:   
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if(val1 < 3):
            val1 = 3
        if(val1 % 2 == 0):
            val1 = val1 + 1
        img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY_INV, val1, val2)
        return img_bin
    
    
    def cntProcess(self,cnt,th_area=0,th_lengh=0,th_len=0):
        cnt_process=()
        for i in range(len(cnt)):
            if cv2.contourArea(cnt[i])<=th_area or cv2.arcLength(cnt[i],closed=False)<=th_lengh or len(cnt)<=th_len:
                continue
            else:
                cnt_process+=(cnt[i],)
        return cnt_process
    
    def cntExtract(self,img,global_th=False):
    # 图像处理
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        if img.ndim==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_mb=cv2.medianBlur(img,9)
        if global_th==True:
            _,img_bin=cv2.threshold(img_mb,0,255,cv2.THRESH_OTSU)
        else:
            img_bin=self.alg_AdaptiveThreshold(img_mb,10,8)
        img_close=cv2.morphologyEx(img_bin,cv2.MORPH_CLOSE,kernel,iterations=1)
        # 边缘处理
        _,cnt,_= cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt=self.cntProcess(cnt,th_area=1000)
        #    print('共检测到',len(cnt),'个边缘')
        return cnt
    
    def cntInfo(self,cnt,img_draw,th_area=0,draw_boundingbox=False,draw_centroid=True,draw_cnt=False):
        #依赖cv_LabelPoint
        img_draw=img_draw.copy()
        centroid=()
        angle_degree=()
        cnt_process=()
        for i in range(len(cnt)):
            if cv2.contourArea(cnt[i])<=th_area:
                continue
            else:
                cnt_process+=(cnt[i],)
                M = cv2.moments(cnt[i])
                centroid0=[int(M['m01'] / M['m00']), int(M['m10'] / M['m00'])]
                centroid+=(centroid0,)
                rect = cv2.minAreaRect(cnt[i])
                angle_degree+=(round(rect[2],1),)
                box = np.int0(cv2.boxPoints(rect))
                if draw_boundingbox==True:
                    cv2.drawContours(img_draw, [box], -1,(255,0, 0), 2)
                if draw_centroid==True:
                    self.labelPoint(img_draw,centroid0)
        if draw_cnt==True:
            cv2.drawContours(img_draw,cnt_process, -1,(0,0, 255), 2)
    #        approx = cv2.approxPolyDP(contours[i], 100, True)
    #        cv2.polylines(img, [approx], True, (0, 255, 0), 2)
    #        cv2.fitLine
        return img_draw,cnt_process,centroid,angle_degree
        
    def selectFilesWithDialog(self,suffix='.*',init_dir='./template'):
        root = tk.Tk()
        root.withdraw() # 隐藏tk自带弹窗
        file_path = filedialog.askopenfilenames(initialdir=init_dir,filetypes=[(suffix,suffix)])
        return file_path
        
    def readFiles(self,read_dir='./template',selection=False,suffix='*'):
        # 依赖 selectFiles
        files_path=()
        files_name=()
        if selection==True:
           files_path=self.selectFiles(suffix)
        if len(files_path)==0:
            print('未选择文件,将读取',read_dir,'下所有',suffix,'文件')
            files_path=()
            for i in os.listdir(read_dir):
                if i.endswith(suffix)==True:
                    files_path+=(''.join([read_dir,'/',i]),)
                    files_name+=(i,)
            print('文件夹',read_dir,'下检索到', suffix,'文件有',files_name)
        if len(files_path)==0:
            print('未在',read_dir,'下找到任何', suffix,'文件')
        return files_path
    
    def readDict(self,npy_dict_path):
        return np.load(npy_dict_path,allow_pickle=True).item()
    
    
    def showExpand(self,img,window_name='img',draw_centroid=False):
        if type(img) is not np.ndarray:   
            img=list(img)
            W,H,dim=(),(),()
            for i in img:
                W+=(i.shape[1],)
                H+=(i.shape[0],)
                dim+=(i.ndim,)
            W,H,dim=max(W),max(H),max(dim)
            for i in range(len(img)):
                if img[i].ndim!=dim:
                    img[i]=cv2.cvtColor(img[i], cv2.COLOR_GRAY2BGR)
                if img[i].shape[1]!=W or img[i].shape[0]!=H:
                    img[i]=cv2.resize(img[i],(W,H))
                if draw_centroid==True:
                    self.labelPoint(img[i],(int(H/2),int(W/2)))
            img=np.hstack((img))
        else:
            if draw_centroid==True:
                self.labelPoint(img,(img.shape[0]/2,img.shape[1]/2))
        cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
        try:
            cv2.imshow(window_name,img)
            cv2.waitKey()
        except:
            cv2.destroyAllWindows()
            print('error!')
        cv2.destroyAllWindows()
        
    def show(self,img,window_name='img',method = 0):
        cv2.namedWindow(window_name,method)
        cv2.imshow(window_name,img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def dshow(self,img):
        if img.ndim==3:
            plt.imshow(img[:,:,::-1])
        elif img.ndim==1:
            plt.imshow(img,cmap='gray')
        plt.show()

        
    def readImgs(self,read_dir='./',suffix='.jpg',name = None):
        # 读取文件夹下所有图片
        name_list = os.listdir(read_dir)
        
        if name is not None:
            img=tuple(cv2.imread(os.path.join(read_dir,i),-1) for i in name_list \
                     if i.endswith(suffix) and name in i)
        else:
            img=tuple(cv2.imread(os.path.join(read_dir,i),-1) for i in name_list \
                     if i.endswith(suffix))        
        return img
    
    def readColorDepth(self, read_dir='./realsense'):
        # 读取文件夹下所有的 color_img,depth_map,depth_img
        color_img=()
        depth_map=()
        depth_img=()
        
        for i in os.listdir(read_dir):
            path=os.path.join(read_dir,i)
            if 'color_img' in i and '.jpg' in i:
                color_img+=(cv2.imread(path),)
            elif 'depth_map' in i and '.png' in i:
                depth_map+=(cv2.imread(path,-1),) #cv2.IMREAD_ANYDEPTH=2
            elif 'depth_img' in i and '.jpg' in i:
                depth_img+=(cv2.imread(path),)
        return color_img, depth_map, depth_img  
    
    def readCamPara(self,cam_para_path):
        # 依赖readjson
        cam_para=self.readJson(cam_para_path)
        for key,value in zip(cam_para.keys(),cam_para.values()):
            if 'cam_matrix' in key:
                value=np.array(value).reshape(3,3) \
                if len(value)!=3 else np.array(value)
                cam_para.update({key:value})
            elif 'dist_coeff' in key:
                cam_para.update({key:np.array(value)})
        return cam_para
        
    
    def readJson(self,path):
        # 读取json文件
        with open(path,'r') as f:
            src = json.load(f)
        return src
    
    def readJsons(self,read_dir,name=None):
        src=()
        for i in os.listdir(read_dir):
            if i.endswith('.json'):
                if name is not None:
                    if name in i:
                        with open(os.path.join(read_dir,i),'r') as f:
                            src+=(json.load(f),)
                else:
                    with open(os.path.join(read_dir,i),'r') as f:
                        src+=(json.load(f),)
        return src
    
    def saveJson(self,src,save_path):
        if not save_path.endswith('.json'):
            save_path = ''.join([save_path,'.json'])
        with open(save_path,'w') as f:
            json.dump(src,f,indent=1)
            
    def saveImg(self,img,num=0,name='',suffix='.jpg',zfill=4,save_dir ='./realsense'):
        path=''.join([save_dir,'/',name,str(num).zfill(zfill),suffix])
        res=cv2.imwrite(path,img)
        return res
        
    def saveDict2txt(self,src,path):
        with open(path,'w') as f:
            for key,value in src.items():
                f.write(str(key)+'\n'+str(value)+'\n \n')
    
    def createDir(self,dir_name):
        os.makedirs(dir_name,exist_ok=True)

    def video2img(self,video_path,save_dir='./frame', zfill=4):
        os.makedirs(save_dir,exist_ok=True)
        video = cv2.VideoCapture(video_path)
        if video.isOpened():
            num=0
            while 1:
                success, img = video.read()
                if success:
                    name =''.join([save_dir,'/','frame',str(num).zfill(zfill),'.jpg' ])
                    cv2.imwrite(name,img)
                    num+=1
                else:
                    break
        video.release()
    
    def h52img(self,h5_path,save_dir='./'):
        h5 = h5py.File(h5_path, 'r')
        os.makedirs(save_dir,exist_ok=True)
        for key in h5.keys():
            img=cv2.imdecode(h5[key].value,-1)
            cv2.imwrite(save_dir+h5[key].name,img)
        h5.close()

    def camcalib_getCamPara(self,imgs,shape_corner,show=False):
        '''
        相机标定,获取相机内参畸变系数,标定板采用棋盘格, 棋盘格行列数不应相等,否则将导致精度下降
        标定形式可分为两种:
            1.相机固定, 移动标定板
            2.标定板固定,移动相机
        经实测,2具有更高精度
        Parameters
        ----------
        img : 含有棋盘格的图像
            DESCRIPTION.支持元组或列表输入, 即多张图片应保存为元组或列表
        shape_corner : 棋盘格角点个数
            DESCRIPTION.若标定采用(7,7)的棋盘格即7行7列, 则该参数应输入(6,6)即应为(行-1,列-1), 

        Returns
        -------
        cam_para : 字典, 包含RMS, cam_matrix, dist_coeff, rvecs, tvecs

        '''
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        w,h=shape_corner[1],shape_corner[0]
        objp = np.zeros((w*h,3), np.float32)
        # solvepnp 中需使用uv及xyz,在此标定算法中默认z=0, objp中前两列为x,y
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        objpoints = [] 
        imgpoints = []
        for i in imgs:
            gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
            if ret == True:
                objpoints.append(objp) #更新xy
                # 求亚像素以获得更高的像素精度
#                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners) #更新uv
                if show==True:
                    cv2.drawChessboardCorners(i, (w,h), corners, ret)
                    cv2.imshow('img', i)
                    cv2.waitKey(500)
        cv2.destroyAllWindows()
        # RMS 标定的均方根误差, 一次合格的标定--参考范围[0.1,1], 若超出该范围应增加采样数量以提高标定精度
        if len(imgpoints)!=0 or len(objpoints)!=0:
            RMS,cam_matrix,dist_coeff,rvec,tvec=cv2.calibrateCamera(objpoints, 
                                                                    imgpoints, 
                                                                    gray.shape[::-1], 
                                                                    None, None)
            
            cam_para={'cam_matrix':cam_matrix,
                      'dist_coeff':dist_coeff,
                      'RMS':RMS,
                      'rvec':np.array(rvec).reshape(-1,3),
                      'tvec':np.array(tvec).reshape(-1,3)}
        return cam_para
    
    def camcalib_saveJson(self,cam_para,save_path):
        cam_para_json={}
        for key,value in cam_para.items():
            if key=='RMS':
                cam_para_json.update({key:value})
            else:
                cam_para_json.update({key:value.tolist()})
        self.saveJson(cam_para_json,save_path)
        
    
    def camcalib_createChessBoard(self,shape_square,size=0.02):

        '''        
        Parameters
        ----------
        num : 方块行数及列数, optional
            DESCRIPTION. 
        size : 单个方块的实际边长,单位m, optional
            DESCRIPTION. The default is 0.02.

        Returns
        -------
        None.

        '''
        size_real=int(1000*3.78*size)
        w,h=shape_square[1],shape_square[0]
        board=np.fromfunction(lambda i, j: (i//size_real)%2 != (j//size_real)%2,
                              (h*size_real,w*size_real)).astype('uint8')
        board[board==1]=255
        
        save_dir='./marker'
        if os.path.exists(save_dir)==False:
            os.mkdir(save_dir)
        name = ''.join([str(h),'x',str(w),'x',str(size),'m','.jpg'])
        path=os.path.join(save_dir,name)
        cv2.imwrite(path,board)
        return board
        
    def camcalib_undistort(self,img,cam_matrix,dist_coeff,crop=True):
        '''
        图像去畸变

        Parameters
        ----------
        img : array, 输入图像
            DESCRIPTION.
        cam_matrix : array, 相机内参矩阵(3x3)
            DESCRIPTION.
        dist_coeff : array, 相机畸变系数
            DESCRIPTION.
        crop : bool,optional, 是否裁剪正畸后导致图像的黑边
            DESCRIPTION. The default is True.

        Returns
        -------
        img : array, 正畸后的图像
            DESCRIPTION.

        '''
        if crop==True:
            h,w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeff, (w,h), 1, (w,h))
            img = cv2.undistort(img, cam_matrix, dist_coeff, None, newcameramtx)
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]
        else:
            img = cv2.undistort(img, cam_matrix, dist_coeff, None, cam_matrix)
        return img
    def pc_depth2xyz(self,depth_map,depth_cam_matrix,flatten=False,depth_scale=1000):
        fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
        cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
        h,w=np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]
        z=depth_map/depth_scale
        x=(w-cx)*z/fx
        y=(h-cy)*z/fy
        xyz=np.dstack((x,y,z)) if flatten==False else np.dstack((x,y,z)).reshape(-1,3)
        return xyz
    
    def pc_xyz2vu(self,xyz,depth_cam_matrix,depth_dist_coeff=np.zeros(5)):
        # 畸变类型带添加
        fx,fy=depth_cam_matrix[0,0],depth_cam_matrix[1,1]
        cx,cy=depth_cam_matrix[0,2],depth_cam_matrix[1,2]
        xyz=np.array(xyz).flatten()
        x,y=xyz[0]/xyz[2],xyz[1]/xyz[2]
        r2=x*x+y*y
        f=1+depth_dist_coeff[0]*r2+depth_dist_coeff[1]*r2*r2+depth_dist_coeff[1]*r2*r2*r2
        x*=f
        y*=f        
        dx=x+2*depth_dist_coeff[2]*x*y+depth_dist_coeff[3]*(r2+2*x*x)
        dy=y+2*depth_dist_coeff[3]*x*y+depth_dist_coeff[2]*(r2+2*y*y)
        x,y=dx,dy
        u,v=x*fx+cx,y*fy+cy
        vu=np.int0([v,u])
        return vu
    
    def pc_get3dLine(self,xyz1,xyz2):
        x1,y1,z1=np.array(xyz1).flatten()
        x2,y2,z2=np.array(xyz2).flatten()
        m,n,p=x2-x1,y2-y1,z2-z1
        x0,y0,z0=x1,y1,z1
        line = x0,y0,z0,m,n,p
        return line
    
    def pc_get3dP(self,xyz1,d,line):
        # 已知直线表达式(line), 直线上一点(xyz1),距离d,求直线上未知点(xyz2)到xyz1距离为d
        x1,y1,z1=np.array(xyz1).flatten()
        x0,y0,z0,m,n,p = line
        x=x0-x1
        y=y0-y1
        z=z0-z1
        a=m*m+n*n+p*p
        b=2*x*m+2*y*n+2*z*p
        c=x*x+y*y+z*z-d*d
        delta = b*b-4*a*c
        if delta>0:
            t1=(-b+np.sqrt(delta))/(2*a)
            t2=(-b-np.sqrt(delta))/(2*a)
            xyz2=np.array([[x0+m*t1,y0+n*t1,z0+p*t1],
                           [x0+m*t2,y0+n*t2,z0+p*t2]])
        elif delta == 0:
            t=-b-np.sqrt(delta)/2*a
            xyz2=np.array([[x0+m*t,y0+n*t,z0+p*t]])
        else:
            xyz2=None
        return xyz2
        
    def pc_getRGBD(self,color_img,depth_map,gray=True,max_depth=3,depth_scale=1000):
        #依赖pc_o3dRGBD
        if type(color_img)==np.ndarray:
            color_img = o3d.geometry.Image(cv2.cvtColor(color_img,cv2.COLOR_BGR2RGB))
        if type(depth_map)==np.ndarray:
            depth_map = o3d.geometry.Image(depth_map)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_map,
            convert_rgb_to_intensity=gray,
            depth_scale=depth_scale, # depth_map/1000
            depth_trunc=max_depth) # 单位m, 深度大于3(scale后)的点将会被剔除
        return rgbd
    
    def pc_getRGBDs(self,color_imgs,depth_maps,max_depth=3,depth_scale=1000,gray=True):
        rgbds= tuple(self.pc_getRGBD(i,j,gray,max_depth,depth_scale) for i,j in zip(color_imgs,depth_maps))
        return rgbds
    # 生成点云
    def pc_getPCFromRGBD(self,rgbd, cam_intr_o3d,remove=True):
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,cam_intr_o3d,
                                                              project_valid_depth_only=remove)
        
    def pc_getPCFromDepth(self,depth_map,cam_intr_o3d,max_depth=3,remove=True,depth_scale=1000):
        if type(depth_map)==np.ndarray:
            depth_map = o3d.geometry.Image(depth_map)
        return o3d.geometry.PointCloud.create_from_depth_image(depth_map,cam_intr_o3d,
                                                               project_valid_depth_only=remove,
                                                               depth_trunc=max_depth,
                                                               depth_scale=depth_scale)
        
    def pc_getPCFromColorDepth(self,color_img,depth_map,cam_intr_o3d,
                               max_depth=3,depth_scale=1000,gray=True,remove=True):
        #依赖pc_getRGBD,pc_pcFromRGBD
        rgbd=self.pc_getRGBD(color_img,depth_map,gray,max_depth,depth_scale)
        pc=self.pc_getPCFromRGBD(rgbd,cam_intr_o3d,remove)
        return pc
    
    def pc_getCamIntr(self,cam_matrix,H,W,save=False,save_dir='./cam_para'):
        fx,fy,cx,cy= cam_matrix[0,0],cam_matrix[1,1],cam_matrix[0,2],cam_matrix[1,2]
        cam_intr_o3d=o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        if save==True:
            name=input('请输入文件名: ')
            if len(name)!=0:
                if name.endswith('.json'):
                    path=''.join([save_dir, '/' , name])
                else:
                    path=''.join([save_dir, '/' , name, '.json'])
                o3d.io.write_pinhole_camera_intrinsic(path,cam_intr_o3d)
        return cam_intr_o3d
    
    def pc_readCamIntr(self,cam_intr_path):
        return o3d.io.read_pinhole_camera_intrinsic(cam_intr_path)
    
    def pc_savePC(self,pc,num,suffix='.pcd',zfill=4,save_dir='./data/pc',name = 'point_cloud'):
        path=''.join([save_dir,'/',name,str(num).zfill(int(zfill)),suffix])
        print(path)
        o3d.io.write_point_cloud(path,pc)
        
    def pc_readPCs(self,read_dir='./data/pc',suffix='.ply',name=None):
        name_list=os.listdir(read_dir)
        if name is not None:
            pc=tuple(o3d.io.read_point_cloud(os.path.join(read_dir,i)) for i in name_list \
                     if i.endswith(suffix) and name in i)
        else:
            pc=tuple(o3d.io.read_point_cloud(os.path.join(read_dir,i)) for i in name_list \
                     if i.endswith(suffix))
        return pc
    
    def pc_show(self,*pc):
        # 功能待添加
        o3d.visualization.draw_geometries(pc)
    def pc_savePoseGraph(self,pose_graph,path):
        if not path.endswith('.json'):
            path=''.join([path,'.json'])
        o3d.io.write_pose_graph(path,pose_graph)
            
    def pc_readPoseGraph(self,path):
        pose_graph=o3d.io.read_pose_graph(path)
        return pose_graph
    
    def pc_readPoseGraphs(self,read_dir,name=None,name_only=True):
        name_list=os.listdir(read_dir)
        if name is not None:            
            pose_graph=tuple(o3d.io.read_pose_graph(os.path.join(read_dir,i)) for i in name_list \
                             if i.endswith('.json') and name in i)
        elif name is None:
            pose_graph=tuple(o3d.io.read_pose_graph(os.path.join(read_dir,i)) for i in name_list \
                             if i.endswith('.json'))

        return pose_graph
    def pc_readImgs(self,read_dir='./',suffix='.jpg',name = None):
        # 读取文件夹下所有图片
        # 该函数与readImgs类似, 但速度较快, 返回o3d的obj, 但颜色通道为rgb, readImgs颜色通道为bgr
        name_list = os.listdir(read_dir)
        if name is not None:
            img=tuple(o3d.io.read_image(os.path.join(read_dir,i)) for i in name_list \
                     if i.endswith(suffix) and name in i)
        else:
            img=tuple(o3d.io.read_image(os.path.join(read_dir,i)) for i in name_list \
                     if i.endswith(suffix))        
        return img
    
    def pc_downSample(self,pc,size=0.01):
        return pc.voxel_down_sample(size)
    
    def pc_segmentPlane(self,pc,distance_threshold=0.01,ransac_n=3,num_iterations=1000,show=False):
        # 分割平面
        # 返回:平面的点云,非平面点云,平面表达式(0=ax+by+cz+d)
        plane_abcd, inliers = pc.segment_plane(distance_threshold,
                                             ransac_n,
                                             num_iterations)
        pc_plane = pc.select_by_index(inliers)
        pc_other = pc.select_by_index(inliers, invert=True)
        if show:
            pc_plane.paint_uniform_color([1.0, 0, 0])
            self.pc_show(pc_plane)
            
        return pc_plane,pc_other,plane_abcd
    
    def pc_getDepthRange(self,pc):
        # 获取点云的z区间
        z_max=pc.get_max_bound()[2]
        z_min=pc.get_min_bound()[2]
        return z_min,z_max
    
    def pc_cluster(self,pc,eps=0.01,min_point=10,show=False):
        # 点云聚类
        labels = np.array(pc.cluster_dbscan(eps=0.01, min_points=10, print_progress=False))
        count = labels.max()+1
        if show==True:
#            o3d.utility.set_verbosity_level(o3d.utility.Debug)
            cmap = plt.get_cmap("tab20")
            colors = cmap(labels / (count if count > 0 else 1))
            colors[labels < 0] = 0
            pc.colors = o3d.utility.Vector3dVector(colors[:, :3])
            self.pc_show(pc)
        return labels,count
    
    def reconstruction_configInit(self):
        config={'preload_files':{'cam_intr_o3d_path':''},
         'make_fragments':{'save_dir':'./reconstruction/make_fragments',
                           'fragment_count':5,
                           'n_keyframes_per_n_frame':5,
                           'tsdf_cubic_size':3.0,
                           'max_depth_diff':0.07
                           }}
        return config
    
    def reconstruction_makeFragment(self,rgbds_gray,rgbds_color,config):
        # 说明如下:
        # 此函数为3d重建的第一步,将一系列的连续的帧分块打包为'碎片(fragment)点云',即将整个场景分为一块一块,
        # 函数将定义每块'碎片'包含多少帧.
        # 打包过程:
        # 通过视觉里程计记录每一帧的相机拍摄姿态(串联起这些姿态即会形成相机的运动轨迹),
        # 距离截断函数将根据相机的运动轨迹融合每块碎片中的所有帧为点云(并将这些帧的姿态从世界坐标系转到到相机坐标系)
        # 视觉里程计+rgbd融合
        #################################初始化##########################################
        make_fragments=config['make_fragments']
        preload_files=config['preload_files']
        
        # 读入相机内参,如路径不存在则使用默认内参
        cam_intr_o3d_path=preload_files['cam_intr_o3d_path']
        cam_intr_o3d=o3d.io.read_pinhole_camera_intrinsic(cam_intr_o3d_path) \
        if os.path.exists(cam_intr_o3d_path)==True else o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        
        
        save_dir=make_fragments['save_dir'] # 保存输出的文件夹
        os.makedirs(save_dir,exist_ok=True) # 如果文件夹不存在则创建文件夹
        # 技术参数    
        fragment_count=make_fragments['fragment_count'] #碎片数量
        n_keyframes_per_n_frame=make_fragments['n_keyframes_per_n_frame'] 
        max_depth_diff=make_fragments['max_depth_diff'] #里程计参数
        tsdf_cubic_size=make_fragments['tsdf_cubic_size']
        
        
        n_frames_per_fragment=int((len(rgbds_gray)-len(rgbds_gray)%(fragment_count))/fragment_count) # 每个碎片包含的帧数
        print('n frames per fragment: ', n_frames_per_fragment)
        odo_init = np.identity(4) # 里程计初始姿态
        # 视觉里程计初始化
        odo_option = o3d.odometry.OdometryOption()
        odo_option.max_depth_diff = max_depth_diff
        pose_graphs=()
        fragments=()
        ###################################开始循环#########################################
        print('-'*10,'start making fragments','-'*10)
        for fragment_id in range(fragment_count):
            # 计算每个碎片的起始帧id 及结束帧id
            print('-'*10,'making fragment_id :',fragment_id,'-'*10,'\n'
                  'integrating frame pairs:')
            start_frame_id=fragment_id * n_frames_per_fragment
            end_frame_id = min(start_frame_id + n_frames_per_fragment, len(rgbds_gray))
            
            # 为每个碎片初始化运动轨迹
            cam_pose = np.identity(4) # 相机运动的初始姿态
            pose_graph = o3d.registration.PoseGraph()
            pose_graph.nodes.append(o3d.registration.PoseGraphNode(cam_pose))        
            
            # 初始化
            volume = o3d.integration.ScalableTSDFVolume(voxel_length=tsdf_cubic_size / 512.0,
            sdf_trunc=0.04,color_type=o3d.integration.TSDFVolumeColorType.RGB8)   
            for s in range(start_frame_id,end_frame_id):
                for t in range(s + 1,end_frame_id):
                    if t == s + 1: #逐帧求姿态
                        print(s,t,'-success')
                        ###################################求解相机姿态###################################
                        [success, new_cam_pose, info] = o3d.odometry.compute_rgbd_odometry(rgbds_gray[s],rgbds_gray[t],
                            cam_intr_o3d,odo_init,o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), odo_option)
                        if success:
                            cam_pose = np.dot(new_cam_pose, cam_pose)# 更新姿态
                            cam_pose_inv=np.linalg.inv(np.dot(new_cam_pose, cam_pose))
                            
                            volume.integrate(rgbds_color[s], cam_intr_o3d,cam_pose)
                            pose_graph.nodes.append(o3d.registration.PoseGraphNode(cam_pose_inv))
                            pose_graph.edges.append(o3d.registration.PoseGraphEdge(s - start_frame_id,
                                                                                   t - start_frame_id,
                                                                                   cam_pose,
                                                                                   info,
                                                                                   uncertain=False))
                        else:
                            print(s,t,'-unsuccess')                                  
                        
                    if s % n_keyframes_per_n_frame == 0 and t % n_keyframes_per_n_frame == 0:
                        [success, new_cam_pose, info] = o3d.odometry.compute_rgbd_odometry(rgbds_gray[s],rgbds_gray[t],
                            cam_intr_o3d,odo_init,o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), odo_option)
                        if success:
                            pose_graph.edges.append(o3d.registration.PoseGraphEdge(s - start_frame_id,
                                                                                   t - start_frame_id,
                                                                                   cam_pose,
                                                                                   info,
                                                                                   uncertain=True))
            pose_graphs+=(pose_graph,)
            mesh = volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            fragment = o3d.geometry.PointCloud()
            fragment.points = mesh.vertices
            fragment.colors = mesh.vertex_colors
            fragments+=(fragment,)
            
            # 运动轨迹存入本地
            pose_graph_save_path=os.path.join(save_dir,'pose_graph'+str(fragment_id).zfill(4)+'.json')
            o3d.io.write_pose_graph(pose_graph_save_path,pose_graph)
        return fragments,pose_graphs
    
if __name__ == '__main__':
    ct=cvtools()
