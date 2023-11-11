import numpy as np
import cv2

def warp(img, UpSampFactor):
    rad = 134.3088158959995
    rad = rad*1.414

    width_to_height = 3

    cent_x = 144.32486476983112
    cent_y = 175.82875006452784
    
    center = [cent_x, cent_y]
    
    new_siz = np.ceil((2*rad+1)*UpSampFactor)

    X, Y = np.meshgrid(np.arange(0, new_siz), np.arange(0, new_siz))

    X = 2*X/new_siz-1 #goes from -1 to +1
    Y = 2*Y/new_siz-1 #goes from -1 to +1

    dis = X**2 + Y**2    
    
    X[dis > 1] = 0
    Y[dis > 1] = 0
    
    mask0 = dis > 1

    X = X*width_to_height
    Y = Y*width_to_height

    rho0 = np.sqrt(X**2 + Y**2)
    phi0 = np.angle(X+1j*Y)

    s0 = 4*rad*rho0*(-1 + (1+3*(1+rho0**2))**(0.5) )/(8*(1+rho0**2))

    X_res = cent_x + s0*np.cos(phi0)
    Y_res = cent_y + s0*np.sin(phi0)

    
    im_warp = np.zeros((X_res.shape[0], X_res.shape[1], img.shape[2]))

    for ch in range(img.shape[2]):
        im_warp[:,:,ch] = (1-mask0)*cv2.remap(img[:,:,ch], np.float32(X_res), np.float32(Y_res), cv2.INTER_LINEAR)
      
    return im_warp


def unwarp(img, UpSampFactor, rad, cent_x, cent_y, width_to_height=3):
    rad = rad*1.414

    new_siz = np.ceil((2*rad+1)*UpSampFactor)
    
    X0, Y0 = np.meshgrid(np.arange(0, 288), np.arange(0, 352))
    X0 = (X0 - cent_x)
    Y0 = (Y0 - cent_y)

    s_new = np.sqrt(X0**2 + Y0**2)

    phi_new = np.angle(X0 + 1j*Y0)
    mask = (s_new > 0.707*rad)

    s_new[s_new > 0.707*rad] = 0


    rho_new = 2*s_new/(2*np.sqrt(rad**2-s_new**2)-rad)

    X_res0 = rho_new*np.cos(phi_new)
    Y_res0 = rho_new*np.sin(phi_new)

    X_res1 = (1+X_res0/width_to_height)*new_siz/2;
    Y_res1 = (1+Y_res0/width_to_height)*new_siz/2;

    im_UN_wrap = np.zeros((X_res0.shape[0], X_res0.shape[1], img.shape[2]))

    for ch in range(img.shape[2]):
        im_UN_wrap[:,:,ch] = (1-mask)*cv2.remap(img[:,:,ch], np.float32(X_res1), np.float32(Y_res1), cv2.INTER_LINEAR)
    
    
    return im_UN_wrap
