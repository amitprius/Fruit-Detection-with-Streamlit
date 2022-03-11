import streamlit as st

import random

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import imutils
import cv2



import warnings
warnings.filterwarnings('ignore')


st.set_page_config(layout="wide", )
st.title("Fruits Detection and Counting")

st.sidebar.header("**This App will Localize the Fruits and Counts in the Image**")


#menu = ['Image Based', 'Video Based']
menu = ['Blue Grape Image', 'Orange', 'Apple', 'PineApple']
st.subheader('Fruits Image Selection')
fruit_choice = st.selectbox('Choose the type of the Fruit Image', menu)


#image = st.file_uploader('Upload your Image here',type=['jpg','jpeg','png'])


col1, col2, col3 = st.columns(3)

with col1:

    if fruit_choice == "Blue Grape Image":

        st.subheader("Input Image")

        image1 = Image.open("bg4.jpg")
        st.image(image1)
        image = cv2.imread('bg4.jpg') #reads the image
        dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        cv2.imwrite('dst.jpg',dst)
        
        

        
        #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        rgb_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        cv2.imwrite('ST_RGB_image.jpg',rgb_image)

        new_image = cv2.medianBlur(rgb_image,5)
        cv2.imwrite('median_blur.jpg',new_image)


        hsv_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)

        cv2.imwrite('H.jpg',h)

        ret, th1 = cv2.threshold(h,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite('Binary_image.jpg',th1)

        kernel = np.ones((5,5), dtype = "uint8")/9
        bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
        erosion = cv2.erode(bilateral, kernel, iterations = 6)

        cv2.imwrite('mask_erosion.jpg', erosion)


        

    #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th1, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever

        min_size = 7000

    #your answer image
        img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
                cv2.imwrite('img2.jpg',img2)
            
        img3 = img2.astype(np.uint8) 
        cv2.imwrite('binary_connected_components.jpg',img3)      
        # find contours in the thresholded image

        # find contours in the thresholded image
        cnts = cv2.findContours(img3.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print("[INFO] {} unique contours found".format(len(cnts)))



        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the contour
            ((x, y), _) = cv2.minEnclosingCircle(c)
            cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        # show the output image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

        cv2.imwrite('Result_BLue_Grape.jpg',image)

        with col2:
            st.subheader("Mask Image")
            image2 = Image.open("img2.jpg")
            st.image(image2)

        with col3:
            st.subheader("Output Image")
            image3 = Image.open("Result_BLue_Grape.jpg")
            st.image(image3)


with col1:

    if fruit_choice == "Orange":

        st.subheader("Input Image")

        image1 = Image.open("orange.jpg")
        st.image(image1)
        image = cv2.imread('orange.jpg') #reads the image
        dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        rgb_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        cv2.imwrite('RGB_image.jpg',rgb_image)

        new_image = cv2.medianBlur(rgb_image,5)
        cv2.imwrite('median_blur.jpg',new_image)

        ycbcr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(ycbcr_image)


        #hsv_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
        #h, s, v = cv2.split(hsv_image)

        cv2.imwrite('Cr.jpg',Cr)

        ret, th1 = cv2.threshold(Cr,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite('Binary_image.jpg',th1)

        kernel = np.ones((5,5), dtype = "uint8")/9
        bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
        erosion = cv2.erode(bilateral, kernel, iterations = 6)

        cv2.imwrite('mask_erosion.jpg', erosion)


        cv2.imwrite('dst.jpg',dst)

        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th1, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever

        min_size = 2000

        #your answer image
        img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
                cv2.imwrite('img2.jpg',img2)
                
        img3 = img2.astype(np.uint8) 
        cv2.imwrite('binary_connected_components.jpg',img3)      
        # find contours in the thresholded image

        # find contours in the thresholded image
        cnts = cv2.findContours(img3.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print("[INFO] {} unique contours found".format(len(cnts)))



        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the contour
            ((x, y), _) = cv2.minEnclosingCircle(c)
            cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        # show the output image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

        cv2.imwrite('Result_Orange_Image.jpg',image)

        with col2:
            st.subheader("Mask Image")
            image2 = Image.open("img2.jpg")
            st.image(image2)

        with col3:
            st.subheader("Output Image")
            image3 = Image.open("Result_Orange_Image.jpg")
            st.image(image3)

with col1:

    if fruit_choice == "Apple":

        st.subheader("Input Image")

        image1 = Image.open("a2.jpg")
        st.image(image1)
        image = cv2.imread('a2.jpg') #reads the image
        dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        rgb_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        cv2.imwrite('RGB_image.jpg',rgb_image)

        new_image = cv2.medianBlur(rgb_image,5)
        cv2.imwrite('median_blur.jpg',new_image)

        ycbcr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(ycbcr_image)


        #hsv_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
        #h, s, v = cv2.split(hsv_image)

        cv2.imwrite('Cr.jpg',Cr)

        ret, th1 = cv2.threshold(Cr,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite('Binary_image.jpg',th1)

        kernel = np.ones((5,5), dtype = "uint8")/9
        bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
        erosion = cv2.erode(bilateral, kernel, iterations = 6)

        cv2.imwrite('mask_erosion.jpg', erosion)


        cv2.imwrite('dst.jpg',dst)

        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th1, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever

        min_size = 2000

        #your answer image
        img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
                cv2.imwrite('img2.jpg',img2)
                
        img3 = img2.astype(np.uint8) 
        cv2.imwrite('binary_connected_components.jpg',img3)      
        # find contours in the thresholded image

        # find contours in the thresholded image
        cnts = cv2.findContours(img3.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print("[INFO] {} unique contours found".format(len(cnts)))



        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the contour
            ((x, y), _) = cv2.minEnclosingCircle(c)
            cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
        # show the output image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

        cv2.imwrite('Result_Apple_Image.jpg',image)

        with col2:
            st.subheader("Mask Image")
            image2 = Image.open("img2.jpg")
            st.image(image2)

        with col3:
            st.subheader("Output Image")
            image3 = Image.open("Result_Apple_Image.jpg")
            st.image(image3)

with col1:

    if fruit_choice == "PineApple":

        st.subheader("Input Image")

        image1 = Image.open("pine3.jpg")
        st.image(image1)
        image = cv2.imread('pine3.jpg') #reads the image
        dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        rgb_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        cv2.imwrite('RGB_image.jpg',rgb_image)

        new_image = cv2.medianBlur(rgb_image,5)
        cv2.imwrite('median_blur.jpg',new_image)

        ycbcr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(ycbcr_image)


        #hsv_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
        #h, s, v = cv2.split(hsv_image)

        cv2.imwrite('Cr.jpg',Cr)

        ret, th1 = cv2.threshold(Cr,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite('Binary_image.jpg',th1)

        kernel = np.ones((5,5), dtype = "uint8")/9
        bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
        erosion = cv2.erode(bilateral, kernel, iterations = 6)

        cv2.imwrite('mask_erosion.jpg', erosion)


        cv2.imwrite('dst.jpg',dst)

        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th1, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever

        min_size = 3000

        #your answer image
        img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
                cv2.imwrite('img2.jpg',img2)
                
        img3 = img2.astype(np.uint8) 
        cv2.imwrite('binary_connected_components.jpg',img3)      
        # find contours in the thresholded image

        # find contours in the thresholded image
        cnts = cv2.findContours(img3.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print("[INFO] {} unique contours found".format(len(cnts)))



        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the contour
            ((x, y), _) = cv2.minEnclosingCircle(c)
            cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
        # show the output image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

        cv2.imwrite('Result_PineApple_Image.jpg',image)

        with col2:
            st.subheader("Mask Image")
            image2 = Image.open("img2.jpg")
            st.image(image2)

        with col3:
            st.subheader("Output Image")
            image3 = Image.open("Result_PineApple_Image.jpg")
            st.image(image3)


if fruit_choice == "Blue Grape Image":

    st.sidebar.subheader("Denoising Filters Used")
    st.sidebar.text("Fast Non-Local Means")

    st.sidebar.subheader("Blur Filters Used")
    st.sidebar.text("Median")

    st.sidebar.subheader("ColorSpace Channel Used")
    st.sidebar.text("H Channel in HSV")

    st.sidebar.subheader("Thresholding Used")
    st.sidebar.text("Binary and OTSU")



    st.text("Fruit Count and Pixel Area Information")
    st.markdown(f"**Number of Blue Grape fruit in the image is** {len(cnts):}") 

    area_list=[]

    for i in range(len(cnts)):
        count = cnts[i] 
        area = cv2.contourArea(count) 
        st.markdown(f"**The area of the Blue Grape in** {i+1:} **object is:** {area:} **Pixels**") 
        area_list.append(area)

if fruit_choice == "Orange":

    st.sidebar.subheader("Denoising Filters Used")
    st.sidebar.text("Fast Non-Local Means")

    st.sidebar.subheader("Blur Filters Used")
    st.sidebar.text("Median")

    st.sidebar.subheader("ColorSpace Channel Used")
    st.sidebar.text("Cr Channel in YCrCb")

    st.sidebar.subheader("Thresholding Used")
    st.sidebar.text("Binary and OTSU")

    st.text("Fruit Count and Pixel Area Information")
    st.markdown(f"**Number of Orange fruit in the image is** {len(cnts):}") 

    area_list=[]

    for i in range(len(cnts)):
        count = cnts[i] 
        area = cv2.contourArea(count) 
        st.markdown(f"**The area of the Orange in ** {i+1:} **object is:** {area:} **Pixels**") 
        area_list.append(area)

if fruit_choice == "Apple":

    st.sidebar.subheader("Denoising Filters Used")
    st.sidebar.text("Fast Non-Local Means")

    st.sidebar.subheader("Blur Filters Used")
    st.sidebar.text("Median")

    st.sidebar.subheader("ColorSpace Channel Used")
    st.sidebar.text("Cr Channel in YCrCb")

    st.sidebar.subheader("Thresholding Used")
    st.sidebar.text("Binary and OTSU")

    st.text("Fruit Count and Pixel Area Information")
    st.markdown(f"**Number of Apple fruit in the image is** {len(cnts):}") 

    area_list=[]

    for i in range(len(cnts)):
        count = cnts[i] 
        area = cv2.contourArea(count) 
        st.markdown(f"**The area of the Apple in ** {i+1:} **object is:** {area:} **Pixels**") 
        area_list.append(area)

if fruit_choice == "PineApple":

    st.sidebar.subheader("Denoising Filters Used")
    st.sidebar.text("Fast Non-Local Means")

    st.sidebar.subheader("Blur Filters Used")
    st.sidebar.text("Median")

    st.sidebar.subheader("ColorSpace Channel Used")
    st.sidebar.text("Cr Channel in YCrCb")

    st.sidebar.subheader("Thresholding Used")
    st.sidebar.text("Binary and OTSU")

    st.text("Fruit Count and Pixel Area Information")
    st.markdown(f"**Number of PineApple fruit in the image is** {len(cnts):}") 

    area_list=[]

    for i in range(len(cnts)):
        count = cnts[i] 
        area = cv2.contourArea(count) 
        st.markdown(f"**The area of the PineApple in ** {i+1:} **object is:** {area:} **Pixels**") 
        area_list.append(area)

        