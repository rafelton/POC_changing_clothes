#Rafael Elton Santos Oliveira https://www.linkedin.com/in/rafelton/
#Computer Vision classes at Beet-ai https://www.linkedin.com/company/beet-ai/ 

import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

_font = cv2.FONT_HERSHEY_SIMPLEX 
_fgbg = cv2.createBackgroundSubtractorKNN(history=1, detectShadows=False) 
_path = os.path.dirname(os.path.realpath(__file__))
_writeOut = True
frameSize = None

def WriteContours(frame, black, mask_blur):
    contourValidMinSize = 120
    middleX = 0
    objectCaption = 0
    rectangleThickness = 2

    contours, hierarchy = cv2.findContours(mask_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    if len(contours) > 0:
        for (i,contour) in enumerate (contours):
            cv2.drawContours(black, contour, 0, (255,255,255), 5)
            (x,y,w,h) = cv2.boundingRect(contour)
            contour_valid = (w >= contourValidMinSize) and (h >= contourValidMinSize)

            if contour_valid:
                centerX = x + w/2

                if(middleX == 0 or centerX < middleX ):
                    middleX = centerX

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),rectangleThickness, cv2.LINE_AA)
                objectCaption = frame[y+rectangleThickness * 2:y+h-rectangleThickness * 2,
                    x+rectangleThickness * 2:x+w-rectangleThickness * 2]
                
    return frame, middleX, objectCaption

def ProcessFrame():
    ret, frame = cap.read()
    if not ret:
        exit()
    blur = (20,20)
    black = np.zeros_like(frame)
    ROI = black.copy()
    ROI[90:,300:] = frame[90:,300:]
    gray = cv2.cvtColor(ROI, cv2.COLOR_RGB2GRAY)
    #gray = cv2.equalizeHist(gray)
    fgmask = _fgbg.apply(gray)
    mask_blur = cv2.blur(fgmask, blur)
    return frame, black, gray, mask_blur

def get_histogram(src):
    bgr_planes = cv2.split(src)
    histSize = 1024
    histRange = (0, histSize)  
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange)
    b = b_hist / sum(b_hist)
    r = r_hist / sum(r_hist)
    g = g_hist / sum(g_hist)
    histogram = np.array([b,g,r]).reshape(-1,1)
    return histogram, b,g,r

def GetDirection(previousMiddleX, middleX):
    if previousMiddleX == 0 or middleX == 0:
        return ''
    elif middleX < previousMiddleX:
        return 'Left'
    elif middleX > previousMiddleX:
        return 'Right'

def ShowMathPlotAsCV2(frame, fig):
    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, frameDetailsSize)
    return img

if __name__ == "__main__":

    xGapDirectionTolerance = 40
    previousCenterX = 0
    lastHistogram = None
    histogramaTextResponse = '--'
    lastObjectFrame = None
    lastDirection = ''
    newDirection = ''
    directionChanged = False

    plt.style.use('dark_background')
    cap = cv2.VideoCapture(_path + '/changing_clothes.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame, black, gray, mask_blur = ProcessFrame()
    frameSize = (frame.shape[1], frame.shape[0])
    frameDetailsSize = (int(frame.shape[1]/3), frame.shape[0])
    frameDetails = cv2.resize(black, frameDetailsSize)
    out = cv2.VideoWriter(_path + '/out.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), frameSize)
    fig, ax = plt.subplots(2,2)
    
    while True:

        frame, black, gray, mask_blur = ProcessFrame()
        frame, centerX, objectFrame = WriteContours(frame, black, mask_blur)

        #detecting a direction based on the movement of the center point X based on the previous position
        if centerX > 0 and (previousCenterX > centerX + xGapDirectionTolerance or previousCenterX < centerX - xGapDirectionTolerance):

            #ruling out absurd movements like moving from X = 30 to x = 600
            if (centerX - previousCenterX < xGapDirectionTolerance * 3):

                newDirection = GetDirection(previousCenterX, centerX)

                #print(str(previousCenterX) +' '+ str(centerX) + ' '+ str(newDirection))

                if newDirection != '':

                    #getting histogram each frame, maybe it can be improved
                    h2,b2,g2,r2 = get_histogram(objectFrame)

                    #plot detais when the direction changes from Left to the Right
                    if lastDirection == 'Left' and newDirection == 'Right':
                    
                        ax[0][0].set_title("Before/After")
                        ax[0][1].set_title("Histogram")
                        ax[0][0].set_axis_off()
                        ax[0][0].imshow(cv2.cvtColor(lastObjectFrame, cv2.COLOR_BGR2RGB))
                        ax[0][1].plot(lastHistogram)
                        ax[1][0].set_axis_off()
                        ax[1][0].imshow(cv2.cvtColor(objectFrame, cv2.COLOR_BGR2RGB))
                        ax[1][1].plot(h2)
                        frameDetails = ShowMathPlotAsCV2(frame, fig)
                        resp = cv2.compareHist(h2, lastHistogram, 0)
                        histogramaTextResponse = str(round(resp, 2))
                        if resp < 0.7:
                           histogramaTextResponse = histogramaTextResponse + " Change of Clothes Detected!!"

                    #rebuild plot
                    elif lastDirection == 'Right' and newDirection == 'Left':
                        fig, ax = plt.subplots(2,2)
                        histogramaTextResponse = '--'

                    
                    lastHistogram = h2
                    lastObjectFrame = objectFrame
                    lastDirection = newDirection

            previousCenterX = centerX
        

        cv2.putText(frame, 'Histogram: ' + histogramaTextResponse, (500,160), _font,.8, [0,255,0], 2, cv2.LINE_AA)
        cv2.putText(frame, 'Flow: ' + lastDirection, (500,120), _font,.8, [0,255,0], 2, cv2.LINE_AA)
       
        img_pil = Image.fromarray(frame)
        img_pil_details = Image.fromarray(frameDetails)
        draw = ImageDraw.Draw(img_pil)  
        img_pil.paste(img_pil_details)
        final = np.array(img_pil)
        cv2.imshow("Camera", final)

        if _writeOut:
            out.write(final)

        ch = cv2.waitKey(1)
        if ch == ord('q'):
            out.release()
            cv2.destroyAllWindows()
            exit()

    out.release()
    cv2.destroyAllWindows()