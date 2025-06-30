import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from . import Algorithm


class InvisCloak (Algorithm):

    """ init function """
    def __init__(self):
        self.N = 5  # Number of frames to average
        self.frame_buffer = []
        self.last_clicked_img = None


    """ Processes the input image"""
    def process(self, img):

        """ 2.1 Vorverarbeitung """
        """ 2.1.1 Rauschreduktion """
        plotNoise = True   # Schaltet die Rauschvisualisierung ein
        if plotNoise:
            self._plotNoise(img, "Rauschen vor Korrektur")
        img = self._211_Rauschreduktion(img)
        if plotNoise:
            self._plotNoise(img, "Rauschen nach Korrektur")



        """ 2.1.2 HistogrammSpreizung """
        img = self._212_HistogrammSpreizung(img)
        self.last_clicked_img = img

        """ 2.2 Farbanalyse """
        """ 2.2.1 RGB """
        #self._221_RGB(img)
        """ 2.2.2 HSV """
        #self._222_HSV(img)


        """ 2.3 Segmentierung und Bildmdifikation """
        img = self._23_SegmentUndBildmodifizierung(img)

        return img

    """ Reacts on mouse callbacks """
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print("A Mouse click happend! at position", x, y)
        if event == cv2.EVENT_LBUTTONUP:
            print(f"Mouse click at ({x}, {y}) — captured image for RGB histogram.")
            self._221_RGB(self.last_clicked_img)
            print(f"Mouse click at ({x}, {y}) — captured image for HSV histogram.")
            self._222_HSV(self.last_clicked_img)

    def _plotNoise(self, img, name:str):
        height, width = np.array(img.shape[:2])
        centY = (height / 2).astype(int)
        centX = (width / 2).astype(int)

        cutOut = 5
        tmpImg = deepcopy(img)
        tmpImg = tmpImg[centY - cutOut:centY + cutOut, centX - cutOut:centX + cutOut, :]

        outSize = 500
        tmpImg = cv2.resize(tmpImg, (outSize, outSize), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(name, tmpImg)
        cv2.waitKey(1)

    def _211_Rauschreduktion(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.1 (Rauschunterdrückung)
            - Implementierung Mittelwertbildung über N Frames
        """
        img_float = img.astype(np.float32)

        self.frame_buffer.append(img_float)

        if len(self.frame_buffer) > self.N:
            self.frame_buffer.pop(0)

        avg_img = np.mean(self.frame_buffer, axis=0).astype(np.uint8)

        return avg_img

    def _212_HistogrammSpreizung(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.2 (Histogrammspreizung)
            - Transformation HSV
            - Histogrammspreizung berechnen
            - Transformation BGR
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Split HSV channels
        h, s, v = cv2.split(hsv)

        # Calculate min and max of V channel
        min_v = np.min(v)
        max_v = np.max(v)

        # Avoid division by zero
        if max_v - min_v == 0:
            return img

        # Apply histogram stretching to V channel
        v_stretched = ((v - min_v) / (max_v - min_v) * 255).astype(np.uint8)

        # Merge channels back and convert to BGR
        hsv_stretched = cv2.merge([h, s, v_stretched])
        img_stretched = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)

        return img_stretched


    def _221_RGB(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.1 (RGB)
            - Histogrammberechnung und Analyse
        """
        #Save image to directory
        cv2.imwrite("./data/Saved_Image.png", img)

        channels = cv2.split(img)
        col = ['b','g','r']

        for i in range(len(channels)):
            hist = cv2.calcHist(channels[i], [i], None, [256], [0, 256])
            plt.clf()
            plt.plot(hist, color=col[i])
            plt.xlim([0, 256])
            plt.savefig("./data/Hist_"+col[i].upper()+".png")
            plt.close()



    def _222_HSV(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.2 (HSV)
            - Histogrammberechnung und Analyse im HSV-Raum
        """
        #Convert to HSV and split channels
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        #plot hue
        hist = cv2.calcHist(h, [0], None, [180], [0, 179])
        plt.clf()
        plt.plot(hist, color='black')
        plt.xlim([0, 256])
        plt.savefig("./data/Hist_H.png")
        plt.close()

        #plot saturation
        hist = cv2.calcHist(s, [0], None, [256], [0, 255])
        plt.clf()
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.savefig("./data/Hist_S.png")
        plt.close()

        # plot value
        hist = cv2.calcHist(v, [0], None, [256], [0, 255])
        plt.clf()
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.savefig("./data/Hist_V.png")
        plt.close()






    def _23_SegmentUndBildmodifizierung (self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (StatischesSchwellwertverfahren)
            - Binärmaske erstellen
        """


        """
            Hier steht Ihr Code zu Aufgabe 2.3.2 (Binärmaske)
            - Binärmaske optimieren mit Opening/Closing
            - Wahl größte zusammenhängende Region
        """


        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (Bildmodifizerung)
            - Hintergrund mit Mausklick definieren
            - Ersetzen des Hintergrundes
        """


        return img