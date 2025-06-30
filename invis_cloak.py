import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from . import Algorithm


class InvisCloak (Algorithm):

    """ init function """
    def __init__(self):
        self.N = 5  # Number of frames to average
        self.sum_images = 0
        self.frame_buffer = []
        self.last_clicked_img = None
        self.current_frame = None
        self.background = None
        


    """ Processes the input image"""
    def process(self, img):
        self.current_frame = img.copy()
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
        if event == cv2.EVENT_LBUTTONUP:
            if self.current_frame is not None:
                self.background = self.current_frame.copy()
                print("Hintergrund gespeichert.")

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
        cv2.imwrite("./data/Saved_Image"+str(self.sum_images)+".png", img)

        channels = cv2.split(img)
        col = ['b','g','r']

        for i in range(len(channels)):
            hist = cv2.calcHist(channels[i], [i], None, [256], [0, 256])
            plt.clf()
            plt.plot(hist, color=col[i])
            plt.xlim([0, 256])
            plt.savefig("./data/Hist_"+col[i].upper()+str(self.sum_images)+".png")
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
        plt.savefig("./data/Hist_H"+str(self.sum_images)+".png")
        plt.close()

        #plot saturation
        hist = cv2.calcHist(s, [0], None, [256], [0, 255])
        plt.clf()
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.savefig("./data/Hist_S"+str(self.sum_images)+".png")
        plt.close()

        # plot value
        hist = cv2.calcHist(v, [0], None, [256], [0, 255])
        plt.clf()
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.savefig("./data/Hist_V"+str(self.sum_images)+".png")
        plt.close()
        self.sum_images += 1





    def _23_SegmentUndBildmodifizierung (self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (StatischesSchwellwertverfahren)
            - Binärmaske erstellen
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Beispielwerte – anpassen an deine Umhangfarbe!
        # Nehmen wir an: der Umhang ist blau
        channel1 = 0  # H
        lower_bound1, upper_bound1 = 170, 180
        is_condition_1_true = (lower_bound1 < hsv[:, :, channel1]) * (hsv[:, :, channel1] < upper_bound1)

        channel2 = 1  # S
        lower_bound2, upper_bound2 = 70, 130
        is_condition_2_true = (lower_bound2 < hsv[:, :, channel2]) * (hsv[:, :, channel2] < upper_bound2)

        binary_mask = is_condition_1_true * is_condition_2_true
        binary_mask = binary_mask.astype(np.uint8) * 255


        """
            Hier steht Ihr Code zu Aufgabe 2.3.2 (Binärmaske)
            - Binärmaske optimieren mit Opening/Closing
            - Wahl größte zusammenhängende Region
        """
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.erode(binary_mask, kernel)
        binary_mask = cv2.dilate(binary_mask, kernel)
        binary_mask = cv2.dilate(binary_mask, kernel)
        binary_mask = cv2.erode(binary_mask, kernel)

        # Größte zusammenhängende Region finden
        (cnts, _) = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            binary_mask = np.zeros_like(binary_mask)
            binary_mask = cv2.drawContours(binary_mask, [c], -1, color=255, thickness=-1)

        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (Bildmodifizerung)
            - Hintergrund mit Mausklick definieren
            - Ersetzen des Hintergrundes
        """
        if self.background is not None:
            inv_mask = cv2.bitwise_not(binary_mask)

            # Maske auf beide Bilder anwenden
            fg = cv2.bitwise_and(img, img, mask=inv_mask)
            bg = cv2.bitwise_and(self.background, self.background, mask=binary_mask)

            # Überlagern
            img = cv2.add(fg, bg)

        return img
