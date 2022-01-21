import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectConf=0.5, minTrackConf=0.5):
        self.staticMode = staticMode
        self.maxFaces =maxFaces
        self.minDetectConf = minDetectConf
        self.minTrackConf = minTrackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, 
                                                self.minDetectConf, self.minTrackConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, 
                                            connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                                            landmark_drawing_spec=self.drawSpec, 
                                            connection_drawing_spec=self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,255,0),1)

                    # print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture("FaceVideos/1.mp4")
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces)!=0:
            print(faces[0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        img = cv2.resize(img, (0,0), fx=0.7, fy=0.7)
        cv2.imshow("Face Image", img)
        cv2.waitKey(30)

if __name__ == "__main__":
    main()