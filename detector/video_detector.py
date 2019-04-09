import numpy as np
import cv2
import time

class VideoDetector:
  def __init__(self, net, inpt, output, minConfidence, threshold, labels, colors):
    self.net = net
    self.input = inpt
    self.output = output
    self.minConfidence = minConfidence
    self.NMSThreshold = threshold
    self.labels = labels
    self.colors = colors
    self.total = None
    self.prop = None
    self.H = None
    self.W = None
    self.writer = None
    self.frame = None

  def detect(self):
    vs = cv2.VideoCapture(self.input)
    self.loop_over_frames(vs)
    return (vs, self.writer)

  def init_video_stream(self):
    try:
      prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
      self.total = int(vs.get(prop))
      print("0")
      print("[INFO] {} total frames in video".format(self.total))

    # an error occurred while trying to determine the total # frames in the video file
    except:
      print("[INFO] could not determine # of frames in video")
      print("[INFO] no approx. completion time can be provided")
      self.total = -1

  def loop_over_frames(self, vs):
    ln = self.get_output_layer_names()
    while True:
      (grabbed, self.frame) = vs.read() # read the next frame from the file
      if not grabbed:
        break # reached end if frame not grabbed
      self.set_blob_input()
      self.set_prediction_values()
      self.update_layer_outputs(ln)
      idxs = self.smooth_response_map()
      if len(idxs) > 0:
        for i in idxs.flatten():
          self.draw_bounding_box(i)
      self.init_video_writer()     

  def draw_bounding_box(self, i):
    color = [int(c) for c in self.colors[self.classIDs[i]]]
    (x, y, w, h) = self.get_box_coordinates(i)
    cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
    text = "{}: {:.4f}".format(self.labels[self.classIDs[i]], self.confidences[i])
    cv2.putText(self.frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  def smooth_response_map(self):
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(
      self.boxes, self.confidences, self.minConfidence, self.NMSThreshold)
    return idxs

  def update_layer_outputs(self, ln):
    layerOutputs = self.get_layer_outputs(ln)
    for output in layerOutputs:
      for detection in output:
          self.update(detection)

  # Update list of bounding box coordinated, confidences and classIDs
  def update(self, detection):
    scores = detection[5:]
    classID = np.argmax(scores)
    confidence = scores[classID]
    if confidence > self.minConfidence:
      box = self.get_box(detection)
      self.boxes.append(box)
      self.confidences.append(float(confidence))
      self.classIDs.append(classID)

######### Getters ###########

  def get_layer_outputs(self, ln):
    self.set_blob_input()
    self.start = time.time()
    layerOutputs = self.net.forward(ln)
    self.end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(self.end - self.start))
    return layerOutputs

  def get_output_layer_names(self):
    ln = self.net.getLayerNames()
    self.init_video_stream()
    ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    return ln
 
  def get_box(self, detection):
    if self.W is None or self.H is None: # if the frame dimensions are empty, grab them
      (self.H, self.W) = self.frame.shape[:2]
    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
    (centerX, centerY, width, height) = box.astype("int")
    x = int(centerX - (width / 2))
    y = int(centerY - (height / 2))
    return [x, y, int(width), int(height)]
  
  def get_box_coordinates(self, i):
    (x, y) = (self.boxes[i][0], self.boxes[i][1])
    (w, h) = (self.boxes[i][2], self.boxes[i][3])
    return (x, y, w, h)

######### Setters ###########

  def set_blob_input(self):
    blob = cv2.dnn.blobFromImage(
      self.frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    self.net.setInput(blob)

  def set_prediction_values(self):
    self.boxes = []
    self.confidences = []
    self.classIDs = []

  def init_video_writer(self):
    if self.writer is None:
      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
      self.writer = cv2.VideoWriter(
        self.output, fourcc, 30, (self.frame.shape[1], self.frame.shape[0]), True)
      if (self.total > 0):
        elap = (self.end - self.start)
        print("[INFO] single frame took {:.4f} seconds".format(elap))
        print("[INFO] estimated total time to finish: {:.4f}".format(elap * self.total))
    
    self.writer.write(self.frame) # write the output frame to disk