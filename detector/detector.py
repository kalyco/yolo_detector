import numpy as np
import cv2
import time

class Detector:
  def __init__(self, net, image, minConfidence, threshold, labels, colors):
    self.net = net
    self.image = image
    self.minConfidence = minConfidence
    self.NMSThreshold = threshold
    self.boxes = []
    self.confidences = []
    self.classIDs = []
    self.labels = labels
    self.colors = colors

  def detect(self):
    self.update_layer_outputs()
    idxs = self.smooth_response_map()
    if len(idxs) > 0:
      for i in idxs.flatten():
        self.draw_bounding_box(i)
    return self.image

  def draw_bounding_box(self, i):
    color = [int(c) for c in self.colors[self.classIDs[i]]]
    (x, y, w, h) = self.get_box_coordinates(i)
    cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
    text = "{}: {:.4f}".format(self.labels[self.classIDs[i]], self.confidences[i])
    cv2.putText(self.image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

  def smooth_response_map(self):
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(
      self.boxes, self.confidences, self.minConfidence, self.NMSThreshold)
    return idxs

  def update_layer_outputs(self):
    layerOutputs = self.get_layer_outputs()
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

  def get_layer_outputs(self):
    ln = self.get_output_layer_names()
    self.set_blob_input()
    start = time.time()
    layerOutputs = self.net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    return layerOutputs

  def get_output_layer_names(self):
    ln = self.net.getLayerNames()
    ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    return ln

  def get_box(self, detection):
    (H, W) = self.image.shape[:2]
    box = detection[0:4] * np.array([W, H, W, H])
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
      self.image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    self.net.setInput(blob)

