### Yolo Detector Class 

A class-based refactor based off [pyimagesearch](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)


## Requirements

* Python 3.5  
* [OpenCV](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)
* A Yolo weight file (too big for github)

## Running Detectors

### Image Detector

ex: `python yolo_image.py --image images/baggage_claim.jpg --yolo yolo-coco`  

<img src="https://i.imgur.com/JBIAP8O.png" width="550">

| Command      | Shortcut | Description                                    | Required | Default |
|--------------|----------|------------------------------------------------|----------|---------|
| --image      | -image   | path to image                                  | True     |         |
| --yolo       | -y       | base path to YOLO directory                    | True     |         |
| --confidence | -c       | minimum probability to filter weak detections  |          | 0.5     |
| --threshold  | -t       | threshold when applying non-maxima suppression |          | 0.3     |
 

### Video Detector

| Command      | Shortcut | Description                                    | Required | Default |
|--------------|----------|------------------------------------------------|----------|---------|
| --input      | -i       | path to input video"                           | True     |         |
| --output     | -o       | path to output video"                          | True     |         |
| --yolo       | -y       | base path to YOLO directory                    | True     |         |
| --confidence | -c       | minimum probability to filter weak detections  |          | 0.5     |
| --threshold  | -t       | threshold when applying non-maxima suppression |          | 0.3     |

ex: `python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco`    

![alt text](https://media.giphy.com/media/TfFjAGTUfAnKZBCIgL/giphy.gif "video detection")
