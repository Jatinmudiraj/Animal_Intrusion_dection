import sys
import time
from playsound import playsound
from twilio.rest import Client
import threading

from datetime import datetime, timedelta



import threading
import numpy as np
from numpy import linalg as LA, true_divide
from flask import Flask, render_template, request, redirect, url_for, jsonify
import subprocess
import os
app = Flask(__name__)

from flask import Flask, render_template, Response

# from object_tracking import objectTracker, object  # Assuming you have an object tracking module
# from video_processing import process_video 
import cv2
import argparse
from scipy.spatial import distance
from munkres import Munkres               # Hungarian algorithm for ID assignment
# from ovms_wrapper.ovms_wrapper import OpenVINO_Model_Server
# from tracker import TrackedObject  # Import the correct module
from line_boundary_check import *

class TrackedObject: 
    def __init__(self, obj_id, pos): 
        self.id = obj_id 
        self.pos = pos 
        self.trajectory = []  # optional

    def update_position(self, new_pos):
        self.pos = new_pos
        self.trajectory.append(new_pos)

SMS_INTERVAL = timedelta(minutes=10)

last_intrusion_time = {'line_cross': None, 'area_intrusion': None}
last_sms_time = {'line_cross': None, 'area_intrusion': None}

# Twilio credentials
account_sid = 'ACa3c6162650f3e601d1da23908c88eaed'
auth_token = 'e32fcc3153dee585e33ae73e1b8fe216'
twilio_number = '+12315887908'
recipient_number = '+919544049564'

client = Client(account_sid, auth_token)


def send_sms_notification():
    message = client.messages.create(
        from_=twilio_number,
        body='There is an intrusion in your personal area.',
        to=recipient_number
    )
    print(f"Message sent with SID: {message.sid}")


prev_counts = {}  # Global dict to store previous counts

def play_sound():
    try:
        playsound("one.mp3")
    except Exception as e:
        print(f"[ERROR] Failed to play sound: {e}")

def play_alert_sound():
    threading.Thread(target=playsound, args=('one.mp3',)).start()


class boundaryLine:
    def __init__(self, line=(0,0,0,0)):
        self.p0 = (line[0], line[1])
        self.p1 = (line[2], line[3])
        self.color = (0,255,255)
        self.lineThinkness = 4
        self.textColor = (0,255,255)
        self.textSize = 4
        self.textThinkness = 2
        self.count1 = 0
        self.count2 = 0

# Draw single boundary line
# def drawBoundaryLine(img, line):
#     (x1, y1), (x2, y2) = line
#     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

def drawBoundaryLine(img, line):
    x1, y1 = line.p0
    x2, y2 = line.p1
    cv2.line(img, (x1, y1), (x2, y2), line.color, line.lineThinkness)
    cv2.putText(img, str(line.count1), (x1, y1), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.putText(img, str(line.count2), (x2, y2), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.drawMarker(img, (x1, y1),line.color, cv2.MARKER_TRIANGLE_UP, 16, 4)
    cv2.drawMarker(img, (x2, y2),line.color, cv2.MARKER_TILTED_CROSS, 16, 4)

# Draw multiple boundary lines
def drawBoundaryLines(img, boundaryLines):
    for line in boundaryLines:
        drawBoundaryLine(img, line)

# in: boundary_line = boundaryLine class object
#     trajectory   = (x1, y1, x2, y2)
from datetime import datetime, timedelta

# Globals to track last intrusion and SMS sent time
SMS_INTERVAL = timedelta(minutes=10)
last_intrusion_time = None
last_sms_time = None

def checkLineCross(boundary_line, trajectory):
    global last_intrusion_time, last_sms_time

    traj_p0  = (trajectory[0], trajectory[1])    # Trajectory of an object
    traj_p1  = (trajectory[2], trajectory[3])
    bLine_p0 = (boundary_line.p0[0], boundary_line.p0[1]) # Boundary line
    bLine_p1 = (boundary_line.p1[0], boundary_line.p1[1])

    intersect = checkIntersect(traj_p0, traj_p1, bLine_p0, bLine_p1)

    if intersect:
        angle = calcVectorAngle(traj_p0, traj_p1, bLine_p0, bLine_p1)
        prev_c1 = boundary_line.count1
        prev_c2 = boundary_line.count2

        if angle < 180:
            boundary_line.count1 += 1
        else:
            boundary_line.count2 += 1

        # Play sound and send SMS only if count1 increased
        if boundary_line.count1 > prev_c1:
            threading.Thread(target=play_sound, daemon=True).start()
            now = datetime.now()
            last_intrusion_time = now

            # Check if SMS can be sent
            if (last_sms_time is None or 
                (now - last_sms_time >= SMS_INTERVAL and
                 now - last_intrusion_time >= SMS_INTERVAL)):

                
                send_sms_notification() #uncomment this line to send SMS
                last_sms_time = now

    # Uncomment if you need to find the intersection point
    # cx, cy = calcIntersectPoint(traj_p0, traj_p1, bLine_p0, bLine_p1)

        # Play sound only if count1 increased (i.e., left line crossed)
        # if boundary_line.count1 > prev_c1:
        #     threading.Thread(target=play_sound, daemon=True).start()
        #     send_sms_notification()
        #cx, cy = calcIntersectPoint(traj_p0, traj_p1, bLine_p0, bLine_p1) # Calculate the intersect coordination

# Multiple lines cross check
def checkLineCrosses(boundaryLines, objects):
    for obj in objects:
        traj = obj.trajectory
        if len(traj)>1:
            p0 = traj[-2]
            p1 = traj[-1]
            for line in boundaryLines:
                checkLineCross(line, [p0[0],p0[1], p1[0],p1[1]])


#------------------------------------
# Area intrusion detection
class area:
    def __init__(self, contour):
        self.contour  = np.array(contour, dtype=np.int32)
        self.count    = 0

warning_obj = None


# Area intrusion check
def checkAreaIntrusion(areas, objects):
    for area in areas:
        area.count = 0
        for obj in objects:
            p0 = (obj.pos[0]+obj.pos[2])//2
            p1 = (obj.pos[1]+obj.pos[3])//2
            #if cv2.pointPolygonTest(area.contour, (p0, p1), False)>=0:
            if pointPolygonTest(area.contour, (p0, p1)):
                area.count += 1

# Draw areas (polygons)
def drawAreas(img, areas):
    for area in areas:
        if area.count>0:
            color=(0,0,255)
        else:
            color=(255,0,0)
        cv2.polylines(img, [area.contour], True, color,4)
        cv2.putText(img, str(area.count), (area.contour[0][0], area.contour[0][1]), cv2.FONT_HERSHEY_PLAIN, 4, color, 2)


#------------------------------------
# Object tracking

class object:
    def __init__(self, pos, feature, id=-1):
        self.feature = feature
        self.id = id
        self.trajectory = []
        self.time = time.monotonic()
        self.pos = pos



class objectTracker:
    def __init__(self):
        self.objectid = 0
        self.timeout  = 3   # sec
        self.clearDB()
        self.similarityThreshold = 0.4
        pass

    def clearDB(self):
        self.objectDB = []

    def evictTimeoutObjectFromDB(self):
        # discard time out objects
        now = time.monotonic()
        for object in self.objectDB:
            if object.time + self.timeout < now:
                self.objectDB.remove(object)     # discard feature vector from DB
                print("Discarded  : id {}".format(object.id))

    # objects = list of object class
    def trackObjects(self, objects):
        # if no object found, skip the rest of processing
        if len(objects) == 0:
            return

        # If any object is registred in the db, assign registerd ID to the most similar object in the current image
        if len(self.objectDB)>0:
            # Create a matix of cosine distance
            cos_sim_matrix=[ [ distance.cosine(objects[j].feature, self.objectDB[i].feature) 
                            for j in range(len(objects))] for i in range(len(self.objectDB)) ]
            # solve feature matching problem by Hungarian assignment algorithm
            hangarian = Munkres()
            combination = hangarian.compute(cos_sim_matrix)

            # assign ID to the object pairs based on assignment matrix
            for dbIdx, objIdx in combination:
                if distance.cosine(objects[objIdx].feature, self.objectDB[dbIdx].feature)<self.similarityThreshold:
                    objects[objIdx].id = self.objectDB[dbIdx].id                               # assign an ID
                    self.objectDB[dbIdx].feature = objects[objIdx].feature                     # update the feature vector in DB with the latest vector (to make tracking easier)
                    self.objectDB[dbIdx].time    = time.monotonic()                            # update last found time
                    xmin, ymin, xmax, ymax = objects[objIdx].pos
                    self.objectDB[dbIdx].trajectory.append([(xmin+xmax)//2, (ymin+ymax)//2])   # record position history as trajectory
                    objects[objIdx].trajectory = self.objectDB[dbIdx].trajectory

        # Register the new objects which has no ID yet
        for obj in objects:
            if obj.id==-1:           # no similar objects is registred in feature_db
                obj.id = self.objectid
                self.objectDB.append(obj)  # register a new feature to the db
                self.objectDB[-1].time = time.monotonic()
                xmin, ymin, xmax, ymax = obj.pos
                self.objectDB[-1].trajectory = [[(xmin+xmax)//2, (ymin+ymax)//2]]  # position history for trajectory line
                obj.trajectory = self.objectDB[-1].trajectory
                self.objectid+=1

    def drawTrajectory(self, img, objects):
        for obj in objects:
            if len(obj.trajectory)>1:
                cv2.polylines(img, np.array([obj.trajectory], np.int32), False, (0,0,0), 4)



#------------------------------------




# boundary lines
boundaryLines = [
    boundaryLine([ 300,  40,  20, 400 ]),
    boundaryLine([ 440,  40, 700, 400 ])
]  

# Areas
areas = [
    area([ [200,200], [500,180], [600,400], [300,300], [100,360] ])
]

_N, _C, _H, _W = 0, 1, 2, 3

clicks= []




@app.route('/')
def home():
    return render_template('index.html')  # Your HTML form goes here


@app.route('/click', methods=['POST'])
def handle_click():
    global clicks
    data = request.get_json()
    x, y = data['x'], data['y']
    clicks.append((x, y))

    response = {'count': len(clicks)}

    if len(clicks) == 4:
        response['boundaryLines'] = [
            {'x1': clicks[0][0], 'y1': clicks[0][1], 'x2': clicks[1][0], 'y2': clicks[1][1]},
            {'x1': clicks[2][0], 'y1': clicks[2][1], 'x2': clicks[3][0], 'y2': clicks[3][1]},
        ]
    elif len(clicks) > 4:
        response['boundaryLines'] = [
            {'x1': clicks[0][0], 'y1': clicks[0][1], 'x2': clicks[1][0], 'y2': clicks[1][1]},
            {'x1': clicks[2][0], 'y1': clicks[2][1], 'x2': clicks[3][0], 'y2': clicks[3][1]},
        ]
        response['areas'] = clicks[4:]

    return jsonify(response)

@app.route('/reset', methods=['POST'])
def reset_clicks():
    print("Resetting clicks")
    global clicks
    clicks = []
    return '', 204


@app.route('/submit-coordinates', methods=['POST'])
def submit_coordinates():
    global boundaryLines, areas, _N, _C, _H, _W  # Declare as global to modify them

    try:
        print("Received form data:")
        print(request.form)
        # ✅ Clean input: Remove brackets and replace commas with spaces
        line1 = list(map(int, request.form.get('line1', '0 0 0 0').replace('[', '').replace(']', '').replace(',', ' ').split()))
        line2 = list(map(int, request.form.get('line2', '0 0 0 0').replace('[', '').replace(']', '').replace(',', ' ').split()))

        # ✅ Clean area coordinates: Remove brackets and format correctly
        area_coords = list(map(int, request.form.get('areas', '').replace('[', '').replace(']', '').replace(',', ' ').split()))
        area_points = [area_coords[i:i+2] for i in range(0, len(area_coords), 2)]

        # ✅ Parse tensor dimensions safely
        _N = int(request.form.get('_N', 0))
        _C = int(request.form.get('_C', 0))
        _H = int(request.form.get('_H', 0))
        _W = int(request.form.get('_W', 0))

        # ✅ Update global variables
        boundaryLines = [
            boundaryLine(line1),
            boundaryLine(line2)
        ]
        areas = [
            area(area_points)

        ]

        return redirect(url_for('video_feed_page'))  # Redirect to video stream page

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})




def update_trajectory(self):
    cx = int((self.pos[0] + self.pos[2]) / 2)
    cy = int((self.pos[1] + self.pos[3]) / 2)
    self.trajectory.append((cx, cy))
    # Keep last N points if you want to limit memory
    if len(self.trajectory) > 30:
        self.trajectory.pop(0)



@app.route('/video-feed')
def video_feed_page():
    return render_template('video.html')  # New video streaming page

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

object_histories = {}

from collections import defaultdict



def get_color(id):
    np.random.seed(int(id))
    return tuple(int(c) for c in np.random.randint(0, 255, size=3))


class object:
    def __init__(self, pos, feature=None, id=-1):
        self.pos = pos
        self.id = id
        self.feature = feature
        self.trajectory = [self._center(pos)]

    def _center(self, pos):
        xmin, ymin, xmax, ymax = pos
        return ((xmin + xmax) // 2, (ymin + ymax) // 2)

    def update(self, new_pos):
        self.pos = new_pos
        self.trajectory.append(self._center(new_pos))


animal_class_ids = [16, 17, 18, 19, 20, 21, 22, 23]  # birds, cats, dogs, horses, sheep, cows, elephants, bears (COCO)

def generate_frames():
    global boundaryLines, areas

    # Initialize YOLOv8 and DeepSORT
    yolo_model = YOLO("yolov8n.pt")  # Replace with 'yolov8s.pt' or your custom-trained model if needed
    tracker = DeepSort(max_age=30)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    # Open video source
    # cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    tracked_objects = {}

    try:
        while True:
            ret, image = cap.read()
            if not ret or image is None:
                print("[WARN] Failed to read frame. Restarting video...")
                cap.release()
                cap = cv2.VideoCapture(video_path)
                continue

            outimg = image.copy()
            results = yolo_model(image)[0]
            detections = []

            # Filter only animal detections
            for det in results.boxes.data.tolist():
                xmin, ymin, xmax, ymax = map(int, det[:4])
                conf, cls_id = det[4], int(det[5])

                if conf > 0.5 and cls_id in animal_class_ids:
                    width = xmax - xmin
                    height = ymax - ymin
                    detections.append(([xmin, ymin, width, height], conf, 'animal'))

            # Update DeepSORT with animal detections
            tracks = tracker.update_tracks(detections, frame=image)
            objects = []

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = track.to_ltrb()
                bbox = [int(l), int(t), int(r), int(b)]

                if track_id not in tracked_objects:
                    tracked_objects[track_id] = object(bbox, id=track_id)
                else:
                    tracked_objects[track_id].update(bbox)

                objects.append(tracked_objects[track_id])

            # Line crossing and area intrusion checks
            checkLineCrosses(boundaryLines, objects)
            drawBoundaryLines(outimg, boundaryLines)

            checkAreaIntrusion(areas, objects)
            drawAreas(outimg, areas)

            # Draw bounding boxes, IDs, and trajectories
            for obj in objects:
                id = obj.id
                color = get_color(id)
                xmin, ymin, xmax, ymax = obj.pos
                cv2.rectangle(outimg, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(outimg, f'ID={id}', (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 1)

                # Optional: draw trajectory
                for i in range(1, len(obj.trajectory)):
                    cv2.line(outimg, obj.trajectory[i - 1], obj.trajectory[i], color, 1)

            # Encode to MJPEG for Flask
            _, buffer = cv2.imencode('.jpg', outimg)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()



# Define animal class IDs as per COCO or your custom model


def upload_frames(video_path):
    global boundaryLines, areas

    # Initialize YOLOv8 and DeepSORT
    yolo_model = YOLO("yolov8n.pt")  # Replace with 'yolov8s.pt' or your custom-trained model if needed
    tracker = DeepSort(max_age=30)

    # Open video source
    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    tracked_objects = {}

    try:
        while True:
            ret, image = cap.read()
            if not ret or image is None:
                print("[WARN] Failed to read frame. Restarting video...")
                cap.release()
                cap = cv2.VideoCapture(video_path)
                continue

            outimg = image.copy()
            results = yolo_model(image)[0]
            detections = []

            # Filter only animal detections
            for det in results.boxes.data.tolist():
                xmin, ymin, xmax, ymax = map(int, det[:4])
                conf, cls_id = det[4], int(det[5])

                if conf > 0.5 and cls_id in animal_class_ids:
                    width = xmax - xmin
                    height = ymax - ymin
                    detections.append(([xmin, ymin, width, height], conf, 'animal'))

            # Update DeepSORT with animal detections
            tracks = tracker.update_tracks(detections, frame=image)
            objects = []

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = track.to_ltrb()
                bbox = [int(l), int(t), int(r), int(b)]

                if track_id not in tracked_objects:
                    tracked_objects[track_id] = object(bbox, id=track_id)
                else:
                    tracked_objects[track_id].update(bbox)

                objects.append(tracked_objects[track_id])

            # Line crossing and area intrusion checks
            checkLineCrosses(boundaryLines, objects)
            drawBoundaryLines(outimg, boundaryLines)

            checkAreaIntrusion(areas, objects)
            drawAreas(outimg, areas)

            # Draw bounding boxes, IDs, and trajectories
            for obj in objects:
                id = obj.id
                color = get_color(id)
                xmin, ymin, xmax, ymax = obj.pos
                cv2.rectangle(outimg, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(outimg, f'ID={id}', (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 1)

                # Optional: draw trajectory
                for i in range(1, len(obj.trajectory)):
                    cv2.line(outimg, obj.trajectory[i - 1], obj.trajectory[i], color, 1)

            # Encode to MJPEG for Flask
            _, buffer = cv2.imencode('.jpg', outimg)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()



# Create an upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-video', methods=['POST'])
def upload_video():
    # if 'video_file' not in request.files:
    #     return "No file part", 400

    # file = request.files['video_file']
    # if file.filename == '':
    #     return "No selected file", 400

    # if file:
    #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #     file.save(filepath)
    #     print(filepath)
    #     # filepath='+'

    #     return redirect(url_for('stream', filename=file.filename))
    global boundaryLines, areas, _N, _C, _H, _W

    if 'video_file' not in request.files:
        return "No file part", 400

    file = request.files['video_file']
    if file.filename == '':
        return "No selected file", 400

    try:
        if file:
            # Save uploaded video
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(f"Saved video to: {filepath}")

            # Parse coordinates
            line1 = list(map(int, request.form.get('line1', '0 0 0 0').replace('[', '').replace(']', '').replace(',', ' ').split()))
            line2 = list(map(int, request.form.get('line2', '0 0 0 0').replace('[', '').replace(']', '').replace(',', ' ').split()))

            area_coords = list(map(int, request.form.get('areas', '').replace('[', '').replace(']', '').replace(',', ' ').split()))
            area_points = [area_coords[i:i+2] for i in range(0, len(area_coords), 2)]

            # Tensor dims
            _N = int(request.form.get('_N', 0))
            _C = int(request.form.get('_C', 0))
            _H = int(request.form.get('_H', 0))
            _W = int(request.form.get('_W', 0))

            # Update global variables
            boundaryLines = [
                boundaryLine(line1),
                boundaryLine(line2)
            ]
            areas = [
                area(area_points)
            ]

            return redirect(url_for('stream', filename=file.filename))

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
@app.route('/stream')
def stream():
    filename = request.args.get('filename')
    if not filename:
        return "No filename provided", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(upload_frames(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # main()

    app.run(debug=True)