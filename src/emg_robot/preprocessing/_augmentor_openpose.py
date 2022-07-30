import numpy as np 
import cv2
from openpose import pyopenpose as op


openpose_proto = "openpose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
openpsoe_weights = "openpose/mpi/pose_iter_160000.caffemodel"

shoulder_idx = 2
elbow_idx = 3
wrist_idx = 4


def find_sync_markers_emg(emg, expected_len_samples, threshold):
    # See https://stackoverflow.com/questions/47519626/using-numpy-scipy-to-identify-slope-changes-in-digital-signals
    pass # TODO


def find_sync_markers_openpose(op_seq):
    pass # TODO


# See https://chowdera.com/2021/07/20210724151058183u.html

def calc_angle_of_points(p0, p1, p2):
    a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
    if a * b == 0:
        return -1.0 
    
    return  math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180 /math.pi


def calc_groundtruth(df, vid, params=None):
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise RuntimeError('Could not open VideoCapture for ' + vid)

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    rate = cap.get(cv2.CAP_PROP_FPS)

    # See https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
    openpose = op.WrapperPython()
    openpose.configure(params or {})
    openpose.start()
    angles = np.zeros([frames])

    for i in range(frames):
        success, frame = cap.read()
        if success:
            pose = op.Datum()
            pose.cvInputData = frame
            openpose.emplaceAndPop(op.VectorDatum([pose]))
            shoulder = (pose.poseKeypoints[0][shoulder_idx][0], pose.poseKeypoints[0][shoulder_idx][1])
            elbow = (pose.poseKeypoints[0][elbow_idx][0], pose.poseKeypoints[0][elbow_idx][1])
            wrist = (pose.poseKeypoints[0][wrist_idx][0], pose.poseKeypoints[0][wrist_idx][1])
            angles[i] = calc_angle_of_points(shoulder, elbow, wrist)
        else: 
            break

    # TODO !!! stretch and interpolate angles according to sync marks so they match EMG data!
    df['openpose'] = angles
