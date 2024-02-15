#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import imutils
import time
import cv2

from settings import STREAM_URL, CAMERA_WIDTH, MIN_AREA, MAX_AREA, DEBUG, \
    REFERENCE_RELEVANT, RELEVANT_DEBOUNCE, OUTPUT_BACKLOG, \
    OUTPUT_INTERVAL, OUTPUT_PATH, OUTPUT_STATIC, NAME_CAM, SHOW_STREAM, SHUFFLE_CAM
import numpy as np
argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-s', '--stream',
    help='Stream URL (RTSP) to get video feed from',
    nargs='?', type=str,
    default=STREAM_URL
)
argparser.add_argument(
    '-w', '--window',
    help='Show relevant feeds as X11 window',
    action='store_true')
argparser.add_argument(
    '--debug',
    help='Output more information',
    action='store_true'
)
argparser.add_argument(
    '-o', '--output',
    help='Stream URL (RTSP) to get video feed from',
    nargs='?', type=str,
    default=NAME_CAM
)
args = argparser.parse_args()
DEBUG = args.debug or DEBUG

# Helper variables:
# last static saving tracking
last_static = datetime.now()
# allow for more consistent "continuous relevant event" handling
debounce_counter = 0


@dataclass
class ReferenceFrames:
    """ Helper class to manage frames """
    frame: object = None
    timestamp: datetime = field(init=False)
    previous: list = field(default_factory=lambda: [])
    latest_capture: object = None

    def set_frame(self, frame: object):
        """ Sets reference frame which is used to calculate difference
        from the current camera image """
        if self.frame is None or self.timestamp <= datetime.now() - timedelta(minutes=REFERENCE_RELEVANT):
            self._set_frame(frame=frame)

    def _set_frame(self, frame: object):
        if DEBUG: print('Updating reference frame')
        self.frame = frame
        self.timestamp = datetime.now()

    def append(self, frame: object, contour: int, contour_index: int, contour_amount: int):
        # Improvement idea: Constant rolling buffer - as soon as occupied=True
        # print("Len previous: ", len(self.previous))
        # if len(self.previous) != 0:
        #     print("Image diff: ", np.sum(cv2.absdiff(frame, self.previous[-1][0])))
        self.previous.append([frame, contour])

        if DEBUG:
            print(f'[{contour_index+1}/{contour_amount}] {contour}')
            self.save_image(frame=frame, contour=contour)

    def unbuffer_previous(self, index_cam: int):
        """ Clean the previous images from buffer and get the most relevant
        photo based on the biggest movement-amount """
        if not self.previous: return

        self.previous = [f for f in self.previous if f[1] < MAX_AREA]
        if len(self.previous) == 0:
            print('Too few pictures to be considered motion event; discarding')
            if DEBUG: print('Too few pictures to be considered motion event; discarding')
            self.previous = list()
            return

        # get middle thirds of list; to get most relevant picture
        image_amount = len(self.previous) // 4
        if image_amount > 4:
            self.previous = self.previous[image_amount:len(self.previous) - image_amount]

        # frame, contour = max(self.previous, key=lambda x: x[1])
        frame, contour = random.choice(self.previous)
        self.latest_capture = frame
        self.previous = list()

        if OUTPUT_BACKLOG:

            file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'

            print(f'Saving image to backlog: {file_name} {str(index_cam)}')
            backlog_path = os.path.join(OUTPUT_PATH, "data", str(index_cam))
            os.makedirs(backlog_path, exist_ok=True)

            cv2.imwrite(os.path.join(backlog_path, file_name), frame)

        if DEBUG: self.save_image(frame=frame, contour=contour, file_prefix='candidate')

    @staticmethod
    def save_image(frame, contour, file_prefix=''):
        if file_prefix: f'{file_prefix}-'
        timestamp = str(datetime.now().strftime('%Y%m%d%H%M%S'))
        file_name = f'{file_prefix}{timestamp}.jpg'
        print(f'{timestamp} Candidate Contour: {contour}')
        print(f'--> saving file to {file_name}')
        cv2.putText(frame, "Contours: {}".format(contour), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(file_name, frame)


def get_stream(url_name=None):
    if url_name:
        return cv2.VideoCapture(url_name, cv2.CAP_FFMPEG)
    if not args.stream:
        print('Stream URI for RTSP server not specified! Exiting')
        exit(1)
    return cv2.VideoCapture(args.stream, cv2.CAP_FFMPEG)


if __name__ == '__main__':
    print('Initializing stream...')
    frames = ReferenceFrames()
    vs = get_stream()
    count = 0
    while True:

        # Renew stream object
        count += 1
        if count == 1000:
            print("Reset Stream!")
            count = 0
            vs.release()
            vs.open(args.stream)
            assert vs.isOpened()

        # grab the current frame and initialize the occupied/unoccupied
        retrieved, full_frame = vs.read()
        if not retrieved:
            print('Error retrieving image from stream; reinitializing')
            vs.release()
            vs = get_stream()
            continue

        if full_frame is None: continue
        occupied = False

        # resize the frame, convert it to grayscale, and blur it
        scaled_frame = imutils.resize(full_frame, width=CAMERA_WIDTH)
        y, x, channels = scaled_frame.shape
        #frame = full_frame[:, START_CROP_X:x]
        frame = scaled_frame.copy()

        # src_cropped = src[top_margin:src.shape[0], :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if frames.frame is None:
            frames.set_frame(frame=gray)
            continue

        # compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(frames.frame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # if the contour is too small, ignore it
        relevant_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
        contour_sizes = [cv2.contourArea(c) for c in relevant_contours]

        for i, (contour, contour_size) in enumerate(zip(relevant_contours, contour_sizes)):
            # reset reference picture; this is to help detect if there's actual motion
            # if multiple consecutive pictures change, it's likely we are dealing with motion
            frames._set_frame(frame=gray)
            # compute the bounding box for the contour, draw it on the frame,
            # and update the status
            # (x, y, w, h) = cv2.boundingRect(contour)
            # x = x + START_CROP_X  # ensure relative boxes are rendered properly
            # cv2.rectangle(scaled_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            debounce_counter = RELEVANT_DEBOUNCE
            occupied = True
            frames.append(
                frame=scaled_frame,
                contour=contour_size,
                contour_index=i,
                contour_amount=len(relevant_contours)
            )

        if not occupied:
            if debounce_counter > 0: debounce_counter -= 1
            if not debounce_counter:
                frames.set_frame(frame=gray)
                frames.unbuffer_previous(index_cam=NAME_CAM)

        # save image to output static image every two seconds
        if OUTPUT_STATIC and last_static < datetime.now() - timedelta(seconds=OUTPUT_INTERVAL):
            last_static = datetime.now()
            file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
            print(f'Saving image to backlog: {file_name} {args.output}')
            backlog_path = os.path.join(OUTPUT_PATH, args.output)
            os.makedirs(backlog_path, exist_ok=True)
            cv2.imwrite(os.path.join(backlog_path, file_name), scaled_frame)

        # show the frame and record if the user presses a key
        if SHOW_STREAM:
            cv2.imshow("Security Feed {}".format(args.output), full_frame)
            if frames.latest_capture is not None:
                cv2.imshow("Captured {}".format(args.output), frames.latest_capture)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    vs.release()  # vs.stop()
    cv2.destroyAllWindows()
