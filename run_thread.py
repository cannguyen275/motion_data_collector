import time
import random
from camera import *
from multiprocessing import cpu_count, Pool

list_cam = []
with open("list_cam.txt", "r") as f:
    list_cam = [i for line in f for i in line.split(',')]
if SHUFFLE_CAM:
    random.shuffle(list_cam)

def process(index):
    rtsp_url = list_cam[index].strip()
    # Helper variables:
    # last static saving tracking
    last_static = datetime.now()
    # allow for more consistent "continuous relevant event" handling
    debounce_counter = 0

    print("Index: {} Cam: {}".format(index, rtsp_url))
    print('Initializing stream...')
    frames = ReferenceFrames()
    vs = get_stream(url_name=rtsp_url)
    count = 0
    connect_retry_count = 0
    # Set the desired frame rate (1 frame per second)
    fps = 1
    milliseconds_per_frame = int(1000 / fps)
    occupied_image_with_cam = False
    patience = 0
    debounce_counter = RELEVANT_DEBOUNCE
    start_time = time.time()
    last_success_time = time.time()
    
    while True:
        # If cam doesnot provide any data for long time, ignore it
        if (count % 60 == 0) and not occupied_image_with_cam:
            print("Increase patience for cam: ", index)
            patience += 1
        if patience > 5:
            print("Cam {} doesn't provide data! Ignore!".format(index))
            break

        # Renew stream object
        count += 1
        if count == 1000:
            print("Reset Stream!")
            count = 0
            vs.release()
            vs.open(rtsp_url)
            # assert vs.isOpened()

        # grab the current frame and initialize the occupied/unoccupied
        vs.set(cv2.CAP_PROP_POS_MSEC, milliseconds_per_frame)
        retrieved, full_frame = vs.read()
        if not retrieved:
            connect_retry_count += 1
            if connect_retry_count >= 5:
                print("Retry not sucessfull!")
                break
            print('Error retrieving image from stream; reinitializing #{}'.format(connect_retry_count))
            vs.release()
            vs = get_stream(url_name=rtsp_url)
            continue

        if full_frame is None: continue


        # resize the frame, convert it to grayscale, and blur it
        scaled_frame = imutils.resize(full_frame, width=CAMERA_WIDTH)
        y, x, channels = scaled_frame.shape
        # frame = full_frame[:, START_CROP_X:x]
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
        # print("len contours: ", len(contours))
        relevant_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
        # print("len relevant contours: ", len(relevant_contours))
        contour_sizes = [cv2.contourArea(c) for c in relevant_contours]
        if len(relevant_contours) == 0:
            if (time.time() - last_success_time) > 300:
                occupied_image_with_cam = False
        for i, (contour, contour_size) in enumerate(zip(relevant_contours, contour_sizes)):
            # reset reference picture; this is to help detect if there's actual motion
            # if multiple consecutive pictures change, it's likely we are dealing with motion
            frames._set_frame(frame=gray)
            # compute the bounding box for the contour, draw it on the frame,
            # and update the status
            # (x, y, w, h) = cv2.boundingRect(contour)
            # x = x + START_CROP_X  # ensure relative boxes are rendered properly
            # cv2.rectangle(scaled_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            debounce_counter -= 1
            # print("Debounce Counter in contour: ", debounce_counter)
            # print("Occupied in contour: ", occupied)
            frames.append(
                frame=scaled_frame,
                contour=contour_size,
                contour_index=i,
                contour_amount=len(relevant_contours)
            )

        # if not occupied:
            # if debounce_counter > 0:
            #         debounce_counter -= 1

        if debounce_counter < 0:
            frames.set_frame(frame=gray)
            frames.unbuffer_previous(index_cam=index)
            occupied_image_with_cam = True
            debounce_counter = RELEVANT_DEBOUNCE
            last_success_time = time.time()
            patience = 0

        # show the frame and record if the user presses a key
        if SHOW_STREAM:
            cv2.imshow("Security Feed {}".format(index), full_frame)
            if frames.latest_capture is not None:
                cv2.imshow("Captured {}".format(index), frames.latest_capture)

        now_static = datetime.now()
        if (now_static - last_static).seconds > 2000:
            break
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    vs.release()  # vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Number of CPU: ", cpu_count())
    # pool = Pool(cpu_count() - 1)
    for i in range(0, 100):
        print("Loop {}".format(i))
        pool = Pool(1)

        try:
            pool.map(process, range(len(list_cam) - 1))
            # pool.map(process, [0, 1])
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()



