# CCTV Motion Detection System
This repository contains a Python script for collecting frames from CCTV cameras when motion is detected
within the camera's view. It reads RTSP addresses from a `list_cam.txt` file and parallelly connects to 
each RTSP stream to collect frames.


## Setting

```

STREAM_URL: URL to grab and analyze the stream from.

CAMERA_WIDTH: Resize original picture to this width (adjust MIN_AREA, MAX_AREA, and CROP_X if changed).

REFERENCE_RELEVANT: Minutes the reference image is considered relevant; important for accurate results during day-night transition.

RELEVANT_DEBOUNCE: Number of frames in order to be irrelevant to consider a movement event finished.

MIN_AREA: Minimum number of pixels that have to change in order to be considered movement.

MAX_AREA: Maximum pixels changing - everything above is considered a change in scenery (e.g., light on during the night), 
which results in not being considered for being a valid candidate.

OUTPUT_STATIC: Periodically save static images of the stream.

OUTPUT_PATH: Path to save static images.

OUTPUT_INTERVAL: Interval for saving static images.

OUTPUT_BACKLOG: Save the most relevant static image of captured motion events.

DEBUG: Enable debug mode.

NAME_CAM: Specify the camera name.
SHOW_STREAM: Show the stream.
SHUFFLE_CAM: Shuffle cameras.
```

## Usage
1. Ensure you have Python installed on your system.
2. Clone this repository.
3. Configure the settings in the script according to your requirements. 
4. Run the script using `python run_thread.py`

## Notes
Make sure to provide the appropriate RTSP addresses in the `list_cam.txt` file.

Adjust the settings according to your environment and preferences.

## Contributing
Contributions are welcome! Feel free to open issues and submit pull requests.





