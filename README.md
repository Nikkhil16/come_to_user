# come_to_user

`come_to_user` is a ROS 2 skill package for the MARS-style robot camera layout. It runs the existing person-following script as a package executable, tracks the largest detected person in the main camera, publishes debug image topics, commands `/cmd_vel`, and latches a stop when the arm camera detects an edge in the forward corridor.

Before enabling the follower, the script moves the arm to a configured start pose through `/mars/arm/goto_js` using the `maurice_msgs/srv/GotoJS` service type. Make sure that interface package is already available in the robot ROS environment.

## Runtime Dependencies

Install the ROS dependencies for your distro:

```bash
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-web-video-server python3-opencv python3-numpy
```

The script also imports Ultralytics YOLO:

```bash
pip3 install -r come_to_user/requirements.txt
```

If you are installing from the package share directory instead of source, run `pip3 install -r install/come_to_user/share/come_to_user/requirements.txt`.

Make sure the configured YOLO model is available on the robot, or pass a model path with `--model`. The package default is `yolo26n.pt` because that is what the working script used; if that is a custom local weight file, keep it beside the launch environment or pass its absolute path.

## Build

From the ROS workspace root:

```bash
colcon build --packages-select come_to_user
source install/setup.bash
```

## Run Directly

```bash
ros2 run come_to_user come_to_user_script.py
```

Useful overrides:

```bash
ros2 run come_to_user come_to_user_script.py \
  --preferred /mars/main_camera/remote/left/image_raw \
  --arm-image-topic /mars/arm/image_raw \
  --cmd-vel-topic /cmd_vel \
  --model yolo26n.pt
```

Use `--skip-arm-start` only when the arm is already in the required pose.

If the browser page does not load from your laptop, first try the printed `Server root` URL. If the script only prints `127.0.0.1`, rerun with the robot's LAN address:

```bash
ros2 run come_to_user come_to_user_script.py --robot-ip ROBOT_LAN_IP
```

If the server root loads but the combined debug view is blank, wait for the script to finish arm start-pose setup and camera topic discovery. The script also prints a raw stream URL for `/combined_debug/image_raw` as a second way to test `web_video_server`.

The default arm edge detector is intentionally a bit lenient: lower gradient thresholds, lower support/span requirements, a wider center corridor, and a slightly earlier stop line. For fewer false stops, increase `--edge-grad-threshold`, `--edge-min-support-fraction`, or `--edge-min-corridor-support-fraction`.

## Use With skill_acq

Rebuild the local catalog after adding or editing this package:

```bash
ros2 run skill_acq build_package_catalog.py --root /home/nikhil/workspace/RoboUniversity/RoboUniversity
```

Then ask `skill_acq` to start it:

```bash
ros2 run skill_acq skill_acq.py \
  "make the robot come to the user" \
  --robot-mode physical \
  --robot-has-estop true \
  --leave-processes-running
```

The runner manifest starts the follower as a background process. Keep `--leave-processes-running` when launching through `skill_acq`, otherwise the runner will clean up the started process after the short confirmation client exits.

When launched through `skill_acq`, the runner first runs `rosdep install` for the package dependencies declared in `package.xml`, then installs this package's `requirements.txt` because `package_runner.json` declares it in `python_requirements`.

The phrase "come to me" is covered by the `package_runner.json` target keywords, so `skill_acq` selects the `come_to_user` target and starts `come_to_user_script.py`; the arm start-pose call happens inside that script before the follower starts.
