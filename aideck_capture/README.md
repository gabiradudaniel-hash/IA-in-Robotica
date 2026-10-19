# AI-deck capture

Save images from a Crazyflie AI-deck WiFi stream. No ROS and no extra Python packages.

## What to install

Nothing beyond Python 3. The script uses only the standard library (`socket`, `struct`, `zlib`).

Check that Python is available:

```bash
python3 --version
```

macOS already includes this. There is no `pip install` step.

## Run

Join the AI-deck WiFi, then:

```bash
cd aideck_capture
python3 capture.py --save-dir recordings
```

The deck address defaults to `192.168.4.1:5000`. Stop with Ctrl-C. Each file is named with the capture time in Unix milliseconds, for example `1712345678901.jpg`.
