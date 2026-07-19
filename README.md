# NVR_Video_Recovery_Honeywell
Forensic pipeline for NVR video recovery (Honeywell HN35080200 NVR) — carves video chunks, attributes source via PRNU or OCR overlay extraction, fills sequence gaps, and reassembles footage.

### The Pipeline
* **Carves video fragments:** Extracts raw video chunks directly from a disk image.
* **Separates fragments:** Distinguishes between deleted video data and newly recorded fragments.
* **Identifies source cameras:** Groups recovered fragments using two methods:
  * **OCR-based:** Extracts visible text overlays from the video frames using PaddleOCR-VL-1.5.
  * **PRNU-based:** Analyzes Photo Response Non-Uniformity (sensor pattern noise).
* **Orders footage:** Sequences the video fragments chronologically using timestamps.
* **Fills sequence gaps:** Restores missing sections using previously unassigned video fragments.
* **Reassembles video:** Concatenates the ordered fragments into a fully playable MP4 file.

### Repository Structure
| File Name | Description | 
| -------- | -------- | 
| carver.py | Carves video fragments from Honeywell NVR disk, detects H.264/H.265 fragments, and separates old and new recording sessions.| 
| paddleOCR.py | Extracts overlay regions, applies image preprocessing (overlay cropping and candidate image generation), extracts the text, groups fragments based on the camera label, and orders based on the extracted timestamp.|
| prnu.py | Extracts PRNU fingerprints, groups fragments based on fingerprint, and orders video fragments based on the embedded timestamp.| 
| gap_filler.py | Detects gaps in the grouped and ordered video fragments based on the embedded timestamp and searches for a candidate fragment that covers the gap.| 
| concat.py | Concatenates the ordered video fragments and merges into a playable MP4 file using FFmpeg.| 
