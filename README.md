# NVR_Video_Recovery_Honeywell
Forensic pipeline for NVR video recovery (Honeywell HN35080200 NVR) — carves video chunks, attributes source via PRNU or OCR overlay extraction, fills sequence gaps, and reassembles footage.

### The Pipeline
* **Carves video fragments:** Extracts raw video chunks directly from a disk image.
* **Separates fragments:** Distinguishes between eleted video data and newly recorded fragments.
* **Identifies source cameras:** Groups recovered fragments using two methods:
  * **OCR-based:** Extracts visible text overlays from the video frames.
  * **PRNU-based:** Analyzes Photo Response Non-Uniformity (sensor pattern noise).
* **Orders footage:** Sequences the video fragments chronologically using timestamps.
* **Fills sequence gaps:** Restores missing sections using previously unassigned video fragments.
* **Reassembles video:** Concatenates the ordered fragments into a fully playable MP4 file.
