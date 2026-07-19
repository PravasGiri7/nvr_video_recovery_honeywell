# NVR_Video_Recovery_Honeywell
Forensic pipeline for NVR video recovery — carves video chunks, attributes source via PRNU or OCR overlay extraction, fills sequence gaps, and reassembles footage.
The pipeline:
1. Carves video fragments from a raw disk image
2. Separates previously recorded (deleted video) and newly recorded fragments.
3. Groups recovered fragments by source camera identification:
   OCR-based source camera identification using visible overlays.
   PRNU-based source camera identification using sensor pattern noise.
4. Orders the video fragments using the timestamp.
5. Restores the gaps using unassigned video fragments.
6. Concatenates the ordered fragments to produce a playable MP4 file.


