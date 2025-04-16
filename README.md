# Wildlife Biodiversity Monitoring with YOLOv8

This project uses a YOLOv8 deep learning model and a Flask API to automatically detect and classify wildlife species from aerial drone footage.

It is part of a larger system for biodiversity monitoring, where users can upload videos of natural habitats and receive instant AI-based analysis of the wildlife observed.

---

##  What This Project Does

- Runs a YOLOv8 object detection model trained on six wildlife species
- Accepts drone-captured video uploads via a POST API endpoint (`/analyze`)
- Returns a list of detected species found in the video
- Works with a simple HTML front-end dashboard for uploads and results

---

##  Technologies Used

- Python + Flask  
- YOLOv8 (Ultralytics)  
- OpenCV (for video frame reading)  
- Render (to host the backend API)  
- HTML + JS (frontend dashboard)

---

##  How to Use

1. Clone this repository
2. Place your trained `yolov8.pt` model in the project root directory
3. Install the dependencies:

4.Run the Flask server:

5.Use the /analyze endpoint to upload videos and receive wildlife species predictions


## Live API on Render
Once deployed, the API will be live at:
https://wildlife-bio.onrender.com/

Use this URL in your HTML dashboard to send video files.

## File Structure

## Developed by
Aaesha & Team
Project: Utilization of Deep Learning for Wildlife Biodiversity Monitoring
Year: 2025
   
