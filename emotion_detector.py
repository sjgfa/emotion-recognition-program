import cv2
from deepface import DeepFace
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def run_emotion_detector():
    """
    Initializes the camera and runs emotion detection.
    """
    # DeepFace may download models the first time it runs, ensure network connection.
    # The emotions list includes emotions DeepFace can recognize.
    emotions_list = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # Define emotion English to Chinese mapping
    emotion_translation_zh = {
        "angry": "生气",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "开心",
        "sad": "悲伤",
        "surprise": "惊讶",
        "neutral": "中性"
    }

    # Define emotion English directly (for English display)
    emotion_translation_en = {
        "angry": "Angry",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happy",
        "sad": "Sad",
        "surprise": "Surprise",
        "neutral": "Neutral"
    }

    # --- Configuration for display language ---
    # Set to 'en' for English, 'zh' for Chinese
    display_language = 'zh' # Default to Chinese for display. Change to 'en' for English.
    # --- End of Configuration ---

    # Load a font file that supports Chinese characters
    # You might need to adjust this path based on your operating system and font installation.
    # Common Windows fonts: "simhei.ttf" (SimHei), "msyh.ttc" (Microsoft YaHei)
    # Common macOS fonts: "/System/Library/Fonts/PingFang.ttc"
    # Common Linux fonts: Look for .ttf or .ttc files in /usr/share/fonts/
    font_path = "C:/Windows/Fonts/msyh.ttc" # Assuming Windows, using Microsoft YaHei
    font_size = 30 # Font size, adjustable as needed
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Could not load font file '{font_path}'. Please ensure the file exists and is accessible. Using default font.")
        font = ImageFont.load_default() # Fallback to default font if loading fails

    cap = cv2.VideoCapture(0) # 0 represents the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting emotion detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='retinaface')

            for result in results:
                # Check if all required keys exist
                if "region" in result and "dominant_emotion" in result and "emotion" in result:
                    face_coords = result["region"]
                    dominant_emotion = result["dominant_emotion"]
                    emotion_scores = result["emotion"]

                    x, y, w, h = face_coords["x"], face_coords["y"], face_coords["w"], face_coords["h"]
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Get the dominant emotion and its confidence score
                    score = emotion_scores.get(dominant_emotion, 0.0)
                    
                    # Translate emotion to desired language
                    if display_language == 'zh':
                        display_emotion = emotion_translation_zh.get(dominant_emotion, dominant_emotion)
                    else: # default to English
                        display_emotion = emotion_translation_en.get(dominant_emotion, dominant_emotion)
                    
                    text = f"{display_emotion}: {score:.2f}"

                    # Draw text using Pillow on OpenCV image
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    # Adjust text position slightly if needed
                    text_x = x
                    text_y = y - font_size - 5
                    # Ensure text is not drawn out of bounds at the top of the image
                    if text_y < 0:
                        text_y = y + h + 5 # Draw below the bounding box if space is limited above
                    
                    draw.text((text_x, text_y), text, font=font, fill=(0, 255, 0, 0)) # Font color, last value is alpha (ignored for RGB)
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                else:
                    # This warning is less likely now due to previous key corrections,
                    # but kept for robustness or can be changed to more detailed logging.
                    pass # Do not print this warning to avoid excessive output

        except ValueError as e:
            if "Face could not be detected" in str(e):
                pass # Do not print error if no face is detected, as it's a common scenario.
            else:
                print(f"DeepFace analysis error: {e}")
        except Exception as e:
            print(f"An unknown error occurred: {e}")

        # Display the result frame
        cv2.imshow('Emotion Detection', frame) # Window title also translated

        # Press 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Emotion detection stopped.")

if __name__ == "__main__":
    run_emotion_detector() 