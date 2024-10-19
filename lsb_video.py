import cv2
import numpy as np
import os

def text_to_bin(text):
    return ''.join([format(ord(char), '08b') for char in text])

def bin_to_text(binary):
    chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
    text = ''.join([chr(int(char, 2)) for char in chars if char != '11111111'])
    return text

def encode_message_in_frame(frame, message, delimiter='11111110'):
    binary_message = text_to_bin(message) + delimiter
    print(binary_message)
    message_length = len(binary_message)
    frame_flat = frame.flatten()
    print(frame_flat[0])
    if message_length > len(frame_flat):
        raise ValueError("Message is too large to encode in this frame.")

    for i in range(message_length):
        _flat = format(frame_flat[i], 'b')
        _flat = _flat[:-1] + binary_message[i]
        frame_flat[i] = int(_flat, 2)

    frame = frame_flat.reshape(frame.shape)
    return frame

def decode_message_from_frame(frame, delimiter='11111110'):
    frame_flat = frame.flatten()
    
    binary_message = ''
    for i in frame_flat:
        bin = format(i, 'b')
        last_char = bin[-1]
        binary_message = binary_message + last_char
        end_index = binary_message.find(delimiter)
        if end_index != -1:
            break
    
    decoded_message = bin_to_text(binary_message)
    return decoded_message

def encode_message_in_video(input_video_path, message):
    temp_video_path = 'temp_video_lossless.avi'
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')

    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video writer for file {temp_video_path}")

    print(f"Message to embed: {message}")
    print(f"Binary message: {text_to_bin(message)}")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == 0:
            try:
                frame = encode_message_in_frame(frame, message)
                print(f"Encoded message into frame {frame_idx + 1}")
            except ValueError as ve:
                print(f"Error: {ve}")
                cap.release()
                out.release()
                return

        out.write(frame)
        frame_idx += 1
        print(f"Processed frame {frame_idx}/{total_frames}", end='\r')

    cap.release()
    out.release()

    os.remove(input_video_path)
    os.rename(temp_video_path, input_video_path)

    print("\nMessage encoded successfully!")
    
def decode_message_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Cannot read the first frame of the video.")

    message = decode_message_from_frame(frame)
    cap.release()

    if message:
        print(f"Decoded message: {message[:-1]}")
    else:
        print("No hidden message found or message is incomplete.")

    return message

if __name__ == "__main__":
    input_video = 'test_1 copy.mp4'
    secret_message = "This is the hidden message!"

    print(text_to_bin(secret_message))

    encode_message_in_video(input_video, secret_message)

    decode_message_from_video(input_video)
