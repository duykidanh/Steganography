import cv2
import numpy as np
import os
import math

def string_to_bits(s):
    bits = []
    for char in s:
        binval = bin(ord(char))[2:].rjust(8, '0')
        bits.extend([int(b) for b in binval])
    return bits

def array_to_string(bits):
    message = ''
    for b in range(len(bits)):
        message += str(bits[b])
    return message

def text_to_bin(text):
    return ''.join([format(ord(char), '08b') for char in text])

def bin_to_text(binary):
    chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
    text = ''.join([chr(int(char, 2)) for char in chars if char != '11111111'])
    return text

def extract_bit_planes(channel):
    bit_planes = []
    for i in range(8):
        # Trích xuất bit i bằng cách dịch phải i bit và lấy bit cuối cùng
        bit_plane = np.bitwise_and(channel, 1 << i) >> i
        bit_planes.append(bit_plane)
    return bit_planes

def calculate_entropy(bit_plane, frame_height, frame_width):
    bit_lane = (frame_height)*(frame_width - 1) + (frame_height - 1)*(frame_width)
    different_lane = 0
    for i in range(frame_height-1):
        for j in range(frame_width-1):
            if bit_plane[i+1, j]:
                if bit_plane[i, j] != bit_plane[i+1, j]:
                    different_lane += 1
            if bit_plane[i, j+1]:
                if bit_plane[i, j] != bit_plane[i, j+1]:
                    different_lane += 1
    entropy_val = different_lane / bit_lane
    return entropy_val

def evaluate_bit_planes(bit_planes, frame_height, frame_width):
    entropies = []
    for plane in bit_planes:
        entropy_val = calculate_entropy(plane, frame_height, frame_width)
        entropies.append(entropy_val)
    return entropies

def select_complex_bit_planes(entropies, threshold=0.05):
    selected = []
    for i, entropy_val in enumerate(entropies):
        if entropy_val >= threshold:
            selected.append(i)
    return selected

def embed_data_into_bit_plane(bit_plane, data_bits, start_index=0):
    rows, cols = bit_plane.shape
    total_bits = len(data_bits)
    embedded = bit_plane.copy()
    bit_count = 0

    for i in range(rows):
        for j in range(cols):
            if bit_count >= total_bits:
                return embedded, bit_count
            embedded[i, j] = data_bits[bit_count]         
            bit_count += 1
    return embedded, bit_count

def encode_message_in_video(input_video_path, message, delimiter='11111110', plane_info_path='bit_plane_locations.txt'):
    temp_video_path = 'temp_video_lossless.avi'
    data_bits = string_to_bits(message)
    data_bits += [int(bit) for bit in delimiter]
    data_length = len(data_bits)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Cannot open video file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'ffv1')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video writer for file {temp_video_path}")

    # Giới hạn số bit nhúng để tránh vượt quá dung lượng video
    max_capacity = frame_width * frame_height * 3 * 3  # 3 kênh màu, 3 bit-plane mỗi kênh
    if data_length > max_capacity:
        print(f"Message is too large to encode in this frame")
        return

    bit_index = 0
    plane_info = []  # List to store bit-plane locations
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        B, G, R = cv2.split(frame)

        bit_planes_R = extract_bit_planes(R)
        bit_planes_G = extract_bit_planes(G)
        bit_planes_B = extract_bit_planes(B)

        entropies_R = evaluate_bit_planes(bit_planes_R, frame_height, frame_width)
        entropies_G = evaluate_bit_planes(bit_planes_G, frame_height, frame_width)
        entropies_B = evaluate_bit_planes(bit_planes_B, frame_height, frame_width)

        selected_R = select_complex_bit_planes(entropies_R)
        selected_G = select_complex_bit_planes(entropies_G)
        selected_B = select_complex_bit_planes(entropies_B)

        if(selected_B != []):
            for plane_idx in selected_B:
                if bit_index >= data_length:
                    break
                bit_planes_B[plane_idx], bits_embedded = embed_data_into_bit_plane(
                    bit_planes_B[plane_idx],
                    data_bits[bit_index:]
                )
                bit_index += bits_embedded
                plane_info.append(('B', frame_index, plane_idx))  # Save plane info
        if(selected_G != []):
            for plane_idx in selected_G:
                if bit_index >= data_length:
                    break
                bit_planes_G[plane_idx], bits_embedded = embed_data_into_bit_plane(
                    bit_planes_G[plane_idx],
                    data_bits[bit_index:]
                )
                bit_index += bits_embedded
                plane_info.append(('G', frame_index, plane_idx))  # Save plane info
        if(selected_R != []):
            for plane_idx in selected_R:
                if bit_index >= data_length:
                    break
                bit_planes_R[plane_idx], bits_embedded = embed_data_into_bit_plane(
                    bit_planes_R[plane_idx],
                    data_bits[bit_index:]
                )
                bit_index += bits_embedded
                plane_info.append(('R', frame_index, plane_idx))  # Save plane info

        def combine_bit_planes(bit_planes):
            channel = np.zeros_like(bit_planes[0], dtype=np.uint8)
            for i in range(8):
                channel += (bit_planes[i] << i)
            return channel

        R_new = combine_bit_planes(bit_planes_R)
        G_new = combine_bit_planes(bit_planes_G)
        B_new = combine_bit_planes(bit_planes_B)

        frame_bgr_new = cv2.merge([B_new, G_new, R_new])

        out.write(frame_bgr_new)

        if bit_index >= data_length:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            break
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save the bit-plane locations to a file
    with open(plane_info_path, 'w') as f:
        for channel, index, plane_idx in plane_info:
            f.write(f"{channel},{index},{plane_idx}\n")

    os.remove(input_video_path)
    os.rename(temp_video_path, input_video_path)

    print("\nMessage encoded successfully!")


def decode_message_from_video(video_path, delimiter='11111110', plane_info_path='bit_plane_locations.txt'):
    cap_embedded = cv2.VideoCapture(video_path)
    if not cap_embedded.isOpened():
        print("Cannot open video file.")
        return

    extracted_bits = []
    extracted_message = ""

    # Read the bit-plane locations from the file
    with open(plane_info_path, 'r') as f:
        plane_info = [line.strip().split(',') for line in f]
    frame_index = 0
    while True:
        ret, frame = cap_embedded.read()
        if not ret:
            break

        B, G, R = cv2.split(frame)

        bit_planes_R = extract_bit_planes(R)
        bit_planes_G = extract_bit_planes(G)
        bit_planes_B = extract_bit_planes(B)

        for channel, index, plane_idx_str in plane_info:
            plane_idx = int(plane_idx_str)
            bits = []
            if(int(index) == frame_index):
                if channel == 'B':
                    bits = bit_planes_B[plane_idx].flatten().tolist()
                elif channel == 'G':
                    bits = bit_planes_G[plane_idx].flatten().tolist()
                elif channel == 'R':
                    bits = bit_planes_R[plane_idx].flatten().tolist()
                print(bits)
                extracted_bits.extend(bits)
                if array_to_string(extracted_bits).find(delimiter) != -1:
                    break

        string_bits = array_to_string(extracted_bits)
        if string_bits.find(delimiter) != -1:
            break
        frame_index += 1

    cap_embedded.release()

    delimiter_index = array_to_string(extracted_bits).find(delimiter)
    if delimiter_index != -1:
        binary_message = array_to_string(extracted_bits)[:delimiter_index]
    else:
        print("Delimiter not found. Message may be incomplete.")
        binary_message = array_to_string(extracted_bits)

    message = bin_to_text(binary_message)
    print(f"Decoded message: {message}")

    return message


if __name__ == "__main__":
    input_video = 'test_2 copy.mp4'
    secret_message = "This is the hidden message!"

    
    #encode_message_in_video(input_video, secret_message)
    decode_message_from_video(input_video)
