import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def string_to_bits(s):
    """
    Chuyển đổi chuỗi ký tự thành danh sách bit.
    """
    bits = []
    for char in s:
        binval = bin(ord(char))[2:].rjust(8, '0')
        bits.extend([int(b) for b in binval])
    return bits

def bits_to_string(bits):
    """
    Chuyển đổi danh sách bit thành chuỗi ký tự.
    """
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b*8:(b+1)*8]
        byte_str = ''.join([str(bit) for bit in byte])
        chars.append(chr(int(byte_str, 2)))
    return ''.join(chars)

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
    """
    Trích xuất 8 bit-plane từ một kênh màu.
    B0 là LSB và B7 là MSB.
    """
    bit_planes = []
    for i in range(8):
        # Trích xuất bit i bằng cách dịch phải i bit và lấy bit cuối cùng
        bit_plane = np.bitwise_and(channel, 1 << i) >> i
        bit_planes.append(bit_plane)
    return bit_planes

def calculate_entropy(bit_plane):
    """
    Tính entropy của một bit-plane để đánh giá độ phức tạp.
    Entropy cao tương ứng với độ phức tạp cao.
    """
    hist, _ = np.histogram(bit_plane, bins=2, range=(0, 1))
    p1 = hist[0] / hist.sum()
    p2 = hist[1] / hist.sum()
    entropy = 0
    for p in [p1, p2]:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def evaluate_bit_planes(bit_planes):
    """
    Đánh giá độ phức tạp của mỗi bit-plane dựa trên entropy.
    Trả về danh sách entropy cho từng bit-plane.
    """
    entropies = []
    for plane in bit_planes:
        entropy = calculate_entropy(plane)
        entropies.append(entropy)
    return entropies

def select_complex_bit_planes(entropies, threshold=0.9):
    """
    Chọn các bit-plane có entropy cao hơn ngưỡng threshold.
    Ngưỡng có thể được điều chỉnh tùy thuộc vào yêu cầu.
    """
    selected = []
    for i, entropy in enumerate(entropies):
        if entropy >= threshold:
            selected.append(i)
    return selected
    
def embed_data_into_bit_plane(bit_plane, data_bits, start_index=0):
    """
    Nhúng dữ liệu vào bit-plane đã chọn bắt đầu từ start_index.
    Trả về bit-plane đã được nhúng và số bit đã nhúng.
    """
    rows, cols = bit_plane.shape
    total_bits = len(data_bits)
    embedded = bit_plane.copy()
    bit_count = 0

    for i in range(rows):
        for j in range(cols):
            if bit_count >= total_bits:
                return embedded, bit_count
            # Nhúng bit vào bit-plane
            embedded[i, j] = data_bits[bit_count]         
            bit_count += 1
    return embedded, bit_count

def extract_data_from_bit_plane(bit_plane, delimiter='11111110'):
    """
    Trích xuất dữ liệu từ bit-plane.
    Trả về danh sách các bit đã trích xuất.
    """
    rows, cols = bit_plane.shape
    extracted_bits = []
    bit_count = 0

    for i in range(rows):
        for j in range(cols):
            string_bits = array_to_string(extracted_bits)
            if (string_bits.find(delimiter) != -1):
                return extracted_bits
            extracted_bits.append(bit_plane[i, j])
            bit_count += 1
    return extracted_bits

def encode_message_in_video(input_video_path, message, delimiter='11111110'):
    temp_video_path = 'temp_video_lossless.avi'
    data_bits = string_to_bits(message)
    for b in range(len(delimiter)):
        data_bits.append(int(delimiter[b]))
    data_length = len(data_bits)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Cannot open video file.")
        return

    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video writer for file {temp_videoS_path}")

    # Giới hạn số bit nhúng để tránh vượt quá dung lượng video
    max_capacity = frame_width * frame_height * 3 * 3  # Giả sử có 3 kênh màu và mỗi kênh chọn tối đa 3 bit-plane
    if data_length > max_capacity:
        print(f"Dữ liệu quá lớn để nhúng vào video. Dung lượng tối đa là {max_capacity} bit.")
        return

    bit_index = 0  # Chỉ số bit dữ liệu đã nhúng

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # # Chuyển đổi từ BGR (OpenCV) sang RGB
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phân tách các kênh màu
        B, G, R = cv2.split(frame)

        # Trích xuất bit-plane cho mỗi kênh màu
        bit_planes_R = extract_bit_planes(R)
        bit_planes_G = extract_bit_planes(G)
        bit_planes_B = extract_bit_planes(B)

        # Đánh giá độ phức tạp (entropy) của mỗi bit-plane
        entropies_R = evaluate_bit_planes(bit_planes_R)
        entropies_G = evaluate_bit_planes(bit_planes_G)
        entropies_B = evaluate_bit_planes(bit_planes_B)

        # Chọn các bit-plane phức tạp cao (entropy >= 0.9)
        threshold = 0.9
        selected_R = select_complex_bit_planes(entropies_R, threshold)
        selected_G = select_complex_bit_planes(entropies_G, threshold)
        selected_B = select_complex_bit_planes(entropies_B, threshold)

        # Nhúng dữ liệu vào các bit-plane được chọn
        for plane_idx in selected_B:
            if bit_index >= data_length:
                break
            # Nhúng dữ liệu vào bit-plane
            bit_planes_B[plane_idx], bits_embedded = embed_data_into_bit_plane(
                bit_planes_B[plane_idx],
                data_bits[bit_index:]
            )
            bit_index += bits_embedded

        for plane_idx in selected_G:
            if bit_index >= data_length:
                break
            bit_planes_G[plane_idx], bits_embedded = embed_data_into_bit_plane(
                bit_planes_G[plane_idx],
                data_bits[bit_index:]
            )
            bit_index += bits_embedded

        for plane_idx in selected_R:
            if bit_index >= data_length:
                break
            bit_planes_R[plane_idx], bits_embedded = embed_data_into_bit_plane(
                bit_planes_R[plane_idx],
                data_bits[bit_index:]
            )
            bit_index += bits_embedded

        # Kết hợp lại các bit-plane thành các kênh màu
        def combine_bit_planes(bit_planes):
            channel = np.zeros_like(bit_planes[0], dtype=np.uint8)
            for i in range(8):
                channel += (bit_planes[i] << i)
            return channel

        R_new = combine_bit_planes(bit_planes_R)
        G_new = combine_bit_planes(bit_planes_G)
        B_new = combine_bit_planes(bit_planes_B)

        # Ghép lại các kênh màu thành khung hình RGB
        frame_bgr_new = cv2.merge([B_new, G_new, R_new])

        # Chuyển đổi từ RGB sang BGR để ghi bằng OpenCV
        # frame_bgr_new = cv2.cvtColor(frame_rgb_new, cv2.COLOR_RGB2BGR)

        out.write(frame_bgr_new)

        if bit_index >= data_length:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            break

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    os.remove(input_video_path)
    os.rename(temp_video_path, input_video_path)

    print("\nMessage encoded successfully!")
    
def decode_message_from_video(video_path, threshold=0.9, delimiter='11111110'):
    cap_embedded = cv2.VideoCapture(video_path)
    if not cap_embedded.isOpened():
        print("Không thể mở video đã được nhúng. Vui lòng kiểm tra đường dẫn.")
        return

    extracted_bits = []
    extracted_length_bits = 0
    extracted_message = ""

    while True:
        ret, frame = cap_embedded.read()
        if not ret:
            break

        # Chuyển đổi từ BGR (OpenCV) sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phân tách các kênh màu
        B, G, R = cv2.split(frame)

        # Trích xuất bit-plane cho mỗi kênh màu
        bit_planes_R = extract_bit_planes(R)
        bit_planes_G = extract_bit_planes(G)
        bit_planes_B = extract_bit_planes(B)

        # Đánh giá độ phức tạp (entropy) của mỗi bit-plane
        entropies_R = evaluate_bit_planes(bit_planes_R)
        entropies_G = evaluate_bit_planes(bit_planes_G)
        entropies_B = evaluate_bit_planes(bit_planes_B)

        # Chọn các bit-plane phức tạp cao (entropy >= 0.9)
        selected_R = select_complex_bit_planes(entropies_R, threshold)
        selected_G = select_complex_bit_planes(entropies_G, threshold)
        selected_B = select_complex_bit_planes(entropies_B, threshold)

        # Trích xuất dữ liệu từ các bit-plane được chọn
        for plane_idx in selected_B:
            string_bits = array_to_string(extracted_bits)
            if (string_bits.find(delimiter) != -1):
                break
            bits = extract_data_from_bit_plane(
                bit_planes_B[plane_idx]
            )
            extracted_bits.extend(bits)
            extracted_length_bits += len(bits)

        for plane_idx in selected_G:
            string_bits = array_to_string(extracted_bits)
            if (string_bits.find(delimiter) != -1):
                break
            bits = extract_data_from_bit_plane(
                bit_planes_G[plane_idx]
            )
            extracted_bits.extend(bits)
            extracted_length_bits += len(bits)

        for plane_idx in selected_R:
            string_bits = array_to_string(extracted_bits)
            if (string_bits.find(delimiter) != -1):
                break
            bits = extract_data_from_bit_plane(
                bit_planes_R[plane_idx]
            )
            extracted_bits.extend(bits)
            extracted_length_bits += len(bits)

        string_bits = array_to_string(extracted_bits)
        if (string_bits.find(delimiter) != -1):
            break

    extracted_message = array_to_string(extracted_bits)

    print(f"Thông điệp bí mật: {bin_to_text(extracted_message)}")

    return extracted_message

if __name__ == "__main__":
    input_video = 'test_1 copy.mp4'
    secret_message = "This is the hidden message!"

    # print(text_to_bin(secret_message))

    encode_message_in_video(input_video, secret_message)

    decode_message_from_video(input_video)

    # print(array_to_string(arrays))
