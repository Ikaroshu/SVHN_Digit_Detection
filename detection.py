from tensorflow.keras.models import Model, load_model
import cv2
import numpy as np


model = load_model('./data/my_model.h5')


# detect and mark numbers on the image
def detect_and_mark(image):
    h = image.shape[0]
    w = image.shape[1]
    window_size = int(min(h / 2, w / 2))
    cropped_image = image.copy()
    cropped_h = 0
    cropped_w = 0
    while window_size >= 32:
        find = False
        window_width = int(window_size)
        window_height = window_size
        step_width = int(window_width / 2)
        step_hight = int(window_height / 2)
        corner_h, corner_w = np.meshgrid(np.arange(0, h - window_height, step_hight, dtype=np.int),
                                         np.arange(0, w - window_width, step_width, dtype=np.int))
        for i in range(corner_h.shape[0]):
            if find:
                break
            for j in range(corner_h.shape[1]):
                hh = corner_h[i, j]
                ww = corner_w[i, j]
                prediction = \
                model.predict(np.array([cv2.resize(image[hh:hh + window_height, ww:ww + window_width], (32, 32))]))[0]
                num = np.argmax(prediction)
                if num < 10 and prediction[num] > 0.99:
                    find = True
                    cropped_image = image[max(0, hh - int(window_height / 2)):min(h, hh + int(window_height * 3 / 2)),
                                    max(0, ww - 2 * window_width):min(w, ww + 3 * window_width)]
                    cropped_h = max(0, hh - int(window_height / 2))
                    cropped_w = max(0, ww - 2 * window_width)
                    break
        if find:
            break
        window_size = int(window_size * 2 / 3)
    h = cropped_image.shape[0]
    w = cropped_image.shape[1]
    window_size = int(min(h, w))
    output_image = image.copy()
    min_window = window_size * (0.75 ** 4)
    numbers = []
    while window_size >= min_window:
        window_width = int(window_size * 2 / 3)
        window_height = window_size
        step_width = int(window_width / 4)
        step_hight = int(window_height / 4)
        corner_h, corner_w = np.meshgrid(np.arange(0, h - window_height, step_hight, dtype=np.int),
                                         np.arange(0, w - window_width, step_width, dtype=np.int))
        for i in range(corner_h.shape[0]):
            for j in range(corner_h.shape[1]):
                hh = corner_h[i, j]
                ww = corner_w[i, j]
                prediction = model.predict(
                    np.array([cv2.resize(cropped_image[hh:hh + window_height, ww:ww + window_width], (32, 32))]))[0]
                num = np.argmax(prediction)

                if num < 10 and prediction[num] > 0.85:
                    inserted = False
                    for pt in numbers:
                        if pt[0] == num and abs(pt[1] / pt[5] - hh) < pt[3] / pt[5] / 3 * 2 and abs(
                                pt[2] / pt[5] - ww) < pt[4] / pt[5] / 3 * 2:
                            inserted = True
                            pt[1] += hh
                            pt[2] += ww
                            pt[3] += window_height
                            pt[4] += window_width
                            pt[5] += 1
                            break
                    if not inserted:
                        numbers.append([num, hh, ww, window_height, window_width, 1])
        window_size = int(window_size * 3 / 4)
    for pt in numbers:
        cv2.rectangle(output_image, (cropped_w + int(pt[2] / pt[5]), cropped_h + int(pt[1] / pt[5])),
                      (cropped_w + int(pt[2] / pt[5] + pt[4] / pt[5]), cropped_h + int(pt[1] / pt[5] + pt[3] / pt[5])),
                      (255, 0, 0), 5)
        cv2.putText(output_image, '{}'.format(pt[0]), (cropped_w + int(pt[2] / pt[5]), cropped_h + int(pt[1] / pt[5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    return output_image


# generate frames in the video, copied from ps3 code
def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None


# write frame to video, copied from ps3 code
def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)



