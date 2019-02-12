from os import listdir

from detection import *


def mark_video(in_path, out_path, fps):
    frame_gen = video_frame_generator(in_path)
    frame = frame_gen.__next__()
    h, w, d = frame.shape
    video_out = mp4_video_writer(out_path, (w, h), fps)
    while frame is not None:
        frame = frame[:, :, ::-1]
        marked = detect_and_mark(frame)
        video_out.write(marked[:, :, ::-1])
        frame = frame_gen.__next__()
    video_out.release()


if __name__ == '__main__':
    for f in listdir('./input_images'):
        img = cv2.imread('./input_images/' + f)[:, :, ::-1]
        out = detect_and_mark(img)[:, :, ::-1]
        cv2.imwrite('./graded_images/'+f[1]+'.png', out)
