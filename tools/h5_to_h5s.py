import os
import h5py
import numpy as np

def binary_search_array(array, x, left=None, right=None, side="left"):
    """
    Binary search through a sorted array.
    """

    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1)

    return binary_search_array(array, x, left=mid + 1, right=right)

def package_image(dst_file, image, timestamp, img_idx):
        image_dset = dst_file.create_dataset("images/image{:09d}".format(img_idx),
                data=image, dtype=np.dtype(np.uint8))
        image_dset.attrs['size'] = image.shape
        image_dset.attrs['timestamp'] = timestamp
        image_dset.attrs['type'] = "greyscale" if image.shape[-1] == 1 or len(image.shape) == 2 else "color_bgr" 

        ts = dst_file["events/ts"][()]
        event_idx = np.searchsorted(ts, timestamp)
        event_idx = max(0, event_idx-1)
        image_dset.attrs['event_idx'] = event_idx

def add_metadata(dst_file, num_pos, num_neg, t0, tk, num_imgs, sensor_size):
    dst_file.attrs['num_events'] = num_pos+num_neg
    dst_file.attrs['num_pos'] = num_pos
    dst_file.attrs['num_neg'] = num_neg
    dst_file.attrs['duration'] = tk-t0
    dst_file.attrs['t0'] = t0
    dst_file.attrs['tk'] = tk
    dst_file.attrs['num_imgs'] = num_imgs
    dst_file.attrs['sensor_resolution'] = sensor_size

class Frames:
    """
    Utility class for reading the APS frames encoded in the HDF5 files.
    """

    def __init__(self):
        self.ts = []
        self.names = []
        self.num_imgs = 0

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]
            self.ts += [h5obj.attrs["timestamp"]]

    def set_frames(self, src_file, dst_file, t0, t1, crop=None, res=None):

        idx0 = binary_search_array(self.ts, t0)
        idx1 = binary_search_array(self.ts, t1)
        self.num_imgs = idx1 - idx0
        for i in range(idx1 - idx0):
            img = src_file["images"]["image{:09d}".format(i+idx0)]
            if crop is not None and res is not None:
                cropped_img = img[crop[0] : crop[0] + res[0], crop[1] : crop[1] + res[1]]
            else:
                cropped_img = img
            package_image(dst_file, cropped_img, img.attrs['timestamp'], i)

def append_to_dataset(dataset, data):
    dataset.resize(dataset.shape[0] + len(data), axis=0)
    if len(data) == 0:
        return
    dataset[-len(data):] = data[:]

def create_dset(dst_file):
    event_xs = dst_file.create_dataset("events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
    event_ys = dst_file.create_dataset("events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
    event_ts = dst_file.create_dataset("events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
    event_ps = dst_file.create_dataset("events/ps", (0, ), dtype=np.dtype(np.bool_), maxshape=(None, ), chunks=True)
    image_dset = dst_file.create_group("images")
    return event_xs, event_ys, event_ts, event_ps, image_dset

dataset_path = 'dataset'
h5_name = 'indoor_forward_3_davis_with_gt_0'
file_path = os.path.join(dataset_path, h5_name + '.h5')
save_path = os.path.join(dataset_path, 'datasets/')
if not os.path.exists(save_path):
     os.makedirs(save_path)

seq_len = 2.0
count = 0
with h5py.File(file_path, "r") as f:

    image_data = f['images']
    event_data = f['events']
    num_events = len(event_data['ts'])
    img_height, img_width = f.attrs['sensor_resolution']
    dst_height, dst_width = 64, 64

    frames = Frames()
    image_data.visititems(frames)

    ts = event_data['ts']
    t0 = ts[0]
    t_last = ts[-1]

    t0_tmp = t0
    idx0_tmp = 0
    g = h5py.File(save_path + h5_name + str(count) + ".h5", "w")
    event_xs, event_ys, event_ts, event_ps, image_dset = create_dset(g)
    for i in range(int((t_last - t0)/seq_len + 1)):
        crop_x = np.random.randint(0, img_height - dst_height)
        crop_y = np.random.randint(0, img_width - dst_width)

        t1_tmp = t0_tmp + seq_len
        idx1_tmp = np.searchsorted(ts, t1_tmp)
        xs_tmp = event_data['xs'][idx0_tmp:idx1_tmp]
        ys_tmp = event_data['ys'][idx0_tmp:idx1_tmp]
        ts_tmp = event_data['ts'][idx0_tmp:idx1_tmp]
        ps_tmp = event_data['ps'][idx0_tmp:idx1_tmp]
        xs_tmp -= crop_x
        ys_tmp -= crop_y
        mask_x = (xs_tmp > 0) * (xs_tmp < dst_height)
        mask_y = (ys_tmp > 0) * (ys_tmp < dst_width)
        mask = mask_x * mask_y
        xs_tmp = xs_tmp[mask]
        ys_tmp = ys_tmp[mask]
        ts_tmp = ts_tmp[mask]
        ps_tmp = ps_tmp[mask]
        append_to_dataset(event_xs, xs_tmp)
        append_to_dataset(event_ys, ys_tmp)
        append_to_dataset(event_ts, ts_tmp)
        append_to_dataset(event_ps, ps_tmp)
        frames.set_frames(f, g, t0_tmp, t1_tmp, (crop_x,crop_y), (dst_height, dst_width))

        num_pos = np.count_nonzero(ps_tmp)
        num_neg = len(ps_tmp) - num_pos
        add_metadata(g, num_pos, num_neg, ts_tmp[0], ts_tmp[-1], frames.num_imgs, (dst_height, dst_width))

        count += 1
        t0_tmp = t1_tmp
        idx0_tmp = idx1_tmp

        g.close()
        if idx1_tmp < num_events - 1:
            g = h5py.File(save_path + h5_name + str(count) + ".h5", "w")
            event_xs, event_ys, event_ts, event_ps, image_dset = create_dset(g)
