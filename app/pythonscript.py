import numpy as np
import cv2
import sys
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
    energy_map = convolved.sum(axis=2)

    return energy_map

def construct_dp(energy_map, img):
	r, c, _ = img.shape
	dp = energy_map.copy()
	backtrack = np.zeros_like(dp, dtype=np.int)
	for i in range(1, r):
		for j in range(0, c):
			if j == 0:
				idx = np.argmin(dp[i-1, j:j+2])
				backtrack[i, j] = idx + j
				min_energy = dp[i-1, idx+j]
			else:
				idx = np.argmin(dp[i-1, j-1:j+2])
				backtrack[i, j] = idx+j-1
				min_energy = dp[i-1, idx+j-1]
			dp[i, j] += min_energy
	return dp, backtrack


def get_mask(dp, backtrack, img):
	r, c, _ = img.shape
	mask = np.ones((r, c), dtype=np.bool)
	j = np.argmin(dp[-1])
	for i in reversed(range(r)):
		mask[i, j] = False
		j = backtrack[i, j]
	return mask

def get_mask_2(dp, backtrack, img):
	r, c, _ = img.shape
	mask = np.ones((r, c), dtype=np.bool)
	j = np.argmin(dp[-1])
	seam = np.zeros((r,), dtype=np.uint32)
	for i in reversed(range(r)):
		seam[i] = j
		mask[i, j] = False
		j = backtrack[i, j]
	return mask, seam

def remove_shortest_path(img, mask):
	r, c, _ = img.shape
	mask = np.stack([mask]*3, axis=2)
	img = img[mask].reshape((r, c-1, 3))
	return img

def remove(img):
	energy_map = calc_energy(img)
	dp, backtrack = construct_dp(energy_map, img)
	mask = get_mask(dp, backtrack, img)
	img = remove_shortest_path(img, mask)
	return img

def column_remover(img, count):
	for i in trange(count):
		img = remove(img)
	return img

def row_remover(img, count):
	img = np.rot90(img, 1, (0, 1))
	img = column_remover(img, count)
	img = np.rot90(img, 3, (0, 1))
	return img;

def remove_seam(img):
	energy_map = calc_energy(img)
	dp, backtrack = construct_dp(energy_map, img)
	mask, seam = get_mask_2(dp, backtrack, img)
	img = remove_shortest_path(img, mask)
	return img, seam

def add_seam(img, seam):
	r, c, _ = img.shape
	output_image = np.zeros((r, c+1, 3))
	for row in range(r):
		col = seam[row]
		for ch in range(3):
			if col == 0:
				avg = np.average(img[row, col:min(col+2, c-1), ch])
				output_image[row, col, ch] = img[row, col, ch]
				output_image[row, col+1, ch] = avg
				output_image[row, col+1:, ch] = img[row, col:, ch]
			else:
				avg = np.average(img[row, col-1:min(c-1, col+1), ch])
				output_image[row, :col, ch] = img[row, :col, ch]
				output_image[row, col, ch] = avg
				output_image[row, col+1:, ch] = img[row, col:, ch]
	return output_image

def column_adder(img, count):
	temp = img.copy()
	seams = []
	for i in trange(count):
		temp, seam = remove_seam(temp)
		seams.append(seam)
	for seam in seams:
		print(seam)
	for seam in seams:
		img = add_seam(img, seam)
	return img

def row_adder(img, count):
	img = np.rot90(img, 1, (0, 1))
	img = column_adder(img, count)
	img = np.rot90(img, 3, (0, 1))
	return img

def save(img):
	imwrite("static/output.jpg", img.astype(np.uint8))

def widthModifier(width):
    img = imread("static/scream.jpg")
    r, c, _ = img.shape
    if width > c:
        print("Scaling up")
        img = column_adder(img, width-c)
    else:
        print("Scaling down")
        img = column_remover(img, c-width)
    
    save(img)

def  heightModifier(height):
	img = imread("static/scream.jpg")
	r, c, _ = img.shape
	if height > r:
		print("Scaling up")
		img = row_adder(img, height-r)
	else:
		print("Scaling down")
		img = row_remover(img, r-height)
	save(img)