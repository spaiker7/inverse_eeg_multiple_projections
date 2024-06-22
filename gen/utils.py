import numpy as np

def norm_data(data):
	norm_data = (data-np.amin(data))/(np.amax(data)-np.amin(data))
	return norm_data

def crop_img_by_zeros(img, by_img=None):
	"""
	Crop image by zeros columns and rows coords of itself or by the
	provided image (by_img)
	"""
	if by_img is not None:
		zeros_columns = np.argwhere(np.all(by_img[..., :] == 0, axis=0))
		zeros_rows = np.argwhere(np.all(by_img[..., :] == 0, axis=1))
	else:
		zeros_columns = np.argwhere(np.all(img[..., :] == 0, axis=0))
		zeros_rows = np.argwhere(np.all(img[..., :] == 0, axis=1))

	img = np.delete(img, zeros_columns, axis=1)
	img = np.delete(img, zeros_rows, axis=0)

	return img