import warnings
from skimage import measure
from skimage.transform import resize
from scipy.stats import wasserstein_distance
import numpy as np
import cv2
import imageio

# specify resized image sizes
height = 2**10
width = 2**10

def get_img(path, norm_size=True, norm_exposure=False):
  img = imageio.imread(path).astype(int)
  if norm_size:
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
  if norm_exposure:
    img = normalize_exposure(img)
  return img

# def get_histogram(img):
#   h, w = img.shape
#   hist = [0.0] * 256
#   for i in range(h):
#     for j in range(w):
#       hist[img[i, j]] += 1
#   return np.array(hist) / (h * w)

# def normalize_exposure(img):
#   img = img.astype(int)
#   hist = get_histogram(img)
#   cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
#   sk = np.uint8(255 * cdf)
#   [height, width] = img.shape
#   normalized = np.zeros_like(img)
#   for i in range(0, height):
#     for j in range(0, width):
#       normalized[i, j] = sk[img[i, j]]
#   return normalized.astype(int)


# def earth_movers_distance(path_a, path_b):
#   img_a = get_img(path_a, norm_exposure=True)
#   img_b = get_img(path_b, norm_exposure=True)
#   hist_a = get_histogram(img_a)
#   hist_b = get_histogram(img_b)
#   return wasserstein_distance(hist_a, hist_b)


def structural_sim(path_a, path_b):
  img_a = get_img(path_a)
  img_b = get_img(path_b)
  sim, diff = measure.compare_ssim(img_a, img_b, full=True, multichannel=True)
  return diff

#
# def pixel_sim(path_a, path_b):
#   img_a = get_img(path_a, norm_exposure=True)
#   img_b = get_img(path_b, norm_exposure=True)
#   return np.sum(np.absolute(img_a - img_b)) / (height*width) / 255
#

# def sift_sim(path_a, path_b):
#   orb = cv2.ORB_create()
#
#   img_a = cv2.imread(path_a)
#   img_b = cv2.imread(path_b)
#
#   kp_a, desc_a = orb.detectAndCompute(img_a, None)
#   kp_b, desc_b = orb.detectAndCompute(img_b, None)
#
#   bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
#   matches = bf.match(desc_a, desc_b)
#   similar_regions = [i for i in matches if i.distance < 70]
#   if len(matches) == 0:
#     return 0
#     return len(similar_regions) / len(matches)


if __name__ == '__main__':
  img_a = 'input1.jpg'
  img_b = 'test3.jpg'


  structural_sim = structural_sim(img_a, img_b)
  # pixel_sim = pixel_sim(img_a, img_b)
  # sift_sim = sift_sim(img_a, img_b)
  # emd = earth_movers_distance(img_a, img_b)
  print(structural_sim*100)