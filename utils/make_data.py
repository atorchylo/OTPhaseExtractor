from PIL import Image, ImageOps
import numpy as np

def make_gauss(r, sigma):
    return np.exp(-(r**2)/(sigma**2))

def make_donut(r, R, sigma):
    return (1 + (r**2 - R**2)**2 / sigma**4)**(-1)

def make_disk(r, R, sigma):
    return (np.abs(r - R) < sigma)

def get_fly_test_image(fpath, N):
    img = Image.open(fpath)
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.resize((N//2, N//2))
    padded_img = Image.new(img.mode, (N, N), 0)
    padded_img.paste(img, (N//4, N//4))
    return np.array(padded_img) / 255

def get_stanford_image(fpath, N):
    img = Image.open(fpath)
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.resize((N//2, N//2))
    padded_img = Image.new(img.mode, (N, N), 0)
    padded_img.paste(img, (N//4, N//4))
    return np.array(padded_img) / 255

def get_hogan_image(fpath, N):
    img = Image.open(fpath)
    img = ImageOps.grayscale(img)
    #img = ImageOps.invert(img)
    img = img.resize((N//2, N//2))
    padded_img = Image.new(img.mode, (N, N), 0)
    padded_img.paste(img, (N//4, N//4))
    return np.array(padded_img) / 255

