import numpy as np
from PIL import Image


def merge():
    path = '/Users/chris/Documents/code/python/rookie/pygame/src/img_ground.png'

    list_im = [path, path, path]
    imgs = [Image.open(i) for i in list_im]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

    # save that beautiful picture
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save('Trifecta.png')

    # for a vertical stacking it is simple: use vstack
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save('Trifecta_vertical.png')


size = [28 * 28, 40, 10]
w = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]
print()
