"""
Original code from (with minor changes):
https://github.com/ouyangzhibo/Image_Foveation_Python
"""

import cv2
import numpy as np
import sys
import os
import multiprocessing

import time


from tqdm import tqdm
from parameters import RESOLUTION, WEAK_FOVEATION, STRONG_FOVEATION


def genGaussiankernel(width, sigma):
    x = np.arange(-int(width / 2), int(width / 2) + 1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d


def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]

    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)

    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width / 2), int(height / 2)))
        pyramids.append(G)

    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i - 1:
                im_size = (curr_im.shape[1] * 2, curr_im.shape[0] * 2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids


def foveat_img(im, fixs, fov_params):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    
    This function outputs the foveated image with given input image and fixations.
    """
    sigma = 0.248
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape

    # compute coef
    p = fov_params['p']  # blur strength
    k = fov_params['k']  # size of foveation
    alpha = fov_params['alpha']  # also size?

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)

    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i - 3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i - 1] = np.sqrt(np.log(2) / k) / (2 ** (i - 3)) * sigma

    omega[omega > 1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i - 1] - Ts[i] + 1e-5))

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i - 1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    # print('num of full-res pixel', np.sum(Ms[0] == 1))
    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov



"""
Original __main__
"""
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Wrong format: python retina_transform.py [image_path]")
#         exit(-1)

#     im_path = sys.argv[1]
#     im = cv2.imread(im_path)
#     print(im)
#     # im = cv2.resize(im, (512, 320), cv2.INTER_CUBIC)
#     xc, yc = int(im.shape[1]/2), int(im.shape[0]/2)

#     im = foveat_img(im, [(xc, yc)])

#     cv2.imwrite(im_path.split('.')[0]+'_RT.jpg', im)


def generate_foveation_points(size):

    fov_points = []
    indices = []

    # loop over the image row by row
    for x_label, x in enumerate(range(1, 10, 2)):
        for y_label, y in enumerate(range(1, 10, 2)):
            indices.append((x_label, y_label))
            fov_points.append((int(size*(x/10)), int(size*(y/10))))

    # also return indices because those are needed to correctly name the output files
    return indices, fov_points


def read_image(path):
    # image transformation - parameters
    # this is needed because otherwise foveation point outside of cropped area
    resize_size = int(RESOLUTION/0.875)  #256 for input_size 224
    margin = int((resize_size - RESOLUTION)/2)

    read_im = cv2.imread(path)
    resized_im = cv2.resize(read_im, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
    cropped_im = resized_im[margin:-margin, margin:-margin]
    return cropped_im


def f_nofoveation(image_class, read_path, write_path):
    im_folder_path = read_path + '\\' + image_class
    os.mkdir(write_path + '\\' + image_class)
    print("\n foveating images in {}".format(im_folder_path))
    im_paths = os.listdir(im_folder_path)

    for im_path in im_paths:
        im = read_image(im_folder_path + '/' + im_path)

        class_folder_path = write_path + '\\' + image_class
        suffix = '.jpg'
        filename = class_folder_path + '/' + im_path.split('.')[0] + suffix
        cv2.imwrite(filename, im)


def f_selection(image_class, read_path, write_path, fov_params, selection):
    im_folder_path = read_path + '\\' + image_class
    os.mkdir(write_path + '\\' + image_class)
    print("\n foveating images in {}".format(im_folder_path))
    im_paths = os.listdir(im_folder_path)

    generated_fov_points = list(zip(*generate_foveation_points(RESOLUTION)))
    generated_fov_points = [generated_fov_points[i][1] for i in selection]
    print(generated_fov_points)

    for im_path in im_paths:
        im = read_image(im_folder_path + '/' + im_path)
        fov_im = foveat_img(im, generated_fov_points, fov_params)

        # show red arrows between foveation locations
        [cv2.arrowedLine(fov_im,
                         generated_fov_points[i],
                         generated_fov_points[i+1],
                         (255, 0, 0), 3, tipLength=0.5) for i in range(len(generated_fov_points)-1)]
        # show red dots at foveation locations
        [cv2.circle(fov_im, fov_point, 5, (0, 0, 255), -1) for fov_point in generated_fov_points]

        class_folder_path = write_path + '\\' + image_class
        suffix = '.jpg'
        filename = class_folder_path + '/' + im_path.split('.')[0] + suffix
        cv2.imwrite(filename, fov_im)


def f(image_class, read_path, write_path, fov_params):
    im_folder_path = read_path + '\\' + image_class
    os.mkdir(write_path + '\\' + image_class)
    print("\n foveating images in {}".format(im_folder_path))
    im_paths = os.listdir(im_folder_path)

    generated_fov_points = list(zip(*generate_foveation_points(RESOLUTION)))

    for im_path in im_paths:
        im = read_image(im_folder_path + '/' + im_path)
        fovim_folder_path = write_path + '\\' + image_class + '\\' + im_path.split('.')[0]
        os.mkdir(fovim_folder_path)

        for index, fov_point in generated_fov_points:
            fov_im = foveat_img(im, [fov_point], fov_params)

            # adding a red dot so that spotting the foveation point is easier
            # cv2.circle(fov_im, fov_point, 5, (0, 0, 255), -1)
            # TODO make sure the filenames are more robust for PyTorch file loader

            fov_location = str(index)
            suffix = '_RT.jpg'
            filename = fovim_folder_path + '/' + im_path.split('.')[0] + '_' + fov_location + suffix
            cv2.imwrite(filename, fov_im)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Wrong format: python retina_transform.py [image_class/image_path]")
    #     exit(-1)

    # read_path = sys.argv[1]
    # read_path = 'E:\ILSVRC2017\second_subdataset\strongfoveation'
    read_path = 'E:\\ILSVRC2017\\10classessecond\\original'

    # write_path = read_path + '\\' + 'nofoveation'
    write_path = read_path + '\\' + 'foveated'

    im_classes = os.listdir(read_path)
    os.mkdir(write_path)
    print('\n created folder {}'.format(write_path))

    processes = []
    for im_class in im_classes:
        # p = multiprocessing.Process(target=f_selection, args=(im_class, read_path, write_path, STRONG_FOVEATION, [12, 17, 16],))
        p = multiprocessing.Process(target=f, args=(im_class, read_path, write_path, WEAK_FOVEATION, ))
        # p = multiprocessing.Process(target=f, args=(im_class, read_path, write_path, STRONG_FOVEATION, ))
        # p = multiprocessing.Process(target=f_nofoveation, args=(im_class, read_path, write_path,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

"""
Second main
"""

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Wrong format: python retina_transform.py [image_path]")
#         exit(-1)
#
#     folder_path = sys.argv[1]
#     im_paths = os.listdir(folder_path)
#
#     # image transformation - parameters
#     # this is needed because otherwise foveation point outside of cropped area
#     input_size = 224
#     resize_size = int(input_size / 0.875)  # 256 for input_size 224
#     margin = int((resize_size - input_size) / 2)
#
#     # fovim_folder_path = 'E:\\ILSVRC2017\\newimages\\notfoveated\\n01531178'
#     fovim_folder_path = 'E:\\ILSVRC2017\\multiplefoveationsTEST\\result'
#     # os.mkdir(fovim_folder_path)
#
#     for im_path in im_paths:
#         start = time.time()
#         im = cv2.imread(folder_path + '/' + im_path)
#
#         # image transformation - application
#         resized_im = cv2.resize(im, (256, 256))
#         cropped_im = resized_im[margin:-margin, margin:-margin]
#
#         # xc, yc = int(cropped_im.shape[1]/2), int(cropped_im.shape[0]/2)
#         # foveated_im = foveat_img(cropped_im, [(xc, yc)])
#         # cv2.imwrite(fovim_folder_path + '/' + im_path.split('.')[0] + '.jpg', cropped_im)
#
#         xc1, yc1 = (int(cropped_im.shape[1] * (5 / 10)), int(cropped_im.shape[0] * (5 / 10)))
#         xc2, yc2 = (int(cropped_im.shape[1] * (7 / 10)), int(cropped_im.shape[0] * (5 / 10)))
#
#         foveated_im = foveat_img(cropped_im, [(xc1, yc1)])
#         # cv2.circle(foveated_im, (xc1, yc1), 5, (0, 0, 255) , -1)
#         # cv2.imwrite(fovim_folder_path + '/' + im_path.split('.')[0] + '_1.jpg', foveated_im)
#
#         # foveated_im = foveat_img(cropped_im, [(xc1, yc1), (xc2, yc2)])
#         # cv2.circle(foveated_im, (xc1, yc1), 5, (0, 0, 255) , -1)
#         # cv2.circle(foveated_im, (xc2, yc2), 5, (0, 0, 255) , -1)
#         # cv2.imwrite(fovim_folder_path + '/' + im_path.split('.')[0] + '_2.jpg', foveated_im)
#
#         # cv2.imwrite(fovim_folder_path + '/' + im_path.split('.')[0] + '_RT.jpg', foveated_im)
#
#         end = time.time()
#         print("done with: " + im_path + " in " + str(end - start) + " seconds.")
