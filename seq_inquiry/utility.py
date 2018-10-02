import matplotlib
matplotlib.use('Agg')
import pylab as plt

import imageio
import cv2
import numpy as np

def set_bound(pos):
    if pos <0:
        return 0
    elif pos>224+40:
        return 224+39
    else:
        return pos

def generate_movies(n_samples=1000):
    np.random.seed(19921010)

    row = 224 + 40
    col = 224 + 40
    # noisy_movies = np.zeros((n_samples, n_frames, row, col, 3), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 3),dtype=np.float)

    for i in range(n_samples):
        # Add 1 to 4 moving squares
        # n = np.random.randint(1, 5)

        if i%100==0:
                print("Generating %d th data." % i )

        # Add 10 to 20 moving squares
        # n = np.random.randint(10, 21)
        n = np.random.randint(4, 11)

        for j in range(n):

            # Initial position
            xstart = np.random.randint(20, 20+224)
            ystart = np.random.randint(20, 20+224)
            # Direction of motion
            # directionx = 0
            directionx = np.random.randint(0, 21) - 10
            # directiony = 0
            directiony = np.random.randint(0, 21) - 10

            # Size of the square
            # w = np.random.randint(2, 4)
            w = np.random.randint(3, 10)
            # w = np.random.randint(30, 59)

            color_r = np.random.randint(100,205) /255.0
            color_g = np.random.randint(100,205) /255.0
            color_b = np.random.randint(100,205) /255.0

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                # noisy_movies[i, t, x_shift - w: x_shift + w,
                             # y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                # if np.random.randint(0, 2):
                #     noise_f = (-1)**np.random.randint(0, 2)
                #     noisy_movies[i, t,
                #                  x_shift - w - 1: x_shift + w + 1,
                #                  y_shift - w - 1: y_shift + w + 1,
                #                  0] += noise_f * 0.1
                
                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)

                shifted_movies[i, t, set_bound(x_shift - w): set_bound(x_shift + w), set_bound(y_shift - w): set_bound(y_shift + w), 0] = color_r
                shifted_movies[i, t, set_bound(x_shift - w): set_bound(x_shift + w), set_bound(y_shift - w): set_bound(y_shift + w), 1] = color_g
                shifted_movies[i, t, set_bound(x_shift - w): set_bound(x_shift + w), set_bound(y_shift - w): set_bound(y_shift + w), 2] = color_b

    # Cut to a 40x40 window
    # noisy_movies = noisy_movies[::, ::, 20:20+224, 20:20+224, ::]
    shifted_movies = shifted_movies[::, ::, 20:20+224, 20:20+224, ::]
    # noisy_movies[noisy_movies >= 1] = 1
    # shifted_movies[shifted_movies >= 1] = 1
    return shifted_movies

def plot_val(which):

    # which = int(N_SAMPLES *0.98)
    track = shifted_movies[which][::, ::, ::, ::]
    track2 = autoencoder.predict(track[np.newaxis, ::, ::, ::, ::])
    track2 = track2[0][::, ::, ::, ::]

    # track2 = autoencoder.predict(track)
    # for j in range(16):
    #     new = new_pos[::, -1, ::, ::, ::]
    #     track = np.concatenate((track, new), axis=0)

    # And then compare the predictions
    # to the ground truth
    # track2 = shifted_movies[which][::, ::, ::, ::]

    for i in range(sequenceLength):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(131)
        # if i >= 7:
        # ax.text(1, 3, 'Ground Truth', fontsize=20, color='w')
        ax.text(2, 4, 'Ground Truth', fontsize=16, color='red')
        # else:
            # ax.text(1, 3, 'Inital trajectory', fontsize=20)
        toplot = track[i, ::, ::, ::]
        plt.imshow(toplot)

        ax = fig.add_subplot(132)
        ax.text(2, 4, 'Recovered', fontsize=16, color='red')
        toplot = track2[i, ::, ::, ::]
        plt.imshow(toplot)

        ax = fig.add_subplot(133)
        ax.text(2, 4, 'Recovered(Rescaled)', fontsize=16, color='red')
        toplot = toplot / np.amax(toplot)
        plt.imshow(toplot)

        plt.savefig('predict/%05i_animate.png' % (i + 1))

    images = []
    STR_PATH = "predict/"
    STR_FILE = ""
    STR_SUFFIX = "_animate.png"

    for i in range(sequenceLength):
        filename = STR_PATH+STR_FILE+ '%05i'%(i+1) +STR_SUFFIX
        im = imageio.imread(filename)
        images.append(im)

    imageio.mimsave('./result'+'_'+setup_name+'_'+str(num_epochs)+'.gif', images)
