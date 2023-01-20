import matplotlib.pyplot as plt

image1 = plt.imread("data/rain100L/test/rain/norain-1.png")
image2 = plt.imread("data/rain100L/test/norain/norain-1.png")
image3 = plt.imread("result/rain100L/norain-1.png")

fig, axs = plt.subplots(1, 3)

axs[0].imshow(image2)
axs[0].set_title('Original Image')

axs[1].imshow(image1)
axs[1].set_title('Image with Added Rain')

axs[2].imshow(image3)
axs[2].set_title('Derained Image')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()