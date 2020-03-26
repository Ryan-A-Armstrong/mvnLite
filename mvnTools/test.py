from matplotlib import pyplot as plt

from mvnTools import TifStack as tf

testimage = tf('/home/ryan/Desktop/image_analysis/data/original_sample.tif')

for i in testimage.tif_pages:
    plt.imshow(i)
    plt.show()
