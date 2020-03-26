import numpy as np
from PIL import Image
from skimage import img_as_uint


class TifStack:
    path_to_tif = ''
    tif_pages = None
    flattened = None

    def __init__(self, _path_to_tif, page_list=True, flat=True):
        self.path_to_tif = _path_to_tif
        if flat:
            self.tif_pages = self.set_pages()
            self.flattened = self.set_flat()
        elif page_list:
            self.tif_pages = self.set_pages()

    def set_pages(self):
        im = Image.open(self.path_to_tif)
        im.convert('L')  # Converts to black and white
        pages = []

        i = 0
        while True:
            try:
                im.seek(i)
                image = np.array(im)
            except EOFError:
                break
            pages.append(image)
            i += 1

        return np.asarray(pages)

    def get_pages(self):
        if self.tif_pages is None:
            self.set_pages()
        return self.tif_pages

    def set_flat(self):
        if self.tif_pages is None:
            self.set_pages()
        flat_sum = np.sum(self.tif_pages, axis=0)

        return img_as_uint(flat_sum)

    def get_flat(self):
        if self.flattened is None:
            self.set_flat()
        return self.flattened
