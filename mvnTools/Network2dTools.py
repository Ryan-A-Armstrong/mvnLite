

def analyze_directionality(img_dir):
    left_diag = img_dir[img_dir == 1 or img_dir == 9]
    img_dir[img_dir == 8] = 2
    img_dir[img_dir == 7] = 3
    img_dir[img_dir == 6] = 4

