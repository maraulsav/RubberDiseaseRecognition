import Augmentor
p = Augmentor.Pipeline(r"/Users/maritzaauliasavitri/Downloads/Rubber_Recognition/Train/Pestalotiopsis")
p.zoom(probability=0.4,min_factor=0.3,max_factor=1.2)
p.random_brightness(probability=0.5, min_factor=0.3, max_factor=1.2)
p.random_distortion(probability=0.7, grid_width=4, grid_height=4, magnitude=8)
p.flip_top_bottom(probability=0.5)
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.sample(8000)
