import Augmentor
p = Augmentor.Pipeline(r"/Users/maritzaauliasavitri/Downloads/Rubber_Recognition/Train/Corynespora")
p.zoom(probability=0.4,min_factor=0.3,max_factor=1.2)
p.random_brightness(probability=0.6, min_factor=0.3, max_factor=1.2)
p.random_distortion(probability=0.8, grid_width=4, grid_height=4, magnitude=8)
p.flip_top_bottom(probability=0.6)
p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
p.sample(8000)
