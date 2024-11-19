import depth

img_path = "color_image.jpg"

depth_estimator=depth.estimate_depth(img_path) #深度视觉估算
print(depth_estimator)