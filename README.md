# icbs - Image Cut and (re)Build Squares
Package to cut an image in overlapping rectangles and to rebuild it.

## Description
It is useful to cut an image like this one:
![](/img/pre_cut.png) \
To obtain an array of images:
![](/img/after_cut.png)

It is useful because it handles well edge cases; from this image:
![](/img/edge_case.png) \
To obtain an array of images:
![](/img/after_cut_with_edge.png) \
With icbs you can also do the inverse operation, and given the images, rebuild the original one.

## Usage
icbs has two methods, cut and rebuild. The parameters are similar for both images and are better explained through an image: \
![](/img/pre_cut_labled.png) \
The method  `cut` returns the n_img_per_col and the array with the images. \
n_img_per_col is particularly important for using the `rebuild` method. \
You can find more info reading the parameter description under both methods.
