import numpy as np
import openslide
import PIL

def _get_downsample(slide, magnification):
    mag = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] 
    if mag not in ["20", "40"]:
        raise ValueError("Found Image with magnification of {}.".format(mag))
    ds = int(int(mag) / magnification)
    if ds not in [1, 2]:
        raise ValueError("Need downsample of {}.".format(mag))
    return ds

def get_dimensions_at_mag(slide, magnification):
    ds = _get_downsample(slide, magnification)
    return [d // ds for d in slide.dimensions]

def read_region_at_mag(slide, location, magnification, size, downsample=1):
    def round_with_tol(x, eps=0.10):
        r = round(x)
        if abs(x - r) > eps:
            raise ValueError("Attempting to round {}.".format(x))
        return r
    ds = _get_downsample(slide, magnification)
    eps = 0.001
    scale = [(downsample * ds / round_with_tol(d)) + eps for d in slide.level_downsamples]
    level = None
    for (i, s) in enumerate(scale):
        if s >= 1:
            level = i
    # valid = False
    # while level >= 0 and not valid:
    #     print(level, flush=True)
    #     # The downsampled images in the svs file are sometimes corrupted.
    #     # This can be identified by looking at the alpha channel.
    #     # Normal images are all 255; corrupted images have other values.
    #     # This appears to only happen when a subsection of the image is loaded.
    #     print([i * ds for i in location])
    #     print(slide.level_dimensions)
    #     print([ds * w // round_with_tol(slide.level_downsamples[level]) for w in size])
    #     X = slide.read_region([i * ds for i in location], level, [ds * w // round_with_tol(slide.level_downsamples[level]) for w in size])
    #     print(np.array(X)[:, :, 3])
    #     print(np.sum(np.array(X)[:, :, 3] != 255))
    #     print(np.sum(np.array(X)[:, :, 3] == 255))
    #     valid = (np.array(X)[:, :, 3] == 255).all()
    #     X = X.convert("RGB")
    #     X = X.resize([s // downsample for s in size], PIL.Image.ANTIALIAS)
    #     level -= 1
    X = slide.read_region([i * ds for i in location], level, [ds * w // round_with_tol(slide.level_downsamples[level]) for w in size])
    X = X.convert("RGB")
    X = X.resize([s // downsample for s in size], PIL.Image.ANTIALIAS)
    return X
