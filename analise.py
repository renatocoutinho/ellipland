from landscape import *

def interpolate_nodata(land, nodata, matrix=None, radius=1):
    from collections import Counter

    nd = where(land==nodata)
    newland = land.copy()
    for i, j in zip(*nd):
        c = Counter(land[max(i-radius, 0):i+radius+1, max(j-radius, 0):j+radius+1].flatten()).most_common()
        if c[0][0] != nodata:
            newland[i,j] = c[0][0]
        elif len(c) > 1:
            newland[i,j] = c[1][0]
        else:
            if matrix is None:
                raise ValueError('No valid data around point (%d,%d)' % (i, j))
            else:
                newland[i,j] = matrix
    return newland

def find_borders(land, nodata):
    rowmin = colmin = 0
    rowmax, colmax = land.shape
    for i in range(land.shape[0]):
        if all(land[i,:] == nodata):
            rowmin = i+1
        else:
            break
    for i in range(land.shape[0]-1, 0, -1):
        if all(land[i,:] == nodata):
            rowmax = i
        else:
            break
    for i in range(land.shape[1]):
        if all(land[:,i] == nodata):
            colmin = i+1
        else:
            break
    for i in range(land.shape[1]-1, 0, -1):
        if all(land[:,i] == nodata):
            colmax = i
        else:
            break

    return slice(rowmin, rowmax), slice(colmin, colmax)

def rotate_landscape(land, angle, nodata):
    from scipy.ndimage.interpolation import rotate
    return rotate(land, angle, order=0, cval=nodata)

def find_rotation(land, nodata):
    from scipy.ndimage.interpolation import rotate
    a = where(land[0,:] != nodata)[0][0]
    b = where(land[:,0] != nodata)[0][0]
    print a, b
    return -rad2deg(arctan2(b, a))

def find_best_pos(land, nodata):
    s = find_borders(land, nodata)
    l1 = land[s[0], s[1]]
    angle = find_rotation(l1, nodata)
    l2 = rotate_landscape(l1, angle, nodata)
    s2 = find_borders(l2, nodata)
    return s, angle, s2

def apply_best_pos(land, nodata, s1, angle, s2):
    return rotate_landscape(land[s1[0], s1[1]], angle, nodata)[s2[0], s2[1]]

def treat_raster(infile, outfile, nodata):
    l = loadtxt(infile, skiprows=6, dtype=int)
    pos = find_best_pos(l, nodata)
    lb = apply_best_pos(l, nodata, *pos)
    lc = interpolate_nodata(lb, -9999, matrix=0)
    savetxt(outfile, lc, '%d')


