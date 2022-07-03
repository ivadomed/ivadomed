import numpy as np


class CroppableArray(np.ndarray):
    """Zero padding slice past end of array in numpy.

    Adapted From: https://stackoverflow.com/a/41155020/13306686
    """

    def __getitem__(self, item):
        all_in_slices = []
        pad = []
        for dim in range(self.ndim):
            # If the slice has no length then it's a single argument.
            # If it's just an integer then we just return, this is
            # needed for the representation to work properly
            # If it's not then create a list containing None-slices
            # for dim>=1 and continue down the loop
            try:
                len(item)
            except TypeError:
                if isinstance(item, int):
                    return super().__getitem__(item)
                newitem = [slice(None)] * self.ndim
                newitem[0] = item
                item = newitem
            # We're out of items, just append noop slices
            if dim >= len(item):
                all_in_slices.append(slice(0, self.shape[dim]))
                pad.append((0, 0))
            # We're dealing with an integer (no padding even if it's
            # out of bounds)
            if isinstance(item[dim], int):
                all_in_slices.append(slice(item[dim], item[dim] + 1))
                pad.append((0, 0))
            # Dealing with a slice, here it get's complicated, we need
            # to correctly deal with None start/stop as well as with
            # out-of-bound values and correct padding
            elif isinstance(item[dim], slice):
                # Placeholders for values
                start, stop = 0, self.shape[dim]
                this_pad = [0, 0]
                if item[dim].start is None:
                    start = 0
                else:
                    if item[dim].start < 0:
                        this_pad[0] = -item[dim].start
                        start = 0
                    else:
                        start = item[dim].start
                if item[dim].stop is None:
                    stop = self.shape[dim]
                else:
                    if item[dim].stop > self.shape[dim]:
                        this_pad[1] = item[dim].stop - self.shape[dim]
                        stop = self.shape[dim]
                    else:
                        stop = item[dim].stop
                all_in_slices.append(slice(start, stop))
                pad.append(tuple(this_pad))

        # Let numpy deal with slicing
        ret = super().__getitem__(tuple(all_in_slices))
        # and padding
        ret = np.pad(ret, tuple(pad), mode='constant', constant_values=0)

        return ret

