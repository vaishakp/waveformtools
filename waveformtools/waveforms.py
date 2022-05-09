##########################################################################
# A Class for handling the waveform data
##########################################################################


class Psi:
    """ A class for handling waveforms."""

    def __init__(self, timeaxis=None, wavedata=None, base_dir=None, data_dir=None, filename=None):

        self.base_dir = base_dir or ""
        self.data_dir = data_dir or ""
        self.filename = filename or ""
        self.timeaxis = timeaxis or []
        self.wavedata = wavedata or []

    def load_data(self):
        full_path = self.base_dir + self.data_dir + self.filename

        with h5py.File(full_path, "r") as f:
            # List all groups
            keys = list(f.keys())

            index = 0
            token = -1

            while token < 0 and index < len(keys):
                key = keys[index]
                token = key.find("Psi")
                # message(key)
                index += 1

            if token < 0:
                message("Waveform dataset not found")
            else:
                message(key)

            # Get the data
            data = np.array(f[key])

            self.timeaxis = data[0]
            self.wavedata = data[1]

    @property
    def dt(self):
        return self.timeaxis[1] - self.timeaxis[0]

    # @base_dir.setter
    # def base_dir(self, base_dir):
    # 	 self.__base_dir = base_dir
