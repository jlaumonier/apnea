import struct

# OSCAR -- Session::LoadEvents(QString filename)
def load_file(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        data = file.read()

        (magicnum,) = struct.unpack('I', data[:4])
        (version,) = struct.unpack('H', data[4:6])
        # header >> type; // File type(quint16)
        # header >> machid; // Device ID(quint32)
        # header >> sessid; // (quint32)
        # header >> s_first; // (qint64)
        # header >> s_last; // (qint64)

        assert magicnum == 3341948587
        assert version == 10


load_file('../data/63324fc0.001')