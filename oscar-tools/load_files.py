import struct

# OSCAR -- Session::LoadEvents(QString filename)
def load_file(filename):
    databytes = None
    with open(filename, mode='rb') as file: # b is important -> binary
        data = file.read()

        # Header
        (magicnum, version, typ, machid, sessid, s_first, s_last) = struct.unpack('IHHIIqq', data[:32])

        assert magicnum == 3341948587
        assert version == 10
        assert typ == 1
        assert machid == 2327
        assert sessid == 1664241600
        assert s_first == 1664241616000
        assert s_last == 1664259258000

        if version >= 10:
            (compmethod, machtype, datasize, crc16) = struct.unpack('HHIH', data[32:42])

            assert compmethod==0
            assert machtype==1
            assert datasize==1964796
            assert crc16==0
        else:
            print('VERSION NOT SUPPORTED')

        temp = data[42:]

        if version >= 10:
            if compmethod > 0:
                print('COMPRESSION NOT SUPPORTED')
            else:
                databytes = temp
        else:
            print('VERSION NOT SUPPORTED')

        # NB of channels
        (mcsize, ) = struct.unpack('h', databytes[:2])

        assert mcsize == 18

        #for c in range(mcsize):
        (code, ) = struct.unpack('I', databytes[2:6])
        assert code == 4354

        (size2, ) = struct.unpack('h', databytes[6:8])
        assert size2 == 1





load_file('../data/63324fc0.001')