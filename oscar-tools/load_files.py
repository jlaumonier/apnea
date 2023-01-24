# Test loading with struct python module

import struct


# OSCAR -- Session::LoadEvents(QString filename)
def load_session(filename):
    with open(filename, mode='rb') as file:  # b is important -> binary
        data = file.read()
        position = 0
        read_session(data, position)


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def unpack(buffer, format, position):
    print('from', position, 'to', position + struct.calcsize(format), 'size', struct.calcsize(format))
    return position + struct.calcsize(format), struct.unpack_from(format, buffer, offset=position)


def read_header(buffer, position):
    position, (magicnum, version, typ, machid, sessid, s_first, s_last) = unpack(buffer, 'IHHIIqq', position)

    assert magicnum == 3341948587
    assert version == 10
    assert typ == 1
    assert machid == 2327
    assert sessid == 1664241600
    assert s_first == 1664241616000
    assert s_last == 1664259258000

    return position, magicnum, version, typ, machid, sessid, s_first, s_last


def read_header_v10p(buffer, position, version):
    compmethod = 0
    machtype = 0
    datasize = 0,
    crc16 = 0
    if version >= 10:
        position, (compmethod, machtype, datasize, crc16) = unpack(buffer, 'HHIH', position)

        assert compmethod == 0
        assert machtype == 1
        assert datasize == 1964796
        assert crc16 == 0
    else:
        print('VERSION NOT SUPPORTED')

    return position, compmethod, machtype, datasize, crc16


def read_event(buffer, data_position):
    print(data_position)
    data_position, (ts1, ts2, evcount, t8, rate, gain, offset, mn, mx, dim, second_field) = unpack(buffer,
                                                                                                   'qqiBfffffs?',
                                                                                                   data_position)


    # data_position, (ts1, ts2, evcount, t8) = unpack(buffer,
    #                                                 'qqiB',
    #                                                 data_position)
    # print(buffer[data_position:data_position + 4])
    # print(data_position)
    # data_position, (rate, gain, offset, mn, mx, dim, second_field) = unpack(buffer,
    #                                                                         'fffffs?', data_position)
    print(data_position)
    print(rate)
    return data_position, ts1, ts2, evcount, t8, rate, gain, offset, mn, mx, dim, second_field


def read_session(buffer, position):
    databytes = None
    compmethod = 0
    # Header
    position, magicnum, version, typ, machid, sessid, s_first, s_last = read_header(buffer, position)

    position, compmethod, machtype, datasize, crc16 = read_header_v10p(buffer, position, version)

    temp = buffer[position:]

    if version >= 10:
        if compmethod > 0:
            print('COMPRESSION NOT SUPPORTED')
        else:
            databytes = temp
    else:
        print('VERSION NOT SUPPORTED')

    data_position = 0
    # NB of channels
    data_position, (mcsize,) = unpack(databytes, 'h', data_position)

    assert mcsize == 18
    assert data_position == 2

    # for c in range(mcsize):
    data_position, (code,) = unpack(databytes, 'I', data_position)
    data_position, (size2,) = unpack(databytes, 'h', data_position)

    assert code == 4354
    assert size2 == 1

    # for e in range(size2):
    data_position, ts1, ts2, evcount, t8, rate, gain, offset, mn, mx, dim, second_field = read_event(databytes,
                                                                                                     data_position)

    assert ts1 == 1664241616000
    assert ts2 == 1664259256000
    assert evcount == 441000
    assert t8 == 0
    print(rate)
    assert rate == 40
    assert gain == 0.0199999996
    assert offset == 0
    assert mn == 1.63999999
    assert mx == 11
    assert dim == 'cmH2O'
    assert second_field == False


load_session('../data/63c6e928.001')
