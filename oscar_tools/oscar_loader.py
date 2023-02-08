# Test loading with struct python module

import struct
from oscar_tools.oscar_data import OSCARSessionHeader, OSCARSession, \
    OSCARSessionData, OSCARSessionChannel, OSCARSessionEvent


# OSCAR -- Session::LoadEvents(QString filename)
def load_session(filename):
    oscar_session_data = None
    with open(filename, mode='rb') as file:
        data = file.read()
        position = 0
        position, oscar_session_data = read_session(data, position)
    return oscar_session_data


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def unpack(buffer, format, position):
    # DEBUG
    # print('from', position,
    #       'to', position + struct.calcsize(format),
    #       'size', struct.calcsize(format),
    #       'of ', format,
    #       'raw bytes ', buffer[position:position + struct.calcsize(format)].hex(),
    #       'values ', struct.unpack_from(format, buffer, offset=position))
    # p = position
    # # do not display for string reading
    # if 's' not in format:
    #     for e in format:
    #         if e not in ['<', '>']:
    #             print(' element format', e,
    #                   'from', p,
    #                   'to', p + struct.calcsize(e),
    #                   'size', struct.calcsize(e),
    #                   'raw bytes ', buffer[p:p + struct.calcsize(e)].hex(),
    #                   'value', struct.unpack_from(e, buffer, offset=p))
    #             p = p + struct.calcsize(e)
    return position + struct.calcsize(format), struct.unpack_from(format, buffer, offset=position)


def read_header(buffer, position):
    header_data = OSCARSessionHeader()
    position, (magicnum, version, typ, machid, sessid, s_first, s_last) = unpack(buffer, 'IHHIIqq', position)
    header_data.magicnumber = magicnum
    header_data.version = version
    header_data.filetype = typ
    header_data.deviceid = machid
    header_data.sessionid = sessid
    header_data.sfirst = s_first
    header_data.slast = s_last

    if version >= 10:
        position, (compmethod, machtype, datasize, crc16) = unpack(buffer, 'HHIH', position)
        header_data.compmethod = compmethod
        header_data.machtype = machtype
        header_data.datasize = datasize
        header_data.crc16 = crc16
    else:
        print('VERSION NOT SUPPORTED')

    return position, header_data


def read_data(buffer, position):
    # NB of channels
    data_data = OSCARSessionData()
    position, (mcsize,) = unpack(buffer, 'h', position)
    data_data.mcsize = mcsize
    for c in range(mcsize):
        position, channel_data = read_channel_metadata(buffer, position)
        data_data.channels.append(channel_data)
    for c in range(mcsize):
        position, channel_data = read_channel_data(buffer, position, data_data, c)
    return position, data_data


def read_channel_metadata(buffer, position):
    position, (code,) = unpack(buffer, 'I', position)
    position, (size2,) = unpack(buffer, 'h', position)
    channel_data = OSCARSessionChannel()
    # Codes are described in OSCAR/schema.cpp
    channel_data.code = code
    channel_data.size2 = size2
    for evt in range(size2):
        position, event_data = read_event_metadata(buffer, position)
        channel_data.events.append(event_data)
    return position, channel_data

def read_channel_data(buffer, position, data_data, channel_num):
    channel_data = data_data.channels[channel_num]
    for evt_id in range(channel_data.size2):
        event_data = channel_data.events[evt_id]
        # 's' is not correct since it interprets as char
        position, data = unpack(buffer, 'h'*event_data.evcount, position)
        event_data.data = list(data)
        if event_data.second_field:
            position, data2 = unpack(buffer, 'h'*event_data.evcount, position)
            event_data.data2 = list(data2)
        if event_data.t8 != 0:
            position, time_data = unpack(buffer, 'i' * event_data.evcount, position)
            event_data.time = list(time_data)
    return position, channel_data


def read_event_metadata(buffer, position):
    position, (ts1, ts2, evcount, t8, rate, gain, offset, mn, mx, len_dim) = unpack(buffer,
                                                                                    '<qqiBdddddi',
                                                                                    position)
    event_data = OSCARSessionEvent()
    event_data.ts1 = ts1
    event_data.ts2 = ts2
    event_data.evcount = evcount
    event_data.t8 = t8
    event_data.rate = rate
    event_data.gain = gain
    event_data.offset = offset
    event_data.mn = mn
    event_data.mx = mx
    event_data.len_dim = len_dim
    # See QT QDataStream.cpp QDataStream &QDataStream::readBytes(char *&s, uint &l)
    # not totally sure about this but seems to work with signed int length
    if len_dim != -1:
        position, (dim,) = unpack(buffer, str(len_dim)+'s', position)
        event_data.dim = dim.decode('UTF-16-LE')
    else:
        event_data.dim = ''
    position, (second_field,) = unpack(buffer, '?', position)
    event_data.second_field = second_field

    if second_field:
        position, (mn2, mx2) = unpack(buffer, 'ff', position)
        event_data.mn2 = mn2
        event_data.mx2 = mx2

    return position, event_data


def read_session(buffer, position):
    databytes = None
    compmethod = 0
    # Header
    position, oscar_session_header = read_header(buffer, position)

    temp = buffer[position:]

    if oscar_session_header.version >= 10:
        if compmethod > 0:
            print('COMPRESSION NOT SUPPORTED YET')
        else:
            databytes = temp
    else:
        print('VERSION NOT SUPPORTED')

    dataposition = 0
    dataposition, oscar_session_data = read_data(databytes, dataposition)

    oscar_session = OSCARSession()
    oscar_session.header = oscar_session_header
    oscar_session.data = oscar_session_data

    return position, oscar_session


def get_channel_from_code(oscar_session_data, channelID):
    list_result =  [item for item in oscar_session_data.data.channels if item.code == channelID.value]
    if len(list_result) > 0:
        return list_result[0]
    else:
        return list_result
