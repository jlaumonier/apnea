# Test loading with Hachoir

from hachoir.stream import StringInputStream, LITTLE_ENDIAN
from hachoir.field import FieldSet, Parser, CString, UInt16, UInt32, UInt64, Int64, Int32, UInt8, Float64


class Header(FieldSet):
    def createFields(self):
        yield UInt32(self, "magicnumber", "Magic number")
        yield UInt16(self, "version", "Version")
        yield UInt16(self, "filetype", "File type")
        yield UInt32(self, "deviceid", "Device ID")
        yield UInt32(self, "sessionid", "Session ID")
        yield UInt64(self, "sfirst", "First")
        yield UInt64(self, "slast", "Last")
        version = self['version'].value
        if version >= 10:
            yield UInt16(self, "compmethod", "Compression Method ")
            yield UInt16(self, "machtype", "Device Type ")
            yield UInt32(self, "datasize", "Size of Uncompressed Data")
            yield UInt16(self, "crc16", "CRC16 of Uncompressed Data")


class ChannelList(FieldSet):
    def createFields(self):
        yield UInt32(self, "code", "Channel Code")
        yield UInt16(self, "size2", "number of event list")


class DataByte(FieldSet):

    def createFields(self):
        yield UInt16(self, "mcsize", "number of Device Code lists")


class Event(FieldSet):

    def createFields(self):
        yield Int64(self, "ts1", "")
        yield Int64(self, "ts2", "")
        yield Int32(self, "evcount", "")
        yield UInt8(self, "t8", "")
        yield Float64(self, "rate", "")
        yield Float64(self, "gain", "")
        yield Float64(self, "offset", "")
        yield Float64(self, "mn", "")
        yield Float64(self, "mx", "")
        yield CString(self, "dim", "")
        yield UInt8(self, "second_field")


class OscarSessionParser(Parser):
    endian = LITTLE_ENDIAN

    PARSER_TAGS = {
        "id": "oscar_session",
        "category": "misc",
        "file_ext": ("001"),
        "description": "Oscar session data",
    }

    def createFields(self):
        header = Header(self, "header")
        yield header
        data = DataByte(self, "data")
        yield data
        mcsize = data['mcsize'].value
        for i in range(mcsize):
            channellist = ChannelList(self, 'channellist[]', 'Canal list')
            yield channellist
            size2 = channellist['size2'].value
            for j in range(size2):
                yield Event(self, 'event[]')


test_values = {'header': {'magicnumber': 3341948587,
                          'version': 10,
                          'filetype': 1,
                          'deviceid': 2327,
                          'sessionid': 1673980200,
                          'sfirst': 1673980203000,
                          'slast': 1673981224000,
                          'compmethod': 0,
                          'machtype': 1,
                          'datasize': 114270,
                          'crc16': 0},
               'data': {'mcsize': 17},
               'channellist[0]': {'code': 4362,
                                  'size2': 1},
               'event[0]': {'ts1': 1673980205440,
                            'ts2': 1673981218560,
                            'evcount': 327,
                            't8': 1,
                            'rate': 0,
                            'gain': 0.019999999552965164,
                            'offset': 0.0,
                            'mn': 0.47999998927116394,
                            'mx': 2.4600000381469727,
                            'dim': "",
                            'second_field': False}
               }


# https://stackoverflow.com/questions/31033549/nested-dictionary-value-from-key-path
# Thanks Tomerikoo
def find_value_in(element, tv_dict):
    keys = element.split('/')
    keys = keys[1:]
    rv = tv_dict
    for key in keys:
        if rv is not None and type(rv) is dict and key in rv:
            rv = rv[key]
        else:
            rv = None
    return rv


def test(parent):
    for field in parent:
        test_value = find_value_in(field.path, test_values)
        if field.name != 'raw[]' and \
                test_value is not None and \
                field.value is not None:
            if test_value != field.value:
                print(field.path, find_value_in(field.path, test_values), ' != ', field.value)
            else:
                print(field.path, find_value_in(field.path, test_values), ' == ', field.value, ' OK ')
        if field.is_field_set: test(field)


def displayTree(parent):
    for field in parent:
        print("%s)=%s at (%s)" % (field.path, field.address, field.display))
        if field.is_field_set: displayTree(field)


filename = '../data/63c6e928.001'
with open(filename, mode='rb') as file:  # b is important -> binary
    data = file.read()
    stream = StringInputStream(data)
    oscar_session = OscarSessionParser(stream)
    test(oscar_session)
    #displayTree(oscar_session)
