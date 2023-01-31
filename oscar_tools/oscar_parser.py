# Test loading with Hachoir

from hachoir.stream import StringInputStream, LITTLE_ENDIAN
from hachoir.field import FieldSet, Parser, UInt16, UInt32, UInt64, Int64, Int32, UInt8, Float64, String
from oscar_tools.oscar_data import OSCARSessionHeader, OSCARSession

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

    def toOSCARStruct(self):
        result = OSCARSessionHeader()
        result.magicnumber = self['magicnumber'].value
        result.version = self['version'].value
        result.filetype = self['filetype'].value
        result.deviceid = self['deviceid'].value
        result.sessionid = self['sessionid'].value
        result.sfirst = self['sfirst'].value
        result.slast = self['slast'].value
        return result


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
        yield Int32(self, "len_dim", "")
        # Find this format on an old forum : https://www.qtcentre.org/threads/39660-Read-QString-from-file
        # QString : no documentation on qt has been found yet... thanks QT !
        # * If the string is null: 0xFFFFFFFF (quint32)
        # * Otherwise: The string length in bytes (quint32) followed by the data in UTF-16
        len_dim = self['len_dim'].value
        if len_dim != -1:
            yield String(self, "dim", len_dim, charset="UTF-16")
        yield UInt8(self, "second_field")
        second_field = self['second_field'].value
        if second_field is True:
            yield Float64(self, "mn2", "")
            yield Float64(self, "mx2", "")



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
        mcorder = []
        sizevec = []
        for i in range(mcsize):
            channellist = ChannelList(self, 'channellist[]', 'Canal list')
            yield channellist
            code = channellist['code'].value
            size2 = channellist['size2'].value
            mcorder.append(code)
            sizevec.append(size2)
            for j in range(size2):
                yield Event(self, 'event[]')


        # for i in range(mcsize):
        #     code = mcorder[i]
        #     size2 = sizevec[i]
        #     for j in range(size2):

    def toOSCARStruct(self):
        result = OSCARSession()
        result.header = self['header'].toOSCARStruct()
        result.data = self['data'].toOSCARStruct()
        return result







