from unittest import TestCase
from hachoir.stream import StringInputStream, LITTLE_ENDIAN
from oscar_tools.oscar_parser import OscarSessionParser

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
                            'len_dim': -1,
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




class TestOscarSessionParser(TestCase):

    def _test(self, parent):
        for field in parent:
            test_value = find_value_in(field.path, test_values)
            if field.name != 'raw[]' and \
                    test_value is not None and \
                    field.value is not None:
                self.assertEqual(test_value, field.value, field.path + ' is incorrect.')
            if field.is_field_set: self._test(field)

    def _displayTree(self, parent):
        for field in parent:
            print("%s)=%s at (%s)" % (field.path, field.display, field.address))
            if field.is_field_set: self._displayTree(field)

    def test_init(self):
        filename = '../data/63c6e928.001'
        with open(filename, mode='rb') as file:  # b is important -> binary
            data = file.read()
            stream = StringInputStream(data)
            oscar_session = OscarSessionParser(stream)
            self._test(oscar_session)
            self._displayTree(oscar_session)





