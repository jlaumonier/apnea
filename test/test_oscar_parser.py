from unittest import TestCase
from dataclasses import asdict
from oscar_tools.oscar_loader import read_session

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
               'data': {'mcsize': 17,
                        'channels': [{'code': 4362,
                                      'size2': 1,
                                      'events': [{'ts1': 1673980205440,
                                                  'ts2': 1673981218560,
                                                  'evcount': 327,
                                                  't8': 1,
                                                  'rate': 0.0,
                                                  'gain': 0.019999999552965164,
                                                  'offset': 0.0,
                                                  'mn': 0.47999998927116394,
                                                  'mx': 2.4600000381469727,
                                                  'len_dim': -1,
                                                  'dim': "",
                                                  'second_field': False,
                                                  'mn2': 0.0,
                                                  'mx2': 0.0,
                                                  'data': [],
                                                  'data2': [],
                                                  'time': []}]
                                      }
                                     ]
                        }
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

    def _test(self, oscar_session_data):
        oscar_session_data_dict = asdict(oscar_session_data)
        self.maxDiff = None
        self.assertDictEqual(test_values, oscar_session_data_dict)

    def test_init(self):
        filename = '../data/63c6e928.001'
        with open(filename, mode='rb') as file:  # b is important -> binary
            data = file.read()
            position = 0
            position, oscar_session_data = read_session(data, position)
            self._test(oscar_session_data)
