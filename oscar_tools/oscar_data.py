from dataclasses import dataclass, field


@dataclass
class OSCARSessionHeader:
    magicnumber: int = 0
    version: int = 0
    filetype: int  = 0
    deviceid: int = 0
    sessionid: int = 0
    sfirst: int = 0
    slast: int = 0
    compmethod: int = 0
    machtype: int = 0
    datasize: int = 0
    crc16: int = 0


@dataclass
class OSCARSessionEvent:
    ts1: int
    ts2: int
    evcount: int
    t8: int
    rate: float
    gain: float
    offset: float
    mn: float
    mx: float
    len_dim: int
    dim: str
    second_field: bool
    mn2: float
    mx2: float
    data: list[int] = field(default_factory=int)
    data2: list[int] = field(default_factory=int)
    time: list[int] = field(default_factory=int)

@dataclass
class OSCARSessionChannel:
    code: int
    size2: int
    events: list[OSCARSessionEvent] = field(default_factory=OSCARSessionEvent)


@dataclass
class OSCARSessionData:
    mcsize: int
    channels: list[OSCARSessionChannel] = field(default_factory=OSCARSessionChannel)


@dataclass
class OSCARSession:
    header: OSCARSessionHeader = None
    data: OSCARSessionData = None
